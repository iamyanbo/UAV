"""One inference owner batches ready observations, with at most 2 ms batch wait."""
from concurrent.futures import Future
from dataclasses import asdict
from collections import deque
import queue
import hashlib
import threading
import time
import numpy as np
import torch
from .contracts import Subgoal,SpatialSnapshot
from .ppo_guidance import RGBMemory


class FeatureBank:
    def __init__(self):
        self.frames={};self.paths={};self.history=deque(maxlen=4);self.memory=RGBMemory();self.next_id=0
        self.goal=None;self.goal_image=None
        self.exercise=None;self.exercise_record=None;self.kind='mission'

    def add(self,raw,path):
        ident=self.next_id;self.next_id+=1;self.frames[ident]=raw.detach().cpu().half();self.paths[ident]=path
        return ident

    def initialize(self,raw,image,path,exercise=None,kind='mission'):
        self.history.clear();self.memory=RGBMemory();self.goal=self.add(raw,path);self.goal_image=image
        self.exercise=exercise;self.exercise_record=None;self.kind=kind

    def context(self,raw,obs,guidance,worker,execution,replace=False):
        ident=self.add(raw,obs['rgb_path'])
        if replace:
            if not self.history or abs(obs['sim_s']-self.history[-1][1])>1e-6:raise ValueError('Boundary timestamp changed')
            self.history.pop()
        elif self.history and obs['sim_s']<=self.history[-1][1]:raise ValueError('Non-increasing frame window')
        self.history.append((ident,obs['sim_s'],obs['preceding_command']))
        current=None
        if guidance:
            current=self.memory.observe(ident,obs['sim_s'],obs['rgb'],raw.float().mean((-1,-2)).cpu().numpy(),obs['rgb_path'])
        record=guidance.poll(worker,obs['sim_s'],self.frames) if guidance and self.kind=='mission' else dict(subgoal=None,spatial=asdict(SpatialSnapshot()),
            reference=None,vector=Subgoal().vector(obs['sim_s'],SpatialSnapshot()),generation=0)
        if self.exercise and self.exercise_record is None:
            ref=f'f{ident}:exercise';roi=self.exercise['roi']
            spatial=SpatialSnapshot(observed_s=obs['sim_s'],image_regions=((ref,ident,obs['sim_s'],*roi),))
            goal=Subgoal.proposal(dict(intention=self.exercise['intention'],altitude=self.exercise['altitude'],
                target_reference=ref,target_source='image_region',confidence=1.,horizon_s=5.),ident,obs['sim_s'],spatial)
            self.exercise_record=(goal,spatial,dict(frame_id=ident,roi=roi,rgb_path=obs['rgb_path']))
        if self.exercise_record:
            goal,spatial,ref=self.exercise_record
            if goal.valid(obs['sim_s'],spatial):record=dict(subgoal=asdict(goal),spatial=asdict(spatial),reference=ref,
                vector=goal.vector(obs['sim_s'],spatial),generation=0,source='offline_visible_execution_exercise')
        if guidance and self.kind=='mission':
            descriptor=self.frames[self.goal].float().mean((-1,-2)).numpy()
            guidance.request(worker,current,self.goal_image,self.memory.select(current,descriptor))
        return dict(observation_schema='observation-context/v3',rgb_pixels_sha256=hashlib.sha256(obs['rgb']).hexdigest(),
            history_ids=[r[0] for r in self.history],stamps=[r[1] for r in self.history],
            preceding=[r[2] for r in self.history],history_rgb=[self.paths[r[0]] for r in self.history],
            goal_id=self.goal,goal_image=self.paths[self.goal],guidance=record,execution=execution)

    def tensors(self,row):
        n=len(row['history_ids']);template=self.frames[row['history_ids'][-1]]
        ref=row['guidance']['reference']
        return dict(history=torch.stack([torch.zeros_like(template)]*(4-n)+[self.frames[i] for i in row['history_ids']]),
            goal=self.frames[row['goal_id']],times=torch.tensor([0.]*(4-n)+[s-row['stamps'][-1] for s in row['stamps']]),
            commands=torch.tensor([[0.]*4]*(4-n)+row['preceding'],dtype=torch.float32),valid=torch.tensor([False]*(4-n)+[True]*n),
            subgoal_vector=torch.tensor(row['guidance']['vector'],dtype=torch.float32),
            reference=self.frames[ref['frame_id']] if ref else torch.zeros_like(template),
            reference_roi=torch.tensor(ref['roi'] if ref else [0,0,1,1],dtype=torch.float32),execution=torch.tensor(row['execution']))

    def trim(self):
        keep={self.goal}|{r[0] for r in self.history}|{r['id'] for r in self.memory.frames}
        if self.exercise_record:keep.add(self.exercise_record[2]['frame_id'])
        self.frames={k:v for k,v in self.frames.items() if k in keep};self.paths={k:v for k,v in self.paths.items() if k in keep}


def batch_for(rows,banks,device):
    tensors=[banks[r['worker_id']].tensors(r) for r in rows]
    result={key:torch.stack([t[key] for t in tensors]).to(device) for key in tensors[0]}
    if 'latent' in rows[0]:
        result.update(latent=torch.tensor([r['latent'] for r in rows],device=device),
            stop=torch.tensor([r['stop'] for r in rows],device=device,dtype=torch.float32))
    return result


class Inference:
    def __init__(self,model,lane,banks,guidance,wait_s=.002):
        self.model=model;self.lane=lane;self.banks=banks;self.guidance=guidance;self.wait_s=wait_s
        self.queue=queue.Queue();self.thread=threading.Thread(target=self.run,daemon=True,name='ppo-inference');self.thread.start()

    def call(self,kind,**args):
        f=Future();self.queue.put((kind,args,f,time.perf_counter()));return f.result(timeout=180)

    def run(self):
        device=next(self.model.parameters()).device
        while True:
            item=self.queue.get()
            if item is None:return
            group=[item];deadline=time.perf_counter()+self.wait_s
            while len(group)<2 and time.perf_counter()<deadline:
                try:
                    other=self.queue.get(timeout=max(0,deadline-time.perf_counter()))
                    if other is None:self.queue.put(None);break
                    group.append(other)
                except queue.Empty:break
            try:
                with torch.inference_mode(),self.lane.fast():
                    stage_start=time.perf_counter()
                    rgb=[]
                    for kind,args,_,_ in group:
                        pixels=np.asarray(args['image']) if kind=='initialize' else np.frombuffer(args['obs']['rgb'],np.uint8).reshape(480,640,3)
                        rgb.append(torch.from_numpy(pixels.copy()).permute(2,0,1))
                    raw=self.model.encode_backbone(torch.stack(rgb).to(device))
                    torch.cuda.synchronize();encoded=time.perf_counter()
                    ready=[]
                    for i,(kind,args,f,queued) in enumerate(group):
                        bank=self.banks[args['worker']]
                        if kind=='initialize':bank.initialize(raw[i],args['image'],args['path'],args.get('exercise'),args.get('task_kind','mission'));f.set_result(None)
                        else:
                            context=bank.context(raw[i],args['obs'],self.guidance,args['worker'],args['execution'],args.get('replace',False))
                            context['worker_id']=args['worker'];ready.append((args,f,queued,context))
                    contextualized=time.perf_counter()
                    if ready:
                        prepared=batch_for([r[3] for r in ready],self.banks,device)
                        # With no guidance all reference descriptors are multiplied
                        # by zero; avoid projecting a fabricated reference image.
                        if not self.guidance and not any(self.banks[r[0]['worker']].exercise for r in ready):
                            prepared['reference']=None
                        normal,stop,value=self.model(**prepared)
                        # Evaluation uses the same stochastic action policy with a
                        # recorded RNG state; no unrelated stop threshold policy.
                        latent=normal.sample();stops=stop.sample();logp=normal.log_prob(latent).sum(-1)+stop.log_prob(stops)
                        for i,(args,f,queued,context) in enumerate(ready):
                            f.set_result(dict(context=context,latent=latent[i].cpu().tolist(),stop=float(stops[i]),
                                stop_probability=float(stop.probs[i]),value=float(value[i]),logprob=float(logp[i]),
                                decision_s=time.perf_counter()-queued,
                                encode_s=encoded-stage_start,context_s=contextualized-encoded,
                                policy_s=time.perf_counter()-contextualized))
            except BaseException as error:
                for _,_,future,_ in group:
                    if not future.done():future.set_exception(error)

    def close(self):self.queue.put(None);self.thread.join(timeout=180)
