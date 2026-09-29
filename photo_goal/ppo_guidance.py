"""Frozen Qwen proposal-only service with one global job and bounded RGB memory.

No evaluator geometry, state, cost field, or hidden goal position is accepted.
Raw proposal records and the exact image-region input survive PPO recomputation.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
import threading
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from .contracts import SpatialSnapshot,Subgoal
from .subgoals import QwenProposer

REGIONS=(('left',(0.,.2,.34,.85)),('centre',(.33,.2,.67,.85)),('right',(.66,.2,1.,.85)),('upper',(0.,0.,1.,.3)))


def region_prompt(references):
    return ('Study photo-goal navigation with one fixed forward camera. Image 0 is current RGB, '
        'image 1 is the goal photo; remaining images are historical RGB. Outlined regions are '
        'image evidence, NOT obstacle clearance or metric positions. Propose ONE defensible next '
        'subgoal. If evidence is insufficient, use search or hold with no reference. Gain/lose '
        'height is only an intention; never claim unseen space is safe. Return JSON only: '
        '{"reasoning":"brief evidence", "proposals":[{"intention":"goal|inspect|approach|search|recover|hold",'
        '"target_reference":null,"target_source":"none|image_region","altitude":"maintain|gain|lose",'
        '"confidence":0.0,"horizon_s":5.0}]}. Use only supplied reference IDs. No coordinates, '
        'stop instructions, velocities or commands. Supplied references: '+json.dumps(references))


class RegionProposer(QwenProposer):
    @torch.inference_mode()
    def region_proposal(self,images,references,spatial,frame,stamp):
        prompt=region_prompt(references)
        messages=[dict(role='user',content=[*[dict(type='image',image=i) for i in images],dict(type='text',text=prompt)])]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=images,return_tensors='pt').to('cuda')
        tokens=self.model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
        raw=self.processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        parsed=json.loads(raw)
        if set(parsed)!={'reasoning','proposals'} or not isinstance(parsed['reasoning'],str) or len(parsed['proposals'])!=1:
            raise ValueError('Expected one structured subgoal')
        goal=Subgoal.proposal(parsed['proposals'][0],frame,stamp,spatial)
        if goal.target_source not in ('none','image_region'):raise ValueError('Unsupported metric/map reference')
        return goal,raw


class RGBMemory:
    def __init__(self):self.frames=deque(maxlen=32)

    def observe(self,ident,stamp,rgb,descriptor,path):
        d=np.asarray(descriptor,dtype=float);d=d/max(np.linalg.norm(d),1e-12)
        row=dict(id=ident,stamp=stamp,rgb=rgb,descriptor=d,path=path)
        if not self.frames or stamp-self.frames[-1]['stamp']>=5 or float(d@self.frames[-1]['descriptor'])<.9:
            self.frames.append(row)
        return row

    def select(self,current,goal):
        candidates=[r for r in self.frames if r['id']!=current['id']]
        goal=np.asarray(goal);goal=goal/max(np.linalg.norm(goal),1e-12);chosen=[]
        while candidates and len(chosen)<3:
            row=max(candidates,key=lambda r:float(r['descriptor']@goal)-.5*max([float(r['descriptor']@s['descriptor']) for s in chosen] or [0]))
            chosen.append(row);candidates=[r for r in candidates if r['id']!=row['id']]
        return chosen


def request_context(current,goal_image,history,revision):
    images=[];references=[];regions=[];lookup={}
    for index,row in enumerate([current]+history):
        pixels=np.frombuffer(row['rgb'],np.uint8).reshape(480,640,3)
        image=Image.fromarray(pixels.copy());draw=ImageDraw.Draw(image)
        for name,roi in REGIONS:
            ident=f"f{row['id']}:{name}";coords=tuple(roi[i]*(640 if i%2==0 else 480) for i in range(4))
            draw.rectangle(coords,outline='yellow',width=2);draw.text(coords[:2],ident,fill='yellow')
            references.append(dict(id=ident,image=0 if index==0 else index+1,roi=roi,frame=row['id'],source_s=row['stamp']))
            regions.append((ident,row['id'],row['stamp'],*roi));lookup[ident]=dict(frame_id=row['id'],roi=list(roi),rgb_path=row['path'])
        images.append(image)
    images.insert(1,goal_image.copy())
    spatial=SpatialSnapshot(observed_s=current['stamp'],publication_wall=time.monotonic(),revision=revision,image_regions=tuple(regions)).validate()
    return images,references,spatial,lookup


class Guidance:
    def __init__(self,proposer,log):
        self.proposer=proposer;self.log=log;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='qwen')
        self.pending=None;self.active={};self.last={};self.generation=0;self.epochs={};self.lock=threading.Lock();self.receipts=[]

    def forget(self,worker):
        with self.lock:
            self.epochs[worker]=self.epochs.get(worker,0)+1
            self.active.pop(worker,None);self.last.pop(worker,None)

    def invalidate(self):
        with self.lock:
            self.generation+=1;self.active.clear();self.last.clear()
            if self.pending:self.pending[0].cancel()

    def poll(self,worker,stamp,available):
        with self.lock:
            if self.pending and self.pending[0].done():
                future,owner,generation,epoch,queued=self.pending;self.pending=None
                try:
                    goal,raw,spatial,lookup,started=future.result()
                    row=dict(worker=owner,generation=generation,raw=raw,proposal=asdict(goal),spatial=asdict(spatial),
                        queue_s=started-queued,execution_s=time.monotonic()-started,completed_wall=time.monotonic())
                    if generation==self.generation and epoch==self.epochs.get(owner,0) and time.monotonic()-queued<=5:
                        self.active[owner]=(goal,spatial,lookup,queued)
                        row['fresh_valid']=True
                    else:row['discarded']='obsolete_generation_or_wall_expiry'
                except Exception as error:row=dict(worker=owner,generation=generation,error=str(error),discarded='invalid_result')
                self.receipts.append(row);self.log(row)
            active=self.active.get(worker)
            if active:
                goal,spatial,lookup,queued=active
                target=lookup.get(goal.target_reference)
                if (time.monotonic()-queued<=5 and goal.valid(stamp,spatial) and
                    (target is None or target['frame_id'] in available)):
                    return dict(subgoal=asdict(goal),spatial=asdict(spatial),reference=target,
                        vector=goal.vector(stamp,spatial),generation=self.generation)
                self.active.pop(worker,None)
            return dict(subgoal=None,spatial=asdict(SpatialSnapshot()),reference=None,
                vector=Subgoal().vector(stamp,SpatialSnapshot()),generation=self.generation)

    def request(self,worker,current,goal_image,history):
        with self.lock:
            now=time.monotonic()
            if self.pending or now-self.last.get(worker,-math.inf)<3:return
            self.last[worker]=now;generation=self.generation
            # Copy only bounded observable inputs. A task dictionary never crosses this boundary.
            def run():
                started=time.monotonic()
                images,references,spatial,lookup=request_context(current,goal_image,history,generation)
                goal,raw=self.proposer.region_proposal(images,references,spatial,current['id'],current['stamp'])
                return goal,raw,spatial,lookup,started
            self.pending=(self.pool.submit(run),worker,generation,self.epochs.get(worker,0),now)

    def close(self):
        self.invalidate();self.pool.shutdown(wait=True,cancel_futures=True)
