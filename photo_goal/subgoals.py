"""Qwen proposals and asynchronous, actor-driven assessment. No commands here."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import wraps
import json
import math
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from .contracts import Subgoal, INTENTIONS, ALTITUDES, STEP_S
from .temporal import rollout


def score_components(predicted):
    reached=(predicted['goal'].sigmoid()[0]>=.8)&(predicted['visibility'].sigmoid()[0]>=.7)
    indices=reached.nonzero().flatten()
    duration=STEP_S*(int(indices[0])+1) if len(indices) else 4.
    return [float(predicted['collision'].sigmoid().amax()),
            float(predicted['goal'].sigmoid()[0,-1]-predicted['goal'].sigmoid()[0,0]),
            float(predicted['progress'][0].sum()),duration,float(predicted['value'][0,-1])]


def prompt(spatial):
    return ('Propose at most four photo-goal navigation subgoals for a fixed forward camera. '
        'Images are current RGB, goal photograph, annotated coarse overhead RGB, then up to three keyframes. '
        'map_lookup gives reference IDs, full raster pixel positions and goal/current hypothesis roles. '
        'The overhead map is a hypothesis and cannot establish obstacle clearance. '
        'Gain/lose height is an intention, not clearance to ascend/descend. '
        'Return JSON only: {"reasoning":string,"proposals":[{"intention":string,'
        '"target_reference":string|null,"target_source":"none"|"map"|"keyframe"|"geometry",'
        '"altitude":"maintain"|"gain"|"lose","confidence":number,"horizon_s":number}]}. '
        'No coordinates, waypoints, stop commands or vehicle velocities. '
        'Horizon is greater than zero and at most five seconds. '
        f'Intentions: {INTENTIONS}. References and spatial uncertainty: '+
        json.dumps(asdict(spatial),allow_nan=False))


class QwenProposer:
    def __init__(self, lane, adapter=None, path='/models/qwen2.5-vl-3b'):
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        from .vision.configurator import install_lora
        from .common import digest
        self.base_identity={p.name:digest(p) for p in sorted(Path(path).iterdir()) if p.suffix in ('.json','.safetensors')}
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,
            min_pixels=64*28*28,max_pixels=256*28*28)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
        self.modules=install_lora(self.model)
        if adapter:
            saved=torch.load(adapter,map_location='cpu',weights_only=True)
            if saved.get('schema')!='subgoal-qwen/v3' or saved['modules']!=self.modules:
                raise ValueError('Qwen adapter uses incompatible proposal contract')
            if saved.get('base_identity')!=self.base_identity:raise ValueError('Qwen base weights/tokenizer changed')
            params=dict(self.model.named_parameters())
            with torch.no_grad():
                for name,value in saved['adapter'].items():
                    if name not in params or not name.endswith(('.a','.b')):raise ValueError('Unexpected adapter parameter')
                    params[name].copy_(value)
        self.model.eval().requires_grad_(False)
        self.lane=lane
        # Each prefill/decode forward yields the GPU lane, including CUDA work.
        # An oversized prefill disables future slow work; it cannot be preempted.
        forward=self.model.forward
        @wraps(forward)
        def bounded_forward(*args,**kwargs):
            with lane.slow():return forward(*args,**kwargs)
        self.model.forward=bounded_forward

    def inputs(self, images, spatial, answer=None):
        messages=[dict(role='user',content=[*[dict(type='image',image=im) for im in images],
                                         dict(type='text',text=prompt(spatial))])]
        if answer is not None:messages.append(dict(role='assistant',content=answer))
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=answer is None)
        return self.processor(text=[text],images=images,return_tensors='pt').to('cuda')

    @torch.inference_mode()
    def propose(self, images, context):
        inputs=self.inputs(images,context.spatial)
        generated=self.model.generate(**inputs,max_new_tokens=384,do_sample=False,use_cache=True)
        raw=self.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        parsed=json.loads(raw)
        if set(parsed)!={'reasoning','proposals'} or not isinstance(parsed['reasoning'],str):
            raise ValueError('Malformed proposal envelope')
        rows=parsed['proposals']
        if not isinstance(rows,list) or not 1<=len(rows)<=4:raise ValueError('Proposal count outside [1,4]')
        goals=[Subgoal.proposal(row,context.frame_ids[-1],context.timestamps[-1],context.spatial) for row in rows]
        return goals,raw


class Mode2:
    def __init__(self, model, lane, variant, latest, calibration, adapter=None):
        self.model=model;self.lane=lane;self.variant=variant;self.latest=latest
        self.calibration=calibration
        self.qwen=QwenProposer(lane,adapter) if variant!='mode1' else None
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='subgoals')
        self.pending=None;self.active=None;self.last_request=-math.inf;self.receipt=None
        self.completion_key=None;self.completion_pose=None;self.completion_count=0

    def completed(self,context,references,match,arrival,speed):
        g=self.active;s=context.spatial
        if g is None or not g.valid(context.timestamps[-1],s):return False
        if g!=self.completion_key:
            self.completion_key=g;self.completion_count=0
            self.completion_pose=s.poses[-1] if s.poses else None
        supported=False
        if g.intention in ('goal','approach') and g.altitude=='maintain':
            supported=min(match,arrival)>=self.calibration['threshold'] and speed is not None and speed<.5
        if s.tracking and s.scale_status=='metric' and s.poses and (s.scale_relative_sigma or 1.)<=.35:
            pose=np.asarray(s.poses[-1]).reshape(4,4)
            if g.target_source=='geometry':
                point=next((r for r in s.geometry if r[0]==g.target_reference and r[4]>=2),None)
                supported=bool(point and np.linalg.norm(np.asarray(point[1:4])-pose[:3,3])*s.meters_per_unit<2.)
            if self.completion_pose and s.camera_to_body and g.altitude!='maintain':
                initial=np.asarray(self.completion_pose).reshape(4,4)
                body=np.asarray(s.camera_to_body).reshape(3,3)@initial[:3,:3].T@(pose[:3,3]-initial[:3,3])*s.meters_per_unit
                supported=body[2]<-1. if g.altitude=='gain' else body[2]>1.
        self.completion_count=self.completion_count+1 if supported else 0
        if self.completion_count<3:return False
        self.active=None;return True

    def update(self, context, images, triggered=False):
        now=context.timestamps[-1]
        if self.active and not self.active.valid(now,context.spatial):self.active=None
        if self.pending is not None and self.pending.done():
            try:
                selected,receipt=self.pending.result();self.receipt=receipt
                self.active=selected if selected and selected.valid(now,context.spatial) else None
            except Exception as exc:
                self.receipt=dict(error=type(exc).__name__+': '+str(exc));self.active=None
            self.pending=None
        if (self.qwen and self.pending is None and self.lane.slow_admitted and
                now-self.last_request>=3 and (self.active is None or triggered)):
            self.last_request=now
            self.pending=self.pool.submit(self._work,context,images,time.monotonic())
        return self.active

    def _work(self, request, images, queued):
        started=time.monotonic();goals,raw=self.qwen.propose(images,request)
        # Qwen can be slow: initialize imagination from current observations,
        # never from the observation that started language generation.
        context,references=self.latest()
        if context is None:return None,dict(discarded='no_current_context',raw=raw)
        goals=[g for g in goals if g.valid(context.timestamps[-1],context.spatial)]
        scores=[]
        for goal in goals:
            if self.variant=='mode1_vlm':
                scores.append(goal.confidence);continue
            predicted=rollout(self.model,context,goal,references,self.lane.slow)
            with self.lane.slow():
                components=score_components(predicted)
            scales=self.calibration['score_scales'];weights=self.calibration['score_weights']
            if len(scales)!=5 or len(weights)!=5 or any(not math.isfinite(s) or s<=0 for s in scales) or any(not math.isfinite(w) for w in weights):
                raise ValueError('Validation score calibration required')
            scores.append(sum(v/s*w for v,s,w in zip(components,scales,weights)))
        current,_=self.latest()
        eligible=[(s,g) for s,g in zip(scores,goals) if math.isfinite(s) and g.valid(current.timestamps[-1],current.spatial)]
        selected=max(eligible,key=lambda p:p[0])[1] if eligible else None
        return selected,dict(raw=raw,scores=scores,proposals=[asdict(g) for g in goals],
            rollout_semantics='unfiltered actor; independent real safety excluded',
            source_frame=request.frame_ids[-1],assessment_frame=context.frame_ids[-1],
            queue_s=started-queued,execution_s=time.monotonic()-started,
            selected=asdict(selected) if selected else None)

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
