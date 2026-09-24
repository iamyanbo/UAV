"""Bounded asynchronous frozen V-JEPA worker shared by replay and live control."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import torch
from runtime_capacity import slow_worker,slow_command
from visual_encoder import FrozenVideoEncoder,causal_indices


class AsyncVideoFeatures:
    def __init__(self,episode_id,output):
        self.episode_id=episode_id;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.model=FrozenVideoEncoder('/upstream/vjepa2','/models/vjepa2-vitl.pt')
        self.cuda_stream=torch.cuda.Stream()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='frozen-video',initializer=slow_worker)
        self.history=deque();self.pending=None;self.last_submitted_ns=-1;self.version=0
        self.unavailable=0;self.busy=0;self.completed=[]
        self.log=(self.output/'outputs.jsonl').open('x')

    def _encode(self,selected,version):
        started=time.monotonic();row=selected[-1][0]
        with torch.cuda.stream(self.cuda_stream):
            tokens=self.model([rgb for _,rgb in selected])[0].detach().cpu().float()
        path=self.output/f'features-{version:06d}.pt'
        torch.save(dict(tokens=tokens,episode_id=self.episode_id,
            source_frame_ids=[r['frame_id'] for r,_ in selected],
            source_sim_ns=[r['sim_ns'] for r,_ in selected]),path)
        receipt=dict(component='video',episode_id=self.episode_id,version=version,
            observation_ns=row['sim_ns'],source_frame_id=row['frame_id'],rgb_sha256=row['rgb_sha256'],
            feature_path=path.name,feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            available_monotonic=time.monotonic(),processing_seconds=time.monotonic()-started,
            encoder_checkpoint_sha256='5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012')
        self.log.write(json.dumps(receipt)+'\n');self.log.flush()
        self.completed.append(receipt)
        return dict(receipt,tokens=tokens)

    def poll(self):
        if self.pending is not None and self.pending.done():
            result=self.pending.result();self.pending=None;return result
        return None

    def observe(self,row,rgb):
        if row['episode_id']!=self.episode_id:raise ValueError('Cross-episode video request')
        self.history.append((dict(row),rgb))
        while len(self.history)>1 and self.history[1][0]['sim_ns']<row['sim_ns']-3200000000:self.history.popleft()
        if row['sim_ns']-self.last_submitted_ns<200000000:return
        if self.pending is not None:self.busy+=1;return
        try:indices=causal_indices([r['sim_ns'] for r,_ in self.history],row['sim_ns'])
        except ValueError:self.unavailable+=1;return
        selected=[self.history[i] for i in indices];self.version+=1
        self.pending=self.executor.submit(self._encode,selected,self.version)
        self.last_submitted_ns=row['sim_ns']

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=False)
        if self.pending is not None:self.pending.result()
        self.log.close();self.history.clear()
        result=dict(status='completed',accepted=False,episode_id=self.episode_id,
            outputs=len(self.completed),unavailable_windows=self.unavailable,busy_requests=self.busy,
            frozen_encoder=True,publication_clock='actual completion in this replay/live process')
        (self.output/'result.json').write_text(json.dumps(result,indent=2))
        return result
