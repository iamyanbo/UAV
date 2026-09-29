"""One pending RGB job: native tracking, metric scale, bounded Gaussian work.

No Splat-SLAM fallback. The native dependency is mandatory, even for ablations.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib
import time
import threading
import numpy as np
import torch
from .vision.metric_depth import MetricDepth
from .contracts import SpatialSnapshot

PIN='f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac'


class BackgroundPerception:
    def __init__(self, settings, vision, lane):
        from .common import digest
        if settings.get('upstream_commit')!=PIN:raise ValueError('Wrong Photo-SLAM source pin')
        if not {settings[k] for k in ('vocabulary','tracking','mapping')}<=set(settings['artifacts']):
            raise ValueError('Missing native asset identities')
        for name,sha in settings['artifacts'].items():
            if digest(name)!=sha:raise ValueError('Changed native Photo-SLAM asset: '+name)
        native=importlib.import_module('photo_slam_live')
        if native.upstream_commit!=PIN or native.bridge_schema!='photo-slam-live/v1':
            raise ValueError('Incompatible Photo-SLAM native adapter')
        if str(native.__file__) not in settings['artifacts']:raise ValueError('Native bridge binary has no identity')
        self.native=native.Session(settings['vocabulary'],settings['tracking'],settings['mapping'])
        self.depth=MetricDepth(vision);self.lane=lane
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='photo-slam')
        self.depth_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='metric-depth')
        self.depth_pending=None;self.lock=threading.Lock();self.speed=None
        self.pending=None;self.snapshot=SpatialSnapshot();self.local=None
        self.scales=deque(maxlen=8);self.previous_pose=None;self.revision=-1
        self.keyframes={};self.images=deque(maxlen=128);self.error=None

    def observe(self, metadata, rgb):
        if self.pending is not None and self.pending.done():
            try:
                self.pending.result();self.error=None
            except Exception as exc:
                self.error=type(exc).__name__+': '+str(exc)
                with self.lock:
                    self.speed=None
                    if self.local:self.local=dict(self.local,speed=None)
                    self.snapshot=replace(self.snapshot,tracking=False,scale_status='lost',
                                          meters_per_unit=None,scale_relative_sigma=None)
            self.pending=None
        if self.depth_pending is not None and self.depth_pending.done():
            try:self.depth_pending.result()
            except Exception as exc:
                self.error='metric-depth: '+str(exc)
                with self.lock:self.local=None
            self.depth_pending=None
        if self.depth_pending is None:
            self.depth_frame=metadata['frame_id']
            self.depth_pending=self.depth_pool.submit(self._depth,dict(metadata),rgb.copy(),time.monotonic())
        if self.pending is None:
            depth_job=self.depth_pending if self.depth_frame==metadata['frame_id'] else None
            self.pending=self.pool.submit(self._work,dict(metadata),rgb.copy(),time.monotonic(),depth_job)
        with self.lock:return self.snapshot,self.local,dict(self.keyframes)

    def _depth(self,metadata,rgb,queued):
        started=time.monotonic()
        with self.lane.background():depth=self.depth(rgb,metadata['calibration']).numpy()
        with self.lock:
            # Publish independently of tracking and Gaussian work. Depth alone
            # does not provide metric velocity or certify arrival.
            speed=self.speed if 0<=metadata['sim_ns']/1e9-self.snapshot.observed_s<=.25 else None
            self.local=dict(depth=depth,calibration=metadata['calibration'],observed_s=metadata['sim_ns']/1e9,
                source_wall=metadata['received_monotonic'],speed=speed,revision=self.snapshot.revision,
                queue_s=started-queued,work_s=time.monotonic()-started,depth_and_queue_s=time.monotonic()-queued)
        return depth

    def _work(self, metadata, rgb, queued, depth_job):
        started=time.monotonic();now=metadata['sim_ns']/1e9
        # Native tracking remains available after Gaussian work is suspended.
        raw=self.native.track(rgb,now,metadata['frame_id'],self.lane.slow_admitted)
        tracking_s=time.monotonic()-started
        if raw['revision']!=self.revision:
            self.scales.clear();self.previous_pose=None;self.revision=raw['revision']
        # Only identical-frame depth establishes scale. It has already been
        # published for safety before this future resolves; Gaussian work follows.
        depth=depth_job.result() if depth_job is not None else None
        depth_finished=time.monotonic()
        ratios=[]
        for u,v,z in raw['depth_samples'] if depth is not None else ():
            if 0<=u<640 and 0<=v<480 and z>0:
                metric=float(depth[int(v),int(u)])
                if .2<metric<80:ratios.append(np.log(metric/z))
        if raw['tracking'] and len(ratios)>=24:
            self.scales.append((now,float(np.median(ratios))))
        elif not raw['tracking']:
            self.scales.clear();self.previous_pose=None
        recent=[v for stamp,v in self.scales if now-stamp<2]
        factor=sigma=None;status='unknown'
        if len(recent)>=3:
            factor=float(np.exp(np.median(recent)))
            sigma=max(.15,float(np.std(recent)))
            status='metric' if sigma<=.35 else 'lost'
        speed=None;pose=np.asarray(raw['pose']).reshape(4,4)
        if status=='metric' and self.previous_pose is not None:
            stamp,old=self.previous_pose
            if 0<now-stamp<=.5:speed=float(np.linalg.norm(pose[:3,3]-old[:3,3])*factor/(now-stamp))
        self.previous_pose=(now,pose) if raw['tracking'] else None
        self.images.append((metadata['frame_id'],rgb))
        cache=dict(self.images)
        keys={row[0]:cache[row[1]] for row in raw['keyframes'] if row[1] in cache}
        snapshot=SpatialSnapshot(observed_s=now,publication_wall=time.monotonic(),revision=raw['revision'],
            camera_to_body=tuple(np.asarray(metadata['calibration']['camera_to_body_rotation']).flatten()),
            tracking=raw['tracking'],poses=(tuple(raw['pose']),) if raw['tracking'] else (),scale_status=status,
            meters_per_unit=factor,scale_relative_sigma=sigma,
            keyframes=tuple(tuple(row) for row in raw['keyframes']),geometry=tuple(tuple(row) for row in raw['geometry'])).validate()
        with self.lock:
            self.snapshot=snapshot;self.keyframes=keys;self.speed=speed
            if self.local and abs(self.local['observed_s']-now)<=.25:
                self.local=dict(self.local,speed=speed,tracking_s=tracking_s)
        if self.lane.slow_admitted:
            try:
                with self.lane.slow():self.native.map_tick()
            except RuntimeError as exc:
                if str(exc)!='prediction_slice_budget_exceeded':raise
        with self.lock:
            if self.local:self.local=dict(self.local,mapping_and_export_s=time.monotonic()-depth_finished)

    def close(self):
        self.depth_pool.shutdown(wait=True,cancel_futures=True)
        self.pool.shutdown(wait=True,cancel_futures=True);self.native.close()
