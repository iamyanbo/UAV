"""Offline dataset construction; this module may read evaluator labels.

All targets live under `labels`; inference never imports this module. Image
windows are indexed, not copied into a second unbounded video archive.
"""
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib
import numpy as np
import torch
from goal_io import load_goal
from episode_store import verified_rgb_storage
from action_intervals import executed_slots
from .common import read, write, digest, config, reserve_memory
from .maps import MapPrior


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rgb_at(episode,row):
    with (Path(episode)/'observations/rgb.zlib').open('rb') as stream:
        stream.seek(row['offset']);size=struct.unpack('!I',stream.read(4))[0]
        if size!=row['compressed_bytes'] or size>640*480*3+1024:raise ValueError('Bad RGB index')
        raw=zlib.decompress(stream.read(size))
    if len(raw)!=640*480*3 or hashlib.sha256(raw).hexdigest()!=row['rgb_sha256']:raise ValueError('Changed RGB frame')
    return raw


def yaw(q):
    x,y,z,w=q
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def build_dataset(registry_path,flight_root,output):
    registry=read(registry_path);scene_by_id={s['scene_id']:s for s in registry['scenes']}
    episodes=[];windows=[];audit=[];goals_positive=goals_negative=0
    for result_path in sorted(Path(flight_root).rglob('episode/result.json')):
        episode=result_path.parent;result=read(result_path)
        scene_id=result.get('scene_id')
        if scene_id not in scene_by_id:
            audit.append(dict(episode=str(episode),reason='unregistered_scene'));continue
        scene=scene_by_id[scene_id]
        if scene['split']=='test':continue
        if result.get('split')!=scene['split']:raise ValueError('Scene/episode split mismatch')
        storage=verified_rgb_storage(episode/'observations')
        rows=lines(episode/'observations/frames.jsonl');labels=lines(episode/'training_labels/frames.jsonl')
        selected={}
        for label in labels:
            if abs(label['observation_label_skew_seconds'])<=.1:
                old=selected.get(label['frame_id'])
                if old is None or abs(label['observation_label_skew_seconds'])<abs(old['observation_label_skew_seconds']):selected[label['frame_id']]=label
        task=read(episode/'evaluator_labels/episode.json');goal=np.asarray(task['goal_ned_m'])
        correction_path=episode/'photo-controller/runtime/decisions.jsonl'
        corrections={r['frame_id']:r for r in lines(correction_path) if r.get('expert_observation_conditioned') and 'subgoal_body' in r} if correction_path.exists() else {}
        prior=MapPrior(scene['map']);commands=lines(episode/'training_labels/commands.jsonl')
        index=len(episodes)
        episodes.append(dict(path=str(episode.resolve()),scene_id=scene_id,split=scene['split'],map=scene['map'],
            map_sha256=prior.identity,goal_sha256=digest(episode/'goal/goal.json'),result_sha256=digest(result_path),
            labels_sha256=digest(episode/'training_labels/frames.jsonl'),index_sha256=digest(episode/'observations/frames.jsonl'),
            commands_sha256=digest(episode/'training_labels/commands.jsonl'),rgb_sha256=storage['stream_sha256'],
            corrections_sha256=digest(correction_path) if correction_path.exists() else None))
        times=[r['sim_ns'] for r in rows];previous=-math.inf
        for i,row in enumerate(rows):
            label=selected.get(row['frame_id'])
            if label is None or row['sim_ns']-previous<200_000_000:continue
            previous=row['sim_ns'];position=np.asarray(label['true_position_ned_m'])
            surface=float(prior.height(position[:2]))
            if not math.isfinite(surface):continue
            tile=int(np.linalg.norm(prior.tiles-position[:2],axis=1).argmin())
            horizontal=np.linalg.norm(position[:2]-goal[:2]);vertical=abs(position[2]-goal[2])
            near=bool(horizontal<=3 and vertical<=2)
            valid=near or horizontal>=12 or vertical>=6
            stop=near and np.linalg.norm(label['true_velocity_ned_mps'])<.5
            if scene['split']=='train':goals_positive+=int(near);goals_negative+=int(valid and not near)
            path=np.asarray(task['reference_path_ned_m']);closest=int(np.linalg.norm(path-position,axis=1).argmin())
            target=path[min(closest+1,len(path)-1)]-position
            heading=yaw(label['true_quaternion_xyzw']);c,s=math.cos(heading),math.sin(heading)
            body=np.array([c*target[0]+s*target[1],-s*target[0]+c*target[1],target[2]])
            slots=executed_slots(commands,row['sim_ns'])
            correction=corrections.get(row['frame_id'])
            if correction is not None:body=np.asarray(correction['subgoal_body'])
            actions=[];future=[];future_positions=[];future_goal=[];future_collision=[]
            for step in range(20):
                stamp=row['sim_ns']+int((step+1)*200_000_000)
                j=bisect_right(times,stamp)-1
                slot=executed_slots(commands,row['sim_ns']+step*200_000_000)
                target_label=selected.get(rows[j]['frame_id']) if j>=0 else None
                if j<=i or stamp-times[j]>100_000_000 or not all(slot['valid']) or target_label is None:break
                future.append(j);actions.append(slot['values'])
                p=np.asarray(target_label['true_position_ned_m'])-position
                future_positions.append([c*p[0]+s*p[1],-s*p[0]+c*p[1],p[2]])
                future_goal.append(float(np.linalg.norm(np.asarray(target_label['true_position_ned_m'])[:2]-goal[:2])<=3 and abs(target_label['true_position_ned_m'][2]-goal[2])<=2))
                future_collision.append(float(target_label['airsim_collision'] or target_label['geometry_collision']))
                if future_collision[-1]:break
            windows.append(dict(episode=index,frame=i,tile=tile,split=scene['split'],future=future,actions=actions,
                labels=dict(offset=(position[:2]-prior.tiles[tile]).tolist(),above_surface=float(surface-position[2]),
                    yaw=[math.sin(heading),math.cos(heading)],near_goal=near,goal_valid=valid,arrival=stop,
                    subgoal_body=body.tolist(),command=correction['teacher_command'] if correction is not None else None if slots['values'] is None else slots['values'][0],
                    policy_valid=bool((correction is not None or result['controller_kind']=='privileged_shortest_path_expert' and all(slots['valid'])) and not label['airsim_collision'] and not label['geometry_collision']),
                    future_position=future_positions,future_goal=future_goal,future_collision=future_collision)))
    write(output,dict(schema='photo-map-dataset/v1',registry_sha256=digest(registry_path),episodes=episodes,windows=windows,
          audit=audit,training_goal_positives=goals_positive,training_goal_negatives=goals_negative,
          action_semantics='post-safety dispatch intervals, not measured actuator application',
          purpose='perception, subgoal execution and transition learning; no omniscient search imitation'))


class Dataset:
    def __init__(self,path,stage,split='train',teacher_root=None):
        self.path=Path(path);self.spec=read(path);self.stage=stage;self.teacher_root=Path(teacher_root) if teacher_root else None
        if self.spec['schema']!='photo-map-dataset/v1':raise ValueError('New dataset schema required')
        if split not in ('train','validation'):raise ValueError('Sealed test data cannot train or tune models')
        self.windows=[w for w in self.spec['windows'] if w['split']==split and
            (stage!='goal' or w['labels']['goal_valid']) and (stage!='policy' or w['labels']['policy_valid']) and
            (stage!='world' or len(w['future'])>=1)]
        if stage=='world':
            eligible=[]
            for window in self.windows:
                ep=self.spec['episodes'][window['episode']]
                key=hashlib.sha256((ep['path']+ep['rgb_sha256']).encode()).hexdigest()[:20]
                if self.teacher_root is None or not (self.teacher_root/(key+'.pt')).exists():
                    raise ValueError('Encode V-JEPA teacher targets before world training')
                artifact=self.teacher(key)
                if str(window['future'][0]) in artifact['features']:eligible.append(window)
            self.windows=eligible
        if not self.windows:raise ValueError('No qualified '+stage+' windows for '+split)
        if stage=='goal' and split=='train':
            if min(self.spec['training_goal_positives'],self.spec['training_goal_negatives'])<100:
                raise ValueError('Goal stage needs at least 100 positive and 100 negative examples')
        for ep in self.spec['episodes']:
            if ep['split']=='test':raise ValueError('Sealed test episode in training dataset')
            root=Path(ep['path'])
            for name,key in [('result.json','result_sha256'),('goal/goal.json','goal_sha256'),('training_labels/frames.jsonl','labels_sha256'),
                             ('observations/frames.jsonl','index_sha256'),('training_labels/commands.jsonl','commands_sha256'),('observations/rgb.zlib','rgb_sha256')]:
                if digest(root/name)!=ep[key]:raise ValueError('Dataset source changed: '+str(root/name))
            corrections=root/'photo-controller/runtime/decisions.jsonl'
            if (digest(corrections) if corrections.exists() else None)!=ep.get('corrections_sha256'):
                raise ValueError('Correction labels changed after dataset construction')

    def __len__(self):return len(self.windows)

    @lru_cache(maxsize=2)
    def teacher(self,key):
        return torch.load(self.teacher_root/(key+'.pt'),map_location='cpu',weights_only=True)

    @lru_cache(maxsize=8)
    def episode(self,index):
        ep=self.spec['episodes'][index];prior=MapPrior(ep['map'])
        if prior.identity!=ep['map_sha256']:raise ValueError('Dataset map changed')
        return ep,lines(Path(ep['path'])/'observations/frames.jsonl'),load_goal(Path(ep['path'])/'goal'),prior

    def get(self,index):
        window=self.windows[index];ep,rows,goal,prior=self.episode(window['episode'])
        def tensor(raw):return torch.from_numpy(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()).permute(2,0,1)
        current=tensor(rgb_at(ep['path'],rows[window['frame']]))
        goals=torch.stack([tensor(raw) for raw in goal.rgb_views])
        tile=torch.from_numpy(prior.tile(window['tile'])).permute(2,0,1)
        distances=np.linalg.norm(prior.tiles-prior.tiles[window['tile']],axis=1)
        negative=int(distances.argmax())
        if len(prior.tiles)<2:raise ValueError('Localization training needs multiple map tiles')
        if self.stage=='localization' and distances[negative]<160:
            raise ValueError('Localization negatives require nonoverlapping map tiles')
        negative_tile=torch.from_numpy(prior.tile(negative)).permute(2,0,1)
        value=dict(current=current,goals=goals,tile=tile,negative_tile=negative_tile,labels=window['labels'],
                   actions=torch.tensor(window['actions'],dtype=torch.float32))
        if self.stage=='world':
            times=[r['sim_ns'] for r in rows];now=times[window['frame']]
            history=sorted({bisect_right(times,now-int(seconds*1e9))-1 for seconds in range(1,33)}-{ -1 })
            value['history_rgb']=[tensor(rgb_at(ep['path'],rows[i])) for i in history]
            if self.teacher_root is None:raise ValueError('World training requires V-JEPA feature cache')
            key=hashlib.sha256((ep['path']+ep['rgb_sha256']).encode()).hexdigest()[:20]
            artifact=self.teacher(key)
            if artifact['rgb_sha256']!=ep['rgb_sha256']:raise ValueError('Stale teacher cache')
            features=[]
            for i in window['future']:
                if str(i) not in artifact['features']:break
                features.append(artifact['features'][str(i)])
            if not features:raise ValueError('Missing causal teacher targets: encode dataset before world training')
            value['teacher']=torch.stack(features)
            value['future_rgb']=torch.stack([tensor(rgb_at(ep['path'],rows[i])) for i in window['future'][:len(features)]])
        return value


def encode_teacher(dataset_path,output,upstream,checkpoint,window):
    from visual_encoder import FrozenVideoEncoder,causal_indices
    encoder=FrozenVideoEncoder(upstream,checkpoint)
    spec=read(dataset_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=digest(checkpoint)
    for episode_index,ep in enumerate(spec['episodes']):
        if not window.remaining():return
        if ep['split']=='test':raise ValueError('Cannot encode sealed test trajectories for training')
        key=hashlib.sha256((ep['path']+ep['rgb_sha256']).encode()).hexdigest()[:20];path=output/(key+'.pt')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['teacher_sha256']!=identity or saved['rgb_sha256']!=ep['rgb_sha256']:raise ValueError('Teacher resume identity mismatch')
        else:saved=dict(features={},teacher_sha256=identity,rgb_sha256=ep['rgb_sha256'])
        rows=lines(Path(ep['path'])/'observations/frames.jsonl');times=[r['sim_ns'] for r in rows]
        needed=sorted({i for w in spec['windows'] if w['episode']==episode_index for i in w['future']})
        for i in needed:
            row=rows[i];reserve_memory()
            if str(i) in saved['features']:continue
            try:indices=causal_indices(times,row['sim_ns'])
            except ValueError:continue
            with torch.inference_mode():saved['features'][str(i)]=encoder([rgb_at(ep['path'],rows[j]) for j in indices])[0].cpu()
            if i%100==0 or not window.remaining():
                tmp=path.with_suffix('.pending');torch.save(saved,tmp);tmp.replace(path)
            if not window.remaining():return
        tmp=path.with_suffix('.pending');torch.save(saved,tmp);tmp.replace(path)
