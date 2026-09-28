"""Offline dataset construction; this module may read evaluator labels.

All targets live under `labels`; inference never imports this module. Image
windows are indexed, not copied into a second unbounded video archive.
"""
from bisect import bisect_right
from dataclasses import asdict
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
from .action_data import aerial_slots, IndexedCommands
from .aerial import camera_pitch
from .common import read, write, digest, config, reserve_memory
from .maps import MapPrior
from .contracts import Subgoal, SpatialSnapshot, snapshot_from_dict


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


def build_dataset(registry_path,flight_root,output,banks=()):
    registry=read(registry_path)
    if registry.get('schema')!='photo-map-scenes/v2':raise ValueError('Frozen v2 scene registry required')
    scene_by_id={s['scene_id']:s for s in registry['scenes']}
    episodes=[];windows=[];audit=[];goals_positive=goals_negative=0
    bank_records=[];bank_identities={}
    for bank_path in banks:
        bank=read(bank_path)
        if bank["registry_sha256"]!=digest(registry_path):raise ValueError("Bank registry mismatch")
        bank_identities[str(Path(bank_path).resolve())]=digest(bank_path)
        for record in bank["records"]:
            if record["split"] not in ("train","validation") or scene_by_id[record["scene_id"]]["split"]!=record["split"]:raise ValueError("Invalid bank split")
            bank_records.append(record)
    for result_path in sorted(Path(flight_root).rglob('episode/result.json')):
        episode=result_path.parent;result=read(result_path)
        scene_id=result.get('scene_id')
        if scene_id not in scene_by_id:
            audit.append(dict(episode=str(episode),reason='unregistered_scene'));continue
        scene=scene_by_id[scene_id]
        if result.get('camera_profile')!='fixed-forward-monocular/v1':
            audit.append(dict(episode=str(episode),reason='historical_camera_schema'));continue
        if scene['split']=='test':continue
        if result.get('split')!=scene['split']:raise ValueError('Scene/episode split mismatch')
        needed=[episode/'observations/storage.json',episode/'training_labels/frames.jsonl',episode/'training_labels/commands.jsonl',episode/'goal/goal.json',episode/'evaluator_labels/episode.json']
        if not all(p.exists() for p in needed):
            audit.append(dict(episode=str(episode),reason='incomplete_attempt_sources',termination=result.get('termination',result.get('status'))));continue
        storage=verified_rgb_storage(episode/'observations')
        rows=lines(episode/'observations/frames.jsonl');labels=lines(episode/'training_labels/frames.jsonl')
        selected={}
        for label in labels:
            if abs(label['observation_label_skew_seconds'])<=.1:
                old=selected.get(label['frame_id'])
                if old is None or abs(label['observation_label_skew_seconds'])<abs(old['observation_label_skew_seconds']):selected[label['frame_id']]=label
        task=read(episode/'evaluator_labels/episode.json');goal=np.asarray(task['goal_ned_m'])
        correction_path=episode/'photo-controller/runtime/decisions.jsonl'
        perception_path=correction_path if correction_path.exists() else episode/'perception/decisions.jsonl'
        perception_receipt=perception_path.parent/'receipt.json'
        if perception_path.exists() and perception_path!=correction_path:
            if not perception_receipt.exists() or read(perception_receipt).get('decisions_sha256')!=digest(perception_path):raise ValueError('Missing or stale perception replay receipt')
        estimates={r['frame_id']:r for r in lines(perception_path)} if perception_path.exists() else {}
        corrections={r['frame_id']:r for r in lines(correction_path) if r.get('expert_observation_conditioned') and 'subgoal_body' in r} if correction_path.exists() else {}
        prior=MapPrior(scene['map']);commands=IndexedCommands(lines(episode/'training_labels/commands.jsonl'))
        index=len(episodes)
        episodes.append(dict(path=str(episode.resolve()),scene_id=scene_id,split=scene['split'],map=scene['map'],
            map_sha256=prior.identity,goal_sha256=digest(episode/'goal/goal.json'),result_sha256=digest(result_path),
            labels_sha256=digest(episode/'training_labels/frames.jsonl'),index_sha256=digest(episode/'observations/frames.jsonl'),
            commands_sha256=digest(episode/'training_labels/commands.jsonl'),rgb_sha256=storage['stream_sha256'],
            corrections_sha256=digest(correction_path) if correction_path.exists() else None,
            perception_receipt_path=str(perception_receipt.resolve()),perception_receipt_sha256=digest(perception_receipt) if perception_receipt.exists() else None,
            perception_path=str(perception_path.resolve()),perception_sha256=digest(perception_path) if perception_path.exists() else None,
            collection_source=result.get('collection_source','legacy_expert')))
        times=[r['sim_ns'] for r in rows];previous=-math.inf
        for i,row in enumerate(rows):
            if abs(camera_pitch(row['calibration']))>1:raise ValueError('Nonfixed camera in temporal dataset')
            label=selected.get(row['frame_id'])
            if label is None or row['sim_ns']-previous<200_000_000:continue
            previous=row['sim_ns'];position=np.asarray(label['true_position_ned_m'])
            surface=float(prior.height(position[:2]))
            if not math.isfinite(surface):continue
            tile=int(np.linalg.norm(prior.tiles-position[:2],axis=1).argmin())
            horizontal=np.linalg.norm(position[:2]-goal[:2]);vertical=abs(position[2]-goal[2])
            geometric_near=bool(horizontal<=3 and vertical<=2)
            heading=yaw(label['true_quaternion_xyzw'])
            desired=math.radians(float(task.get('goal_yaw_degrees',0.)))
            yaw_error=abs((heading-desired+math.pi)%(2*math.pi)-math.pi)
            near=bool(geometric_near and yaw_error<=math.radians(30) and abs(camera_pitch(row['calibration']))<=20)
            valid=geometric_near or horizontal>=12 or vertical>=6
            stop=near and np.linalg.norm(label['true_velocity_ned_mps'])<.5
            if scene['split']=='train':goals_positive+=int(near);goals_negative+=int(valid and not near)
            path=np.asarray(task['reference_path_ned_m']);closest=int(np.linalg.norm(path-position,axis=1).argmin())
            target=path[min(closest+1,len(path)-1)]-position
            heading=yaw(label['true_quaternion_xyzw']);c,s=math.cos(heading),math.sin(heading)
            body=np.array([c*target[0]+s*target[1],-s*target[0]+c*target[1],target[2]])
            slots=aerial_slots(commands,row['sim_ns'],camera_pitch(row['calibration']))
            correction=corrections.get(row['frame_id'])
            estimate=estimates.get(row['frame_id'],{})
            recorded=estimate.get('context')
            spatial=snapshot_from_dict(recorded['spatial']) if recorded else SpatialSnapshot(observed_s=row['sim_ns']/1e9)
            subgoal=Subgoal(**recorded['subgoal']) if recorded and recorded.get('subgoal') else None
            execution=label.get('observable_execution_intention')
            if result.get('collection_source')=='learner' and estimate and recorded:
                # Observation-only DAgger safety teacher: retain safe learner
                # proposals, or inspect with yaw after a fresh visible veto.
                # This query never sees the remaining privileged route.
                command=estimate['proposed_command']
                if estimate.get('braked'):
                    command=[0.,0.,0.,15.] if not estimate.get('perception_error') and estimate.get('speed_mps') is not None else [0.]*4
                correction=dict(teacher_command=command)
            if not recorded and execution:
                subgoal=Subgoal(**label['subgoal'])
                spatial=snapshot_from_dict(label['spatial_snapshot'])
            if recorded:
                by_frame={r['frame_id']:j for j,r in enumerate(rows[max(0,i-256):i+1],max(0,i-256))}
                history=[by_frame[f] for f in recorded['frame_ids'] if f in by_frame]
            else:
                history=sorted({max(0,bisect_right(times,row['sim_ns']-j*50_000_000)-1) for j in range(4)})
            estimated_belief=estimate.get('belief',{})
            estimated_hypotheses=estimated_belief.get('hypotheses',[])
            estimated_tile=estimated_hypotheses[0]['tile_id'] if estimated_belief.get('aligned') and estimated_hypotheses else None
            if correction is not None and 'subgoal_body' in correction:body=np.asarray(correction['subgoal_body'])
            actions=[];future=[];future_positions=[];future_goal=[];future_collision=[];future_yaw=[]
            for step in range(20):
                stamp=row['sim_ns']+int((step+1)*200_000_000)
                j=bisect_right(times,stamp)-1
                slot=aerial_slots(commands,row['sim_ns']+step*200_000_000,camera_pitch(row['calibration']))
                target_label=selected.get(rows[j]['frame_id']) if j>=0 else None
                if j<=i or stamp-times[j]>100_000_000 or not all(slot['valid']) or target_label is None:break
                future.append(j);actions.append(slot['values'])
                p=np.asarray(target_label['true_position_ned_m'])-position
                future_positions.append([c*p[0]+s*p[1],-s*p[0]+c*p[1],p[2]])
                future_yaw.append((yaw(target_label['true_quaternion_xyzw'])-heading+math.pi)%(2*math.pi)-math.pi)
                future_goal.append(float(np.linalg.norm(np.asarray(target_label['true_position_ned_m'])[:2]-goal[:2])<=3 and abs(target_label['true_position_ned_m'][2]-goal[2])<=2
                    and abs((yaw(target_label['true_quaternion_xyzw'])-desired+math.pi)%(2*math.pi)-math.pi)<=math.radians(30)
                    and abs(camera_pitch(rows[j]['calibration']))<=20))
                future_collision.append(float(target_label['airsim_collision'] or target_label['geometry_collision']))
                if future_collision[-1]:break
            windows.append(dict(episode=index,frame=i,tile=tile,split=scene['split'],future=future,actions=actions,
                history=history,subgoal=asdict(subgoal) if subgoal else None,spatial=asdict(spatial),
                observation_schema='observation-context/v1',
                runtime_state=estimate.get('predictor_state'),estimated_tile=estimated_tile,flight_phase=(estimate.get('flight_phase') if estimate.get('flight_phase') not in (None,'unlabelled') else label.get('flight_phase','unlabelled')),
                camera_pitch_deg=camera_pitch(row['calibration']),
                labels=dict(offset=(position[:2]-prior.tiles[tile]).tolist(),above_surface=float(surface-position[2]),
                    yaw=[math.sin(heading),math.cos(heading)],near_goal=near,goal_valid=valid,arrival=stop,
                    subgoal_body=body.tolist(),command=correction['teacher_command'] if correction is not None else None if slots['values'] is None else slots['values'][0],
                    policy_valid=bool((correction is not None or execution is not None and all(slots['valid'])) and not label['airsim_collision'] and not label['geometry_collision']),
                    terminal_value=float(bool(result.get('success')))-2*float(result.get('termination') in ('collision','geometry_collision'))
                        -.001*max(0.,result.get('elapsed_sim_seconds',0.)-(row['sim_ns']-rows[0]['sim_ns'])/1e9),
                    future_position=future_positions,future_goal=future_goal,future_collision=future_collision,future_yaw=future_yaw)))
    write(output,dict(schema='photo-map-dataset/v4',bank_records=bank_records,bank_identities=bank_identities,registry_sha256=digest(registry_path),episodes=episodes,windows=windows,
          audit=audit,training_goal_positives=goals_positive,training_goal_negatives=goals_negative,
          action_semantics='post-safety dispatch intervals, not measured actuator application',
          purpose='perception, subgoal execution and transition learning; no omniscient search imitation'))


class Dataset:
    def __init__(self,path,stage,split='train',teacher_root=None):
        self.path=Path(path);self.spec=read(path);self.stage=stage;self.teacher_root=Path(teacher_root) if teacher_root else None
        if self.spec['schema']!='photo-map-dataset/v4':raise ValueError('Fixed-camera temporal dataset required')
        if split not in ('train','validation'):raise ValueError('Sealed test data cannot train or tune models')
        self.teacher_identity=None
        self.windows=[w for w in self.spec['windows'] if w['split']==split and
            (stage!='goal' or w['labels']['goal_valid']) and (stage not in ('policy','dagger') or w['labels']['policy_valid']) and
            (stage!='odometry' or len(w['future'])>=1) and
            (stage!='world' or len(w['future'])>=1)]
        if stage=='world':
            eligible=[]
            for window in self.windows:
                ep=self.spec['episodes'][window['episode']]
                key=hashlib.sha256((ep['path']+ep['rgb_sha256']).encode()).hexdigest()[:20]
                if self.teacher_root is None or not (self.teacher_root/(key+'.pt')).exists():
                    raise ValueError('Encode V-JEPA teacher targets before world training')
                artifact=self.teacher(key)
                if self.teacher_identity is None:self.teacher_identity=artifact['teacher_sha256']
                if self.teacher_identity!=artifact['teacher_sha256']:raise ValueError('Mixed teacher identities')
                if any(str(i) in artifact['features'] for i in window['future']):eligible.append(window)
            self.windows=eligible
        self.banks=[r for r in self.spec.get('bank_records',[]) if r['split']==split] if stage=='localization' else []
        if not self.windows and not self.banks:raise ValueError('No qualified '+stage+' windows for '+split+'; world data requires recorded/replayed runtime estimates')
        for name,identity in self.spec.get('bank_identities',{}).items():
            if digest(name)!=identity:raise ValueError('Changed perception bank')
        for row in self.banks:
            if digest(row['image'])!=row['image_sha256']:raise ValueError('Changed bank image')
        if stage=='goal' and split=='train':
            if min(self.spec['training_goal_positives'],self.spec['training_goal_negatives'])<100:
                raise ValueError('Goal stage needs at least 100 positive and 100 negative examples')
        for ep in self.spec['episodes']:
            if ep['split']=='test':raise ValueError('Sealed test episode in training dataset')
            root=Path(ep['path'])
            for name,key in [('result.json','result_sha256'),('goal/goal.json','goal_sha256'),('training_labels/frames.jsonl','labels_sha256'),
                             ('observations/frames.jsonl','index_sha256'),('training_labels/commands.jsonl','commands_sha256'),('observations/rgb.zlib','rgb_sha256')]:
                if digest(root/name)!=ep[key]:raise ValueError('Dataset source changed: '+str(root/name))
            if ep.get('perception_receipt_sha256') and digest(ep['perception_receipt_path'])!=ep['perception_receipt_sha256']:raise ValueError('Perception receipt changed')
            if ep.get('perception_sha256') and digest(ep['perception_path'])!=ep['perception_sha256']:raise ValueError('Runtime estimates changed')
            corrections=root/'photo-controller/runtime/decisions.jsonl'
            if (digest(corrections) if corrections.exists() else None)!=ep.get('corrections_sha256'):
                raise ValueError('Correction labels changed after dataset construction')

    def __len__(self):return len(self.windows)+len(self.banks)

    def sample_index(self, rng, learner_mix=False):
        # Hierarchical sampling prevents long cruises and long episodes dominating.
        if not hasattr(self,'groups'):
            self.groups={}
            for i,w in enumerate(self.windows):
                ep=self.spec['episodes'][w['episode']]
                key=(ep['collection_source']=='learner',ep['scene_id'],w['episode'],w.get('flight_phase','unlabelled'))
                self.groups.setdefault(key,[]).append(i)
            for i,row in enumerate(self.banks):
                self.groups.setdefault((False,row['scene_id'],'bank',str(row['camera_pitch_deg'])),[]).append(len(self.windows)+i)
        keys=list(self.groups)
        if self.stage=='goal':
            desired=rng.choice([True,False])
            if not hasattr(self,'goal_groups'):
                self.goal_groups={flag:{key:[i for i in indices if self.windows[i]['labels']['near_goal']==flag] for key,indices in self.groups.items()} for flag in (True,False)}
            matching=self.goal_groups[desired]
            keys=[key for key in keys if matching[key]]
            if not keys:raise ValueError('Both positive and negative goal examples required')
        if learner_mix:
            family=rng.choice(sorted({k[0] for k in keys}));keys=[k for k in keys if k[0]==family]
        scene=rng.choice(sorted({k[1] for k in keys}));keys=[k for k in keys if k[1]==scene]
        episode=rng.choice(sorted({str(k[2]) for k in keys}));keys=[k for k in keys if str(k[2])==episode]
        key=rng.choice(keys)
        return rng.choice(matching[key] if self.stage=='goal' else self.groups[key])

    @lru_cache(maxsize=8)
    def prior(self,path):return MapPrior(path)

    def localization_sample(self,current,prior,position,heading,pitch,labels=None):
        from PIL import Image
        labels={} if labels is None else dict(labels)
        tile=int(np.linalg.norm(prior.tiles-np.asarray(position[:2]),axis=1).argmin())
        offset=np.asarray(position[:2])-prior.tiles[tile]
        valid=(np.abs(prior.tiles[:,0]-position[0])<=64)&(np.abs(prior.tiles[:,1]-position[1])<=48)
        invalid=np.flatnonzero(~valid)
        if len(invalid)==0:raise ValueError('Need distinct nonoverlapping localization negatives')
        distance=np.linalg.norm(prior.tiles[invalid]-np.asarray(position[:2]),axis=1)
        nearby=int(invalid[distance.argmin()]);distant=int(invalid[distance.argmax()])
        def histogram(array):
            return np.concatenate([np.histogram(array[...,k],bins=8,range=(0,256),density=True)[0] for k in range(3)])
        if not hasattr(prior,'histograms'):prior.histograms=np.stack([histogram(prior.tile(i)[::16,::16]) for i in range(len(prior.tiles))])
        appearance=histogram(current.permute(1,2,0).numpy()[::16,::16])
        hard=int(invalid[np.linalg.norm(prior.histograms[invalid]-appearance,axis=1).argmin()])
        # Two containing tiles are jointly positive. Never call overlapping true tiles negative.
        positives=list(np.flatnonzero(valid));positives.sort(key=lambda i:np.linalg.norm(prior.tiles[i]-position[:2]))
        chosen=[tile,positives[min(1,len(positives)-1)],nearby,distant,hard]
        tiles=torch.stack([torch.from_numpy(prior.tile(i)).permute(2,0,1) for i in chosen])
        texture=float(current.float().mean(0).std())>=5.
        labels.update(offset=offset.tolist(),above_surface=float(prior.height(position[:2])-position[2]),
                      yaw=[math.sin(heading),math.cos(heading)],localization_usable=texture)
        return dict(current=current,tile=tiles[0],map_path=str(prior.root),positive_tile_ids=np.flatnonzero(valid).tolist(),position_ned_m=np.asarray(position).tolist(),retrieval_tiles=tiles,retrieval_positive=torch.tensor([True,True,False,False,False]),
                    labels=labels,camera_pitch_deg=pitch)


    @lru_cache(maxsize=2)
    def teacher(self,key):
        return torch.load(self.teacher_root/(key+'.pt'),map_location='cpu',weights_only=True)

    @lru_cache(maxsize=8)
    def episode(self,index):
        ep=self.spec['episodes'][index];prior=MapPrior(ep['map'])
        if prior.identity!=ep['map_sha256']:raise ValueError('Dataset map changed')
        return ep,lines(Path(ep['path'])/'observations/frames.jsonl'),load_goal(Path(ep['path'])/'goal'),prior

    @lru_cache(maxsize=8)
    def spatial_history(self,index):
        ep=self.spec['episodes'][index]
        if not ep.get('perception_sha256'):return []
        return [(r['observed_s'],snapshot_from_dict(r['context']['spatial'])) for r in lines(ep['perception_path']) if r.get('context')]

    def get(self,index):
        def tensor(raw):return torch.from_numpy(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy()).permute(2,0,1)
        if index>=len(self.windows):
            from PIL import Image
            row=self.banks[index-len(self.windows)];prior=self.prior(row['map'])
            if prior.identity!=row['map_sha256']:raise ValueError('Changed bank map')
            current=torch.from_numpy(np.asarray(Image.open(row['image']).convert('RGB')).copy()).permute(2,0,1)
            return self.localization_sample(current,prior,np.asarray(row['position_ned_m']),row['yaw_rad'],row['camera_pitch_deg'])
        window=self.windows[index];ep,rows,goal,prior=self.episode(window['episode'])
        current=tensor(rgb_at(ep['path'],rows[window['frame']]))
        labels=window['labels'];position=np.r_[prior.tiles[window['tile']]+labels['offset'],0.]
        position[2]=float(prior.height(position[:2]))-labels['above_surface']
        if self.stage=='localization':
            return self.localization_sample(current,prior,position,math.atan2(*labels['yaw']),window['camera_pitch_deg'],labels)
        goals=torch.stack([tensor(raw) for raw in goal.rgb_views])
        input_tile=window['estimated_tile'] if window.get('estimated_tile') is not None else 0
        tile=torch.from_numpy(prior.tile(input_tile)).permute(2,0,1)
        value=dict(current=current,goals=goals,tile=tile,labels=labels,camera_pitch_deg=window['camera_pitch_deg'],
                   runtime_state=window.get('runtime_state'),actions=torch.tensor(window['actions'],dtype=torch.float32))
        history=window['history'];times=[r['sim_ns']/1e9 for r in rows]
        value['history_rgb']=[tensor(rgb_at(ep['path'],rows[i])) for i in history]
        value['history_times']=[times[i] for i in history]
        value['history_ids']=[rows[i]['frame_id'] for i in history]
        value['history_commands']=[]
        for i in history:
            commands=rows[i].get('command_history',[])
            previous=[r for r in commands if r['sim_ns']<=rows[i]['sim_ns']]
            value['history_commands'].append(previous[-1]['values'] if previous else [0.]*4)
        value['subgoal']=Subgoal(**window['subgoal']) if window['subgoal'] else None
        value['spatial']=snapshot_from_dict(window['spatial'])
        now=value['history_times'][-1]
        past=[snapshot for stamp,snapshot in self.spatial_history(window['episode']) if 0<now-stamp<=3]
        value['delayed_spatial']=past[::max(1,len(past)//3)][:3]
        value['reference_rgb']=None
        if value['subgoal'] and value['subgoal'].target_source=='map':
            ref=value['subgoal'].target_reference
            if ref not in value['spatial'].map_references:raise ValueError('Unsupported recorded map target')
            value['reference_rgb']=torch.from_numpy(prior.tile(int(ref.split('-')[-1]))).permute(2,0,1)
        elif value['subgoal'] and value['subgoal'].target_source=='keyframe':
            source=next(r[1] for r in value['spatial'].keyframes if r[0]==value['subgoal'].target_reference)
            index=next(i for i,r in enumerate(rows) if r['frame_id']==source)
            value['reference_rgb']=tensor(rgb_at(ep['path'],rows[index]))
        elif value['subgoal'] and value['subgoal'].target_source=='geometry':value['reference_rgb']=current
        value['future_rgb']=[tensor(rgb_at(ep['path'],rows[i])) for i in window['future']]
        if self.stage=='world':
            if self.teacher_root is None:raise ValueError('World training requires V-JEPA feature cache')
            key=hashlib.sha256((ep['path']+ep['rgb_sha256']).encode()).hexdigest()[:20]
            artifact=self.teacher(key)
            if artifact['rgb_sha256']!=ep['rgb_sha256']:raise ValueError('Stale teacher cache')
            features=[]
            for i in window['future']:
                features.append(artifact['features'].get(str(i),torch.zeros(64,1024)))
            if not features:raise ValueError('Missing causal teacher targets: encode dataset before world training')
            value['teacher']=torch.stack(features)
            value['teacher_valid']=torch.tensor([str(i) in artifact['features'] for i in window['future']])
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
