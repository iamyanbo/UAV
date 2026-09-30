"""Recorded city RGB -> released V-JEPA targets -> fast visual basis.

No expert controls, policy imitation, Qwen gradients or world-model gradients.
The resulting actor heads are initialized, not PPO-trained. Training examples are
actual causal flight clips, split by episode before teacher/PCA fitting.
"""
import gc
import json
import math
from pathlib import Path
import random
import signal
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .common import read, write, digest, contained, FlightLock
from .mission_contracts import city_config, identity, BUNDLE_SCHEMA
from .mission_resources import Resources, RunWindow
from .project_city import ProjectCity
from .vision.visual_encoder import causal_indices, FrozenVideoEncoder
from .vision.goal_matching import SharedSpatialEncoder


def emit(**row):
    print(json.dumps(row),flush=True)


def atomic_torch(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');torch.save(payload,pending);pending.replace(path)


def collect(root,output,flights,resources,window,seed):
    manifest=output/'clips.json'
    data=read(manifest) if manifest.exists() else dict(schema='photo-goal-visual-bootstrap-clips/v1',
        scene='CityEnviron-Linux-1.0.1',seed=seed,episodes=[],clips=[],rejected=[],
        runtime_model_inputs=['RGB'],expert_controls=False,mission_navigation_evidence=False,
        calibration_id='front-monocular640x480-fov90-project-city/v1')
    if data['seed']!=seed:raise ValueError('Collection seed changed')
    rng=random.Random(seed)
    # Fixed episode splits precede collection; held-out episodes never
    # enter teacher compression fitting or optimizer batches.
    candidates=[]
    for index in range(flights*3):
        split='development' if index%8==7 else 'train'
        position=[rng.uniform(-140,140),rng.uniform(-140,140),-rng.choice([20.,30.,40.,50.])]
        yaw=rng.uniform(-180,180)
        command=[rng.uniform(-1,1),rng.uniform(-1,1),rng.uniform(-.3,.3),rng.uniform(-12,12)]
        candidates.append((split,position,yaw,command))
    already={r['index'] for r in data['episodes']}|{r['index'] for r in data['rejected']}
    if len(data['episodes'])>=flights:return manifest
    with ProjectCity(root,output/'scene') as session:
        resources.check()
        scene_receipt=read(output/'scene/scene-receipt.json')
        if data.get('scene_binary_sha256') and data['scene_binary_sha256']!=scene_receipt['binary_sha256']:
            raise ValueError('Collected scene identity changed')
        data['scene_binary_sha256']=scene_receipt['binary_sha256']
        data['settings_sha256']=scene_receipt['settings_sha256']
        for index,(split,position,yaw,command) in enumerate(candidates):
            if index in already:continue
            if len(data['episodes'])>=flights or not window.admits(180):break
            resources.check(growth_bytes=200*2**20,checkpoint_bytes=100*2**20)
            episode=output/f'flight-{index:04d}'
            if episode.exists():episode=output/f'flight-{index:04d}-retry-{time.time_ns()}'
            episode.mkdir(exist_ok=False)
            frames=[];commands=[];error=None
            try:
                state=session.reset(position,yaw)
                k=state['kinematics'];actual=list(k['pose']['position'].values())
                speed=math.sqrt(sum(v*v for v in k['twist']['linear'].values()))
                if math.dist(position,actual)>.5 or speed>.3:
                    raise RuntimeError('Endpoint did not hold the requested pose')
                baseline=len(session.collision_events)
                start_ns=state['sim_ns'];last_ns=start_ns-1;started=time.perf_counter()
                while True:
                    if not window.admits(60):raise RuntimeError('Window exhausted during RGB collection')
                    rgb,receipt=session.image();stamp=int(receipt['time_stamp'])
                    if stamp<start_ns or stamp<=last_ns:continue
                    if time.perf_counter()-receipt['received_wall']>.25:
                        session.command([0.,0.,0.,0.]);raise RuntimeError('Camera freshness exceeded during collection')
                    _,ack=session.command(command,duration=.2)
                    commands.append(ack)
                    path=episode/f'{len(frames):05d}.png'
                    Image.fromarray(rgb).save(path,compress_level=1)
                    frames.append(dict(image=str(path.relative_to(output)),sha256=digest(path),sim_ns=stamp,
                        received_wall=receipt['received_wall'],camera={k:v for k,v in receipt.items() if k.startswith(('pos_','rot_'))}))
                    last_ns=stamp
                    if stamp-start_ns>=8*10**9:break
                    if time.perf_counter()-started>60:raise RuntimeError('Eight simulated seconds exceeded bounded wall window')
                session.command([0.,0.,0.,0.],duration=1)
                time.sleep(1)
                after=session.state()
                collisions=session.collision_events[baseline:]
                if any(r['event'].get('time_stamp',0)>0 for r in collisions):
                    raise RuntimeError('Contact during visual bootstrap flight')
                qualified=[]
                timestamps=[r['sim_ns'] for r in frames]
                # At most four distinct ends per episode; no repeated missing frames.
                for fraction in (.4,.6,.8,1.):
                    end=timestamps[min(len(frames)-1,round((len(frames)-1)*fraction))]
                    try:selected=causal_indices(timestamps,end)
                    except ValueError:continue
                    qualified.append(dict(id=f'flight-{index:04d}-{end}',episode=f'flight-{index:04d}',
                        split=split,end_ns=end,frames=[frames[i] for i in selected],
                        calibration_id=data['calibration_id']))
                if not qualified:raise RuntimeError('No strictly causal 16-frame clip qualified')
                record=dict(schema='photo-goal-bootstrap-flight/v1',index=index,split=split,
                    reset_label=dict(position=position,yaw_deg=yaw,state=state),after_label=after,
                    frames=frames,commands=commands,collisions=collisions,continuous_physics=True,
                    episode_complete=True,terminal_kind='collection_duration',navigation_success=None,
                    action_timing_qualified=False,usable_for_policy_imitation=False)
                write(episode/'receipt.json',record)
                data['episodes'].append(dict(index=index,id=f'flight-{index:04d}',split=split,
                    receipt=str((episode/'receipt.json').relative_to(output)),sha256=digest(episode/'receipt.json')))
                data['clips'].extend(qualified)
                emit(stage='recorded-rgb',episodes=len(data['episodes']),clips=len(data['clips']),frames=len(frames),index=index)
            except RuntimeError as exc:
                error=str(exc);data['rejected'].append(dict(index=index,reason=error))
                emit(stage='recorded-rgb',index=index,rejected=error)
            finally:
                session.command([0.,0.,0.,0.])
                write(manifest,data)
    return manifest


def targets(root,manifest,output,resources,window,device='cuda',limit=None):
    data=read(manifest)
    train=[r for r in data['clips'] if r['split']=='train']
    development=[r for r in data['clips'] if r['split']=='development']
    if limit is None and (len({r['episode'] for r in train})<8 or not development):
        raise RuntimeError('Need eight distinct training flights and held-out episodes for visual bootstrap')
    model=root/'assets/models/vjepa2-vitl.pt'
    teacher_sha=digest(model)
    encoder=FrozenVideoEncoder(root/'assets/upstream/vjepa2',model,device)
    resources.check()
    fast=SharedSpatialEncoder(root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt').to(device).eval()
    fast.requires_grad_(False)
    resources.check()
    rows=[]
    for clip in (train+development)[:limit]:
        if not window.admits(120):break
        resources.check()
        key=identity(dict(clip=clip,teacher=teacher_sha,spatial_resolution=16,
                          alignment='remove-32px-top-bottom-letterbox-then-bilinear15x20/v1'))
        path=output/'features'/f'{key}.pt'
        if not path.exists():
            pixels=[]
            for row in clip['frames']:
                source=contained(manifest.parent,row['image'])
                if digest(source)!=row['sha256']:raise RuntimeError('Recorded RGB changed')
                with Image.open(source) as im:
                    if im.mode!='RGB' or im.size!=(640,480):raise RuntimeError('Invalid recorded RGB geometry')
                    pixels.append(im.tobytes())
            with torch.inference_mode():
                teacher=encoder(pixels,spatial_resolution=16).reshape(1,16,16,1024).permute(0,3,1,2)
                teacher=F.interpolate(teacher[:,:,2:14,:],(15,20),mode='bilinear',align_corners=False).cpu()
                image=torch.from_numpy(np.frombuffer(pixels[-1],np.uint8).copy().reshape(480,640,3)).permute(2,0,1).unsqueeze(0).to(device).float()/255
                raw=fast.backbone((image-fast.rgb_mean)/fast.rgb_std).cpu()
            atomic_torch(path,dict(schema='photo-goal-bootstrap-alignment/v1',raw=raw.half(),teacher=teacher.half(),
                                   episode=clip['episode'],split=clip['split'],teacher_sha256=teacher_sha,clip=clip))
        rows.append(dict(path=str(path),episode=clip['episode'],split=clip['split'],sha256=digest(path)))
        if len(rows)%20==0:emit(stage='official-video-targets',clips=len(rows),resources=resources.check())
    del encoder,fast;gc.collect();torch.cuda.empty_cache()
    write(output/'features.json',dict(schema='photo-goal-bootstrap-features/v1',rows=rows,
                                    source_manifest_sha256=digest(manifest),teacher_sha256=teacher_sha))
    return output/'features.json'


def fit(root,features,output,updates,resources,window,seed):
    torch.set_num_threads(8);torch.manual_seed(seed)
    data=read(features);source_sha=digest(features)
    train=[torch.load(r['path'],weights_only=True,map_location='cpu') for r in data['rows'] if r['split']=='train']
    dev=[torch.load(r['path'],weights_only=True,map_location='cpu') for r in data['rows'] if r['split']=='development']
    if len({r['episode'] for r in train})<8 or not dev:raise RuntimeError('Insufficient independent train/development flight clips')
    pca_file=output/'teacher-pca.pt'
    if not pca_file.exists():
        values=torch.cat([r['teacher'].float().flatten(2).transpose(1,2).reshape(-1,1024)[::3] for r in train])
        mean=values.mean(0);centered=values-mean
        _,_,components=torch.pca_lowrank(centered,q=256,center=False,niter=4)
        atomic_torch(pca_file,dict(mean=mean,components=components,fit_split='train',
            episode_ids=sorted({r['episode'] for r in train}),source_manifest_sha256=data['source_manifest_sha256'],
            features_sha256=source_sha,teacher_sha256=data['teacher_sha256']))
        del values,centered
    pca=torch.load(pca_file,weights_only=True,map_location='cpu')
    if pca['features_sha256']!=source_sha:raise RuntimeError('Training corpus changed after PCA fitting')
    projection=pca['components'].cuda();mean=pca['mean'].cuda()
    def packed(rows):
        raw=torch.cat([r['raw'].float() for r in rows]).cuda()
        teacher=torch.cat([r['teacher'].float().flatten(2).transpose(1,2) for r in rows]).cuda()
        return raw,(teacher-mean)@projection
    train_raw,train_target=packed(train);dev_raw,dev_target=packed(dev)
    encoder=SharedSpatialEncoder(root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt').cuda().eval()
    encoder.backbone.requires_grad_(False)
    trainable=list(encoder.projection.parameters())+list(encoder.position.parameters())
    optimizer=torch.optim.AdamW(trainable,lr=3e-4,weight_decay=1e-4)
    latest=output/'latest-bootstrap.pt';start=0;rng=random.Random(seed)
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['features_sha256']!=source_sha or saved['target_updates']!=updates:raise ValueError('Bootstrap resume corpus/update target differs')
        encoder.load_state_dict(saved['encoder'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        start=saved['updates'];rng.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def forward(raw):
        feature=encoder.projection(raw)
        y,x=torch.meshgrid(torch.linspace(-1,1,15,device='cuda'),torch.linspace(-1,1,20,device='cuda'),indexing='ij')
        return (feature+encoder.position(torch.stack((x,y))[None])).flatten(2).transpose(1,2)
    def loss(raw,target):
        prediction=forward(raw)
        return F.mse_loss(F.normalize(prediction,dim=-1),F.normalize(target,dim=-1))*256+(1-F.cosine_similarity(prediction,target,dim=-1)).mean()
    resources.check(checkpoint_bytes=100*2**20)
    torch.cuda.reset_peak_memory_stats()
    interrupted=[False]
    def stop(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    actual=start;last_dev=None
    def checkpoint():
        from .mission_checkpoint import cpu_copy
        atomic_torch(latest,dict(schema='photo-goal-visual-bootstrap/v1',encoder=cpu_copy(encoder.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
             updates=actual,target_updates=updates,features_sha256=source_sha,pca_sha256=digest(pca_file),
             python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
             no_policy_imitation=True,ppo_updates=0,world_updates=0,qwen_updates=0))
        write(output/'training-receipt.json',dict(schema='photo-goal-visual-bootstrap/v1',updates=actual,target_updates=updates,
             complete=actual>=updates,training_clips=len(train),training_episodes=len({r['episode'] for r in train}),
             development_clips=len(dev),development_loss=last_dev,checkpoint_sha256=digest(latest),
             gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),resources=resources.check(),
             ppo_transitions=0,world_updates=0,qwen_updates=0,scope='learned fast RGB representation; navigation performance unmeasured'))
    try:
        with (output/'loss.jsonl').open('a') as stream:
            for update in range(start,updates):
                if interrupted[0] or not window.admits(120):break
                resources.check(checkpoint_bytes=100*2**20)
                ids=torch.tensor([rng.randrange(len(train)) for _ in range(16)],device='cuda')
                optimizer.zero_grad(set_to_none=True);value=loss(train_raw[ids],train_target[ids])
                if not torch.isfinite(value):raise RuntimeError('Nonfinite visual alignment loss')
                value.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step();actual=update+1
                if actual==1 or actual%100==0:
                    with torch.inference_mode():last_dev=float(loss(dev_raw,dev_target))
                    row=dict(stage='visual-bootstrap',update=actual,loss=float(value.detach()),development_loss=last_dev)
                    stream.write(json.dumps(row)+'\n');stream.flush();emit(**row)
                if actual==1 or actual%500==0:checkpoint()
    finally:checkpoint()
    if actual>=updates:
        from .mission_policy import CityActorCritic,owned_optimizer
        from .mission_world import CityWorld
        from .mission_checkpoint import encoder_identity,save_bundle
        cfg=city_config();actor=CityActorCritic(root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt',cfg['stop_prior']).cuda()
        actor.encoder.load_state_dict(encoder.state_dict(),strict=True);actor.encoder.requires_grad_(False).eval()
        world=CityWorld().cuda();actor_optimizer=owned_optimizer(actor,cfg['learning_rate'])
        world_optimizer=torch.optim.AdamW(world.parameters(),lr=cfg['world']['learning_rate'])
        metadata=dict(schema=BUNDLE_SCHEMA,config_sha256=identity(cfg),backbone_sha256=digest(root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt'),
            encoder_basis_sha256=encoder_identity(actor),source_checkpoint_sha256=digest(latest),
            historical_counts={},counts=dict(city_batches=0,accepted_transitions=0,world_updates=0),qwen_adapter=None,
            world_ranking_qualified=False,pending=None,initialization='official-backbone-and-recorded-city-VJEPA-distillation',
            visual_bootstrap_updates=actual,visual_basis_qualified_for_navigation=False,old_flight_checkpoint_recovered=False)
        save_bundle(root/'checkpoints/city-initialized.pt',actor,actor_optimizer,world,world_optimizer,metadata,cfg)
        emit(stage='visual-bootstrap',status='complete',checkpoint=str(root/'checkpoints/city-initialized.pt'),ppo_transitions=0)


def run(args):
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Operator or eight-hour window requested shutdown')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    root=Path(args.root).resolve();output=root/'data/visual-bootstrap'
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    resources=Resources(root,city_config(),'cuda');window=RunWindow(args.hours)
    resources.check()
    with FlightLock(root,'city-visual-bootstrap'):
        manifest=collect(root,output,args.flights,resources,window,args.seed)
        features=targets(root,manifest,output,resources,window)
        fit(root,features,output,args.updates,resources,window,args.seed)
