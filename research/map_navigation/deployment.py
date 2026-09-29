"""Inference packaging and network-isolated Spark process launch.

No runtime container receives registry files, dataset roots or simulator RPC.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .common import SCHEMA,read,write,digest,contained
from .maps import MapPrior


def package(checkpoint,vision,output,learned_local_policy=True,photo_slam=None,qwen=None,perception_only=False):
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    required={'localization','goal'}|({'policy'} if not perception_only else set())
    if saved.get('schema')!=SCHEMA or not required<=set(saved['trained_stages']):
        raise ValueError('Train temporal actor and perception before packaging')
    if photo_slam is None:raise ValueError('Pinned Photo-SLAM deployment descriptor required')
    mapping=read(photo_slam)
    if mapping.get('upstream_commit')!='f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac' or mapping.get('schema')!='photo-slam-live/v1':
        raise ValueError('Incompatible native dependency descriptor')
    if not saved.get('calibration'):raise ValueError('Calibrate arrival on validation data before packaging')
    from .provenance import verify_release
    verify_release(saved,world='world' in saved['trained_stages'] and not perception_only,perception_only=perception_only)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    # Strip labels, optimizer and training-only identities from inference package.
    torch.save({key:saved[key] for key in ('schema','model','trained_stages','backbone_sha256','calibration','budget_usage','world_actor_identity') if key in saved},out/'model.pt')
    shutil.copyfile(vision,out/'vision.json')
    shutil.copyfile(photo_slam,out/'photo-slam.json')
    if qwen:
        adapter=torch.load(qwen,map_location='cpu',weights_only=True)
        if adapter.get('schema')!='subgoal-qwen/v3':raise ValueError('Incompatible Qwen adapter')
        torch.save({k:adapter[k] for k in ('schema','modules','adapter','base_identity','budget_usage')},out/'qwen.pt')
    write(out/'package.json',dict(schema='photo-map-package/v6',trained_stages=saved['trained_stages'],
          capability='perception' if perception_only else 'navigation',
          learned_local_policy=not perception_only,accepted=False,files={p.name:digest(p) for p in out.iterdir()}))


class PhotoControllerProcess:
    def __init__(self,socket_path,episode_id,checkpoints,map_prior,output,variant,sample_policy=False,initial_subgoal=None,perception_only=False,teacher_spec=None):
        root=Path.home()/'uav-rgb-flight';source_root=Path(__file__).resolve().parents[2]
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.writable=self.output/'runtime';self.writable.mkdir()
        package_root=Path(checkpoints).resolve();spec=read(package_root/'package.json')
        required={'model.pt','vision.json','photo-slam.json'}
        if spec['schema']!='photo-map-package/v6' or not required<=set(spec['files']) or set(spec['files'])-required-{'qwen.pt'}:raise ValueError('Unexpected inference assets')
        if spec.get('capability')!=('perception' if perception_only else 'navigation'):raise ValueError('Package capability mismatch')
        allowed={'package.json',*spec['files']}
        if {p.name for p in package_root.iterdir()}!=allowed:raise ValueError('Inference package has undeclared files')
        for name,sha in spec['files'].items():
            if digest(contained(package_root,name))!=sha:raise ValueError('Checkpoint identity mismatch')
        prior=MapPrior(map_prior)
        if {p.name for p in prior.root.iterdir()}!={'map.json',*prior.meta['files']}:
            raise ValueError('Map namespace has undeclared files')
        socket_path=Path(socket_path).resolve()
        if {p.name for p in socket_path.parent.iterdir()}!={socket_path.name}:raise ValueError('IPC contains more than socket')
        source=self.output/'source';(source/'research').mkdir(parents=True)
        # Copy source files only, never evidence/training data adjacent to code.
        for folder in ('map_navigation','rgb_flight'):
            target=source/'research'/folder;target.mkdir()
            for path in (source_root/'research'/folder).glob('*.py'):shutil.copyfile(path,target/path.name)
        shutil.copyfile(Path(__file__).with_name('campaign.json'),source/'research/map_navigation/campaign.json')
        write(self.output/'source_hashes.json',{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()})
        import uuid
        image=read(root/'receipts/model-stack.json')['image_id'];self.name='photo-map-'+str(os.getpid())+'-'+uuid.uuid4().hex[:8]
        self.identity=digest(package_root/'package.json')
        command=['docker','run','--rm','--name',self.name,'--cidfile',str(self.output/'container.cid'),'--gpus','all','--network','none','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}','--cpus','8','--shm-size','2g',
            '-v',str(source)+':/source:ro','-v',str(root/'assets/models')+':/models:ro',
            '-v',str(root/'deps/Metric3D')+':/upstream/metric3d:ro',
            '-v',str(root/'deps/Photo-SLAM')+':/upstream/photo-slam:ro',
            '-e','PYTHONPATH=/models/photo-slam/lib',
            '-e','LD_LIBRARY_PATH=/models/photo-slam/lib:/upstream/photo-slam/lib:/upstream/photo-slam/ORB-SLAM3/lib',
            '-v',str(package_root)+':/navigation:ro','-v',str(prior.root)+':/prior:ro',
            '-v',str(socket_path.parent)+':/ipc:ro','-v',str(self.writable)+':/output',
            '-w','/source',image,'python','-m','research.map_navigation.runtime','--episode-id',episode_id,'--variant',variant]
        if sample_policy:command.append('--sample-policy')
        if perception_only:command.append('--perception-only')
        if initial_subgoal:command+=['--initial-subgoal',initial_subgoal]
        if teacher_spec:
            # The descriptor contains thresholds/provenance only, never labels.
            descriptor=read(teacher_spec)
            allowed_teacher={'schema','implementation_sha256','perception_sha256','minimum_inlier_ratio',
                             'arrival_threshold','qualified','qualification_sha256'}
            if set(descriptor)!=allowed_teacher:raise ValueError('Teacher descriptor has undeclared fields')
            target=self.output/'teacher.json';write(target,descriptor)
            image_index=command.index(image)
            command[image_index:image_index]=['-v',str(target.resolve())+':/teacher.json:ro']
            command+=['--teacher','/teacher.json']
        write(self.output/'request.json',dict(command=command,map_sha256=prior.identity,checkpoint_sha256=self.identity))
        self.log=(self.output/'stdout.log').open('x');self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT)

    def ready(self,timeout=240):
        deadline=time.monotonic()+timeout
        while not (self.writable/'CONTROLLER_READY').exists():
            self.check()
            if time.monotonic()>deadline:raise RuntimeError('Photo controller initialization timeout')
            time.sleep(.1)

    def check(self):
        if self.process.poll() is not None:raise RuntimeError('Photo controller exited; inspect its preserved log')

    def close(self):
        (self.writable/'CONTROLLER_STOP').touch()
        try:self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','-t','5',self.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.process.wait(timeout=20)
        finally:self.log.close()
        path=self.writable/'result.json'
        return read(path) if path.exists() else dict(status='failed',reason='missing_runtime_receipt')
