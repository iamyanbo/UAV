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


def package(checkpoint,vision,output,learned_local_policy=False):
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or not {'localization','goal'}<=set(saved['trained_stages']):
        raise ValueError('Train localization and goal recognition before packaging')
    if learned_local_policy and 'policy' not in saved['trained_stages']:raise ValueError('Local policy has not been trained')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    # Strip labels, optimizer and training-only identities from inference package.
    torch.save({key:saved[key] for key in ('schema','model','trained_stages','backbone_sha256')},out/'model.pt')
    shutil.copyfile(vision,out/'vision.json')
    write(out/'package.json',dict(schema='photo-map-package/v1',trained_stages=saved['trained_stages'],
          learned_local_policy=learned_local_policy,accepted=False,files={p.name:digest(p) for p in out.iterdir()}))


class PhotoControllerProcess:
    def __init__(self,socket_path,episode_id,checkpoints,map_prior,output,variant):
        root=Path.home()/'uav-rgb-flight';source_root=Path(__file__).resolve().parents[2]
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.writable=self.output/'runtime';self.writable.mkdir()
        package_root=Path(checkpoints).resolve();spec=read(package_root/'package.json')
        if set(spec['files'])!={'model.pt','vision.json'}:raise ValueError('Unexpected inference assets')
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
        image=read(root/'receipts/model-stack.json')['image_id'];self.name='photo-map-'+str(os.getpid())
        self.identity=digest(package_root/'package.json')
        command=['docker','run','--rm','--name',self.name,'--gpus','all','--network','none','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}','--cpus','8','--shm-size','2g',
            '-v',str(source)+':/source:ro','-v',str(root/'assets/models')+':/models:ro',
            '-v',str(root/'deps/Metric3D')+':/upstream/metric3d:ro',
            '-v',str(package_root)+':/navigation:ro','-v',str(prior.root)+':/prior:ro',
            '-v',str(socket_path.parent)+':/ipc:ro','-v',str(self.writable)+':/output',
            '-w','/source',image,'python','-m','research.map_navigation.runtime','--episode-id',episode_id,'--variant',variant]
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
