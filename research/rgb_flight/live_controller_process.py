"""Host-owned isolated Mode 1 process; no evaluator mounts or network."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


class LiveControllerProcess:
    def __init__(self,socket_path,episode_id,checkpoints,output,maximum_speed,*,integration_only=False,sample_policy=False,with_deliberation=False,demonstrate=False):
        root=Path.home()/'uav-rgb-flight';checkpoints=Path(checkpoints).resolve();socket_path=Path(socket_path).resolve()
        if not checkpoints.is_relative_to(root.resolve()) or not socket_path.is_relative_to(root.resolve()):
            raise ValueError('Controller inputs must remain in the study workspace')
        manifest=checkpoints/'checkpoints.json';pack=json.loads(manifest.read_text())
        if not {'goal','odometry','projection','policy','safety'}<=pack['artifacts'].keys():
            raise ValueError('Complete trained Mode 1 checkpoint and safety artifacts required before flight')
        allowed={'checkpoints.json',*(item['path'] for item in pack['artifacts'].values())}
        if {p.name for p in checkpoints.iterdir()}!=allowed or any(p.is_symlink() or not p.is_file() for p in checkpoints.iterdir()):
            raise ValueError('Checkpoint namespace contains undeclared or nonregular files')
        for item in pack['artifacts'].values():
            if hashlib.sha256((checkpoints/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Modified controller artifact')
        if not integration_only and pack.get('accepted') is not True:raise ValueError('Deployment requires accepted checkpoint set')
        if {p.name for p in socket_path.parent.iterdir()}!={socket_path.name} or not socket_path.is_socket():
            raise ValueError('IPC namespace may contain only the runtime socket')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        source=self.output/'source';source.mkdir();writable=self.output/'runtime';writable.mkdir();self.writable=writable
        modules=('startup.py','runtime_capacity.py','learned_controller.py','controller.py','navigation_state.py','contracts.py','goal_matching.py','goal_io.py',
            'learning_models.py','metric_alignment.py','spatial_memory.py','episode_store.py','wire.py',
            'visual_encoder.py','action_intervals.py','async_video.py','async_mapping.py',
            'reconstruct.py','causal_pose.py','causal_mapper.py','mapping_worker.py','memory_snapshot.py',
            'live_deliberation.py','configurator.py','predictive_planner.py')
        hashes={}
        for name in modules:
            target=source/name;shutil.copyfile(Path(__file__).parent/name,target)
            hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        self.identity=hashlib.sha256(manifest.read_bytes()).hexdigest()
        (self.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
        image=json.loads((root/'receipts/model-stack.json').read_text())['image_id'];self.name='rgb-live-mode1-'+str(os.getpid())
        command=['docker','run','--rm','--name',self.name,'--label','rgb-flight.job='+os.environ['RGB_JOB_DIR'],
            '--gpus','all','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--user',f'{os.getuid()}:{os.getgid()}','--cpus','8','--shm-size','2g',
            '-v',str(source)+':/source:ro','-v',str(root/'assets/models')+':/models:ro',
            '-v',str(root/'deps/vjepa2')+':/upstream/vjepa2:ro',
            '-v',str(root/'ports/Splat-SLAM')+':/upstream/splat:ro',
            '-v',str(socket_path.parent)+':/ipc:ro','-v',str(checkpoints)+':/navigation:ro',
            '-v',str(writable)+':/output',image,'python','/source/learned_controller.py',
            '--episode-id',episode_id,'--maximum-speed-mps',str(maximum_speed),
            *(['--integration-only'] if integration_only else []),*(['--sample-policy'] if sample_policy else []),
            *(['--with-deliberation'] if with_deliberation else []),*(['--demonstrate'] if demonstrate else [])]
        (self.output/'request.json').write_text(json.dumps(dict(command=command,checkpoint_set_sha256=self.identity,
            episode_id=episode_id,mode='mode_1_geometry_safety_planner_disabled'),indent=2))
        self.log=(self.output/'stdout.log').open('x');self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT)

    def ready(self,timeout=180):
        deadline=time.monotonic()+timeout
        while not (self.writable/'CONTROLLER_READY').exists():
            self.check()
            if time.monotonic()>=deadline:raise RuntimeError('Live controller initialization deadline')
            time.sleep(.1)

    def check(self):
        if self.process.poll() is not None:raise RuntimeError('Live controller exited; inspect preserved inference receipt/log')

    def close(self):
        (self.writable/'CONTROLLER_STOP').touch()
        try:self.process.wait(timeout=160)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','-t','5',self.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.process.wait(timeout=15)
        finally:self.log.close()
        receipt=self.writable/'result.json'
        return json.loads(receipt.read_text()) if receipt.exists() else dict(status='failed',reason='Controller ended without receipt')
