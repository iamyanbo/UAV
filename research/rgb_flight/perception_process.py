"""Host lifecycle for isolated live RGB perception, with no label mounts."""
import json
import os
from pathlib import Path
import subprocess
import time


class PerceptionProcess:
    def __init__(self, ipc, output, episode_id):
        root=Path.home()/'uav-rgb-flight'
        image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
        self.output=Path(output)
        self.output.mkdir(exist_ok=False)
        Path(ipc).mkdir(parents=True,exist_ok=True)
        self.name='rgb-live-perception-'+str(os.getpid())
        self.log=(self.output/'container.log').open('w')
        command=['docker','run','--rm','--name',self.name,'--label','rgb-flight.job='+os.environ['RGB_JOB_DIR'],
                 '--gpus','all','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
                 '--user',f'{os.getuid()}:{os.getgid()}','--cpus','8','--shm-size','2g',
                 '-e','HF_HUB_OFFLINE=1','-e','HF_HOME=/tmp/huggingface','-e','TORCH_HOME=/tmp/torch',
                 '-v',str(Path(__file__).resolve().parent)+':/source:ro',
                 '-v',str(root/'assets/models')+':/models:ro',
                 '-v',str(root/'deps/vjepa2')+':/upstream/vjepa2:ro',
                 '-v',str(root/'ports/Splat-SLAM')+':/upstream/splat:ro',
                 '-v',str(ipc)+':/ipc:ro','-v',str(self.output)+':/output',
                 image,'python','/source/live_perception.py','--episode-id',episode_id]
        self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT)

    def wait_ready(self):
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Live perception exited before readiness; inspect retained logs')
            if (self.output/'READY').exists():
                return json.loads((self.output/'READY').read_text())
            time.sleep(.2)
        raise RuntimeError('Live perception load deadline exceeded')

    def request_stop(self):
        (self.output/'PERCEPTION_STOP').touch()

    def coverage(self):
        counts = {}
        for component in ('vjepa', 'qwen'):
            receipt = json.loads((self.output/component/'result.json').read_text())
            counts[component] = receipt['outputs']
        tracking = json.loads((self.output/'reconstruction/result.json').read_text())
        counts['initialized_tracking'] = tracking['tracking_initialized_frames']
        counts['gaussian_memory_versions'] = tracking['memory_versions']
        return dict(counts=counts, all_components_produced_outputs=all(value > 0 for value in counts.values()),
                    scope='Nonempty outputs only; accuracy, scale, latency and route diversity remain separate gates')

    def close(self):
        self.request_stop()
        try:
            self.process.wait(timeout=135)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','-t','10',self.name],check=True,timeout=20)
            self.process.wait(timeout=10)
        finally:
            self.log.close()
        if self.process.returncode:
            raise RuntimeError('Live perception did not finish cleanly; inspect retained component logs')
        return json.loads((self.output/'result.json').read_text())
