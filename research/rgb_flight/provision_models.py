"""Provision the native ARM64 model stack; never mark model/flight gates passed."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def checked(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['image', 'sources', 'cuda'], required=True)
    args = parser.parse_args()
    root = Path.home() / 'uav-rgb-flight'
    receipts = root / 'receipts'
    receipts.mkdir(exist_ok=True)
    if args.step == 'image':
        tag = 'nvcr.io/nvidia/pytorch:25.11-py3'
        manifests = json.loads(checked(['docker', 'manifest', 'inspect', '--verbose', tag]))
        arm = [m for m in manifests if m['Descriptor']['platform']['architecture'] == 'arm64']
        if len(arm) != 1:
            raise RuntimeError('Expected exactly one native ARM64 manifest')
        digest = arm[0]['Descriptor']['digest']
        image = tag.split(':')[0] + '@' + digest
        receipt = dict(tag=tag, image=image, platform='linux/arm64', status='pulling')
        path = receipts / 'model-image.json'
        path.write_text(json.dumps(receipt, indent=2))
        subprocess.run(['docker', 'pull', '--quiet', '--platform', 'linux/arm64', image], check=True)
        metadata = json.loads(checked(['docker', 'image', 'inspect', image]))[0]
        if metadata['Architecture'] != 'arm64':
            raise RuntimeError('Image architecture mismatch')
        receipt.update(status='downloaded', image_id=metadata['Id'], size=metadata['Size'])
        path.write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    elif args.step == 'sources':
        lock_path = receipts / 'model-sources.json'
        lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        repos = {'Splat-SLAM': 'https://github.com/google-research/Splat-SLAM.git',
                 'vjepa2': 'https://github.com/facebookresearch/vjepa2.git',
                 'OpenFly-Platform': 'https://github.com/SHAILAB-IPEC/OpenFly-Platform.git',
                 'Metric3D': 'https://github.com/YvanYin/Metric3D.git'}
        for name, url in repos.items():
            target = root / 'deps' / name
            if not target.exists():
                subprocess.run(['git', 'clone', '--recursive', url, str(target)], check=True)
                if name == 'Metric3D':
                    revision=json.loads(Path(__file__).with_name('metric-vision.json').read_text())['source_revision']
                    subprocess.run(['git','-C',str(target),'checkout','--detach',revision],check=True)
            commit = checked(['git', '-C', str(target), 'rev-parse', 'HEAD']).strip()
            if name in lock and lock[name]['commit'] != commit:
                raise RuntimeError('Source revision changed: ' + name)
            lock[name] = dict(repository=url, commit=commit,
                              submodules=checked(['git', '-C', str(target), 'submodule', 'status', '--recursive']))
            lock_path.write_text(json.dumps(lock, indent=2))
            print(json.dumps({name: lock[name]}), flush=True)
    else:
        image = json.loads((receipts / 'model-image.json').read_text())['image']
        name = 'rgb-cuda-check-' + str(os.getpid())
        source = Path(__file__).resolve().parent
        job = Path(os.environ.get('RGB_JOB_DIR', receipts))
        def terminate(signum, frame):
            raise KeyboardInterrupt('Job terminated')
        signal.signal(signal.SIGTERM, terminate)
        try:
            subprocess.run(['docker', 'run', '--rm', '--name', name, '--gpus', 'all',
                            '--label','rgb-flight.job='+str(job),
                            '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                            '--user', str(os.getuid()) + ':' + str(os.getgid()),
                            '-v', str(source) + ':/source:ro', '-v', str(job) + ':/output',
                            image, 'python', '/source/native_cuda_check.py'], check=True)
            (receipts / 'native-cuda.json').write_bytes((job / 'native-cuda.json').read_bytes())
        finally:
            subprocess.run(['docker', 'stop', '-t', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
