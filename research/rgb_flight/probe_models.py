"""Run isolated component processes, without privileged scene/label mounts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from episode_store import verified_rgb_storage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--component',choices=['vjepa','policy','splat','qwen','tracking','all'],default='all')
    parser.add_argument('--tracking-frames',type=int,default=256)
    args=parser.parse_args()
    root=Path.home()/'uav-rgb-flight'
    observations=args.observations.resolve()
    if not observations.is_relative_to(root) or observations.name!='observations':
        raise ValueError('Only an observation directory may be mounted')
    if set(p.name for p in observations.iterdir())-{'rgb.zlib','frames.jsonl','storage.json','onboard-lossless.mkv','video.json','onboard-viewable.mp4','viewing-copy.json','color_calibration.json'}:
        raise ValueError('Unexpected files in observation-only directory')
    verified_rgb_storage(observations)
    job=Path(os.environ['RGB_JOB_DIR'])
    image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
    def terminate(signum,frame):
        raise KeyboardInterrupt('Model probe interrupted')
    signal.signal(signal.SIGTERM,terminate)
    components=['vjepa','policy','splat','qwen'] if args.component=='all' else [args.component]
    statuses=[]
    for component in components:
        name='rgb-model-probe-'+component+'-'+str(os.getpid())
        try:
            result=subprocess.run(['docker','run','--rm','--name',name,'--label','rgb-flight.job='+str(job),
                '--gpus','all','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
                '--user',str(os.getuid())+':'+str(os.getgid()),'--cpus','8','--shm-size','2g',
                '-e','HF_HUB_OFFLINE=1','-e','HF_HOME=/tmp/huggingface','-e','TORCH_HOME=/tmp/torch',
                '-v',str(Path(__file__).resolve().parent)+':/source:ro',
                '-v',str(root/'assets/models')+':/models:ro',
                '-v',str(root/'deps/vjepa2')+':/upstream/vjepa2:ro',
                '-v',str(root/'ports/Splat-SLAM')+':/upstream/splat:ro',
                '-v',str(observations)+':/observations:ro','-v',str(job)+':/output',
                image,'python',*(['/source/splat_tracking_probe.py','--frames',str(args.tracking_frames)] if component=='tracking' else ['/source/model_probe.py',component])])
            statuses.append(dict(component=component,return_code=result.returncode))
        finally:
            subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (job/'components.json').write_text(json.dumps(statuses,indent=2))
    return 0 if all(r['return_code']==0 for r in statuses) else 2


if __name__=='__main__':
    raise SystemExit(main())
