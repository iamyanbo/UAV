"""Bounded offline artifact utilities in the pinned native model image.

This is an engineering utility, NOT the inference launcher (which grants only
the broker socket/model assets). Mounts must stay inside the study workspace.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--script',choices=['archive_video.py','splat_raster_check.py','visual_contract_check.py',
                                            'obstacle_field.py','episode_generation.py','build_visual_dataset.py'],required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--field',type=Path)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    root=Path.home()/'uav-rgb-flight'
    image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
    mounts=[]
    if args.data:
        data=args.data.resolve()
        if not data.is_relative_to(root.resolve()) or not data.is_dir():
            raise ValueError('Data must be an existing study directory')
        mounts=['-v',str(data)+':/data']
    if args.script=='obstacle_field.py':
        if not args.data or not (data/'captures.jsonl').is_file():
            raise ValueError('Field fusion requires a completed depth/semantic capture directory')
        if (data.parent/'obstacle-field.npz').exists():
            raise ValueError('Refusing to overwrite an existing obstacle field')
        mounts=['-v',str(data.parent)+':/survey']
    if args.script=='episode_generation.py':
        field = args.field.resolve() if args.field else None
        if field is None or not field.is_relative_to(root.resolve()) or not field.is_file():
            raise ValueError('Manifest generation requires a study-local obstacle-field file')
        if (field.parent/'manifests').exists():
            raise ValueError('Refusing to overwrite an existing deterministic manifest')
        mounts=['-v',str(field.parent)+':/survey']
    job=Path(os.environ['RGB_JOB_DIR'])
    name='rgb-offline-'+str(os.getpid())
    def terminate(signum,frame):
        raise KeyboardInterrupt('Job interrupted')
    signal.signal(signal.SIGTERM,terminate)
    try:
        subprocess.run(['docker','run','--rm','--name',name,'--label','rgb-flight.job='+os.environ.get('RGB_OWNING_JOB_DIR',str(job)),
            '--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--user',str(os.getuid())+':'+str(os.getgid()),'--cpus','4',
            *(['--gpus','all'] if args.gpu else []),
            '-v',str(Path(__file__).resolve().parent)+':/source:ro','-v',str(job)+':/output',*mounts,
            image,'python','/source/'+args.script,
            *(['--captures','/survey/'+data.name,'--output','/survey/obstacle-field.npz']
              if args.script=='obstacle_field.py' else
              ['--obstacle-field','/survey/'+field.name,'--output','/survey/manifests']
              if args.script=='episode_generation.py' else
              ['--collection-root','/data'] if args.script=='build_visual_dataset.py' else
              ['/data'] if args.data else [])],check=True)
    finally:
        subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__':
    main()
