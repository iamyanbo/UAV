"""One restricted container, separate native processes for upstream namespaces."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode-id',required=True)
    args=parser.parse_args()
    root=Path('/output')
    commands=[('tracker',[sys.executable,'/source/reconstruct.py','--broker-socket','/ipc/rgb.sock',
                         '--episode-id',args.episode_id,'--asynchronous-map']),
              ('vjepa',[sys.executable,'/source/live_visual.py','--component','vjepa','--episode-id',args.episode_id]),
              ('qwen',[sys.executable,'/source/live_visual.py','--component','qwen','--episode-id',args.episode_id])]
    children=[]
    try:
        for component,command in commands:
            stream=(root/(component+'.log')).open('w')
            children.append((component,subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT),stream))
        while not (root/'PERCEPTION_STOP').exists():
            failed=[(name,child.returncode) for name,child,_ in children if child.poll() is not None]
            if failed:
                raise RuntimeError('Live perception process exited: '+str(failed))
            if all((root/(name+'.ready')).exists() for name,_,_ in children):
                if not (root/'READY').exists():
                    (root/'READY').write_text(json.dumps(dict(episode_id=args.episode_id,components=[name for name,_,_ in children],monotonic=time.monotonic())))
            time.sleep(.2)
    finally:
        (root/'PERCEPTION_STOP').touch()
        deadline=time.monotonic()+120
        for _,child,_ in children:
            try:
                child.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
        for _,_,stream in children:
            stream.close()
        result=dict(components={name:child.returncode for name,child,_ in children},
                    ready=(root/'READY').exists(),scope='live RGB-only perception resource foundation; not trained navigation')
        (root/'result.json').write_text(json.dumps(result,indent=2))
    return 0 if all(child.returncode==0 for _,child,_ in children) else 2


if __name__=='__main__':
    raise SystemExit(main())
