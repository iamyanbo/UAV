"""Archive Spark execution receipts without transferring weights or raw RGB."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import base64,json
from pathlib import Path
root=Path('/home/iamyanbo/uav-rgb-flight');selected=set()
run_patterns=('result.json','source_hashes.json','inference-source-hashes.json','stdout.log',
 'collection/*.json','training/result.json','training/metrics.jsonl',
 'visual-training/result.json','visual-training/metrics.jsonl','ppo/result.json','ppo/metrics.jsonl',
 'configurator/result.json','configurator/metrics.jsonl','configuration-preferences/*.json',
 'world-data/manifest.json','world-data/result.json','policy-data/manifest.json','policy-data/result.json',
 'navigation-replay/result.json','navigation-replay/manifest.json','navigation-replay/alignment.jsonl',
 'navigation-replay/reconstruction/result.json','navigation-replay/video/result.json',
 'odometry-qualification/result.json','odometry-qualification/episodes.jsonl')
for folder in root.joinpath('runs').iterdir():
 if folder.name<'20260924T030000Z':continue
 for pattern in run_patterns:selected.update(folder.glob(pattern))
for folder in root.joinpath('rounds').iterdir():
 if not folder.name.startswith(('learning-repair-','learning-continuation-','navigation-repair-','navigation-mapping-')):continue
 for pattern in ('*.json','execution/state.json','execution/*/result.json','execution/*/checkpoints.json',
                 'cumulative-window/*.json','cumulative-window/execution/state.json'):
  selected.update(folder.glob(pattern))
for folder in root.joinpath('launches').iterdir():
 if folder.name<'20260924T030000Z':continue
 for pattern in ('result.json','episode/result.json','episode/learned-controller/source_hashes.json',
                 'episode/learned-controller/runtime/result.json','episode/learned-controller/runtime/termination.json',
                 'episode/learned-controller/runtime/capacity.json','episode/learned-controller/runtime/deliberation/*.json',
                 'episode/learned-controller/runtime/alignment.jsonl',
                 'episode/learned-controller/runtime/deliberation/events.jsonl',
                 'episode/learned-controller/runtime/reconstruction/result.json',
                 'episode/learned-controller/runtime/reconstruction/memory/versions.jsonl',
                 'episode/learned-controller/runtime/video/result.json'):
  selected.update(folder.glob(pattern))
result=[]
for path in sorted(selected):
 if path.is_file() and path.stat().st_size<=2_000_000:
  try:data=path.read_bytes()
  except FileNotFoundError:continue
  result.append([str(path.relative_to(root)),base64.b64encode(data).decode('ascii')])
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='iamyanbo@10.31.12.8')
    parser.add_argument('--key',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'evidence/learning-repair-20260924')
    args=parser.parse_args()
    raw=subprocess.check_output(['ssh','-i',str(args.key),'-o','BatchMode=yes',args.host,
        shlex.join(['python3','-c',REMOTE])],text=True,timeout=90)
    args.output.mkdir(parents=True,exist_ok=True)
    for relative,encoded in json.loads(raw):
        destination=(args.output/relative).resolve()
        if not destination.is_relative_to(args.output.resolve()):raise ValueError('Escaping evidence path')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(base64.b64decode(encoded,validate=True))
    files=[]
    for path in sorted(args.output.rglob('*')):
        if not path.is_file() or path==args.output/'archive.json':continue
        raw=path.read_bytes()
        files.append(dict(path=path.relative_to(args.output).as_posix(),bytes=len(raw),
                          sha256=hashlib.sha256(raw).hexdigest()))
    archive=dict(observed_utc=datetime.now(timezone.utc).isoformat(),source_root='/home/iamyanbo/uav-rgb-flight',
        scope='Execution receipts and derived summaries; active counters are snapshots; raw RGB and weights remain on Spark',files=files)
    (args.output/'archive.json').write_text(json.dumps(archive,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),bytes=sum(row['bytes'] for row in files),observed_utc=archive['observed_utc'])))


if __name__=='__main__':main()
