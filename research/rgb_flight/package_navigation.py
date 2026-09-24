"""Create an immutable inference-only checkpoint namespace from trained artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def checksum(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('goal','odometry'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('world','policy','projection','safety','qwen','configuration'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--goal-match-threshold',type=float,required=True)
    args=parser.parse_args()
    if not 0<args.goal_match_threshold<=1:raise ValueError('Invalid recognition threshold')
    for role in ('goal','odometry','world','policy','projection','safety','qwen','configuration'):
        source=getattr(args,role)
        if source is not None and not source.is_file():raise ValueError('Missing checkpoint: '+str(source))
    args.output.mkdir(parents=True,exist_ok=False);artifacts={}
    for role in ('goal','odometry','world','policy','projection','safety','qwen','configuration'):
        source=getattr(args,role)
        if source is None:continue
        target=args.output/(role+('.json' if role in ('safety','configuration') else '.pt'));shutil.copyfile(source,target)
        artifacts[role]=dict(path=target.name,sha256=checksum(target),source=str(source.resolve()))
        if artifacts[role]['sha256']!=checksum(source):raise ValueError('Source checkpoint changed during copy')
    spec=dict(schema='visual-navigation-checkpoints/v1',status='completed',accepted=False,artifacts=artifacts,
        goal_match_threshold=args.goal_match_threshold,
        scope='Integration snapshot. Packaging does not establish model compatibility or acceptance.')
    (args.output/'checkpoints.json').write_text(json.dumps(spec,indent=2));print(json.dumps(spec))


if __name__=='__main__':main()
