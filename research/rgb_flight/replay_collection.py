"""Resolve one immutable collected attempt into the isolated runtime replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--collection',type=Path,required=True)
parser.add_argument('--checkpoints',type=Path,required=True)
parser.add_argument('--frames',type=int,default=400)
args=parser.parse_args()
rows=json.loads(args.collection.read_text())
if len(rows)!=1:raise ValueError('Choose exactly one preserved physical attempt for this replay stage')
episode=Path(rows[0]['episode_path'])
raise SystemExit(subprocess.run([sys.executable,str(Path(__file__).with_name('navigation_process.py')),
    '--observations',str(episode/'observations'),'--goal',str(episode/'goal'),
    '--checkpoints',str(args.checkpoints),'--frames',str(args.frames),
    '--integration-only','--video','--mapping']).returncode)
