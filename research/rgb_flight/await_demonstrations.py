"""Dependency barrier for an already running immutable collection job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

parser=argparse.ArgumentParser();parser.add_argument('--collection',type=Path,required=True)
args=parser.parse_args();job=Path(os.environ['RGB_JOB_DIR'])
receipt=args.collection.parent/'result.json';guard=args.collection.parent.parent/'result.json'
while not receipt.exists():
    if guard.exists():raise RuntimeError('Collection guard ended without completed demonstrations')
    if (job/'CHECKPOINT_REQUEST').exists():raise RuntimeError('Collection wait checkpointed; resume in new window')
    time.sleep(2)
value=json.loads(receipt.read_text());rows=json.loads(args.collection.read_text())
if value.get('status')!='completed' or len(rows)!=10:raise ValueError('Ten completed demonstrations required')
result=dict(status='completed',accepted=False,collection_sha256=hashlib.sha256(args.collection.read_bytes()).hexdigest(),
            completed_demonstrations=len(rows))
(job/'demonstrations-ready.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
