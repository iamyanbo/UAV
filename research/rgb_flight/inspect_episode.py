"""Summarize measured timing and trajectory from a retained engineering episode."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('episode',type=Path)
args=parser.parse_args()
rows=[json.loads(line) for line in (args.episode/'observations/frames.jsonl').read_text().splitlines()]
states=json.loads((args.episode/'engineering_only/states.json').read_text())
times=np.array([r['sim_ns'] for r in rows],dtype=np.int64)/1e9
wall=np.array([r['received_monotonic'] for r in rows])
requests=np.array([r['request_started_monotonic'] for r in rows])
positions=np.array([r['position'] for r in states])
result=dict(frames=len(rows),sim_duration=times[-1]-times[0],wall_duration=wall[-1]-wall[0],
            request_quantiles=np.quantile(wall-requests,[.5,.9,.95,.99,1]).tolist(),
            sim_interval_quantiles=np.quantile(np.diff(times),[.5,.9,.95,.99,1]).tolist(),
            interrequest_gap_quantiles=np.quantile(requests[1:]-wall[:-1],[.5,.9,.95,.99,1]).tolist(),
            physical_distance_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
            start=positions[0].tolist(),end=positions[-1].tolist())
print(json.dumps(result,indent=2))
