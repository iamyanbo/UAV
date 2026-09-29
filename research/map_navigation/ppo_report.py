"""Complete-flight outcomes remain separate from auxiliary exercises and updates."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from .common import read,write


def interval(success,total):
    if not total:return None
    p=success/total;z=1.96;d=1+z*z/total
    middle=(p+z*z/(2*total))/d;radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,middle-radius),min(1,middle+radius)]


def summarize(root):
    root=Path(root);groups=defaultdict(list)
    for path in root.glob('train-*-outcome.json'):
        row=read(path);groups[(row['kind'],row.get('distance_bin'),row.get('difficulty'))].append(row)
    outcomes=[]
    for key,rows in sorted(groups.items(),key=lambda r:str(r[0])):
        counts=Counter(r['event'] for r in rows);n=len(rows)
        outcomes.append(dict(kind=key[0],distance_bin=key[1],difficulty=key[2],episodes=n,outcomes=dict(counts),
            success_rate=counts['success']/n,success_95_interval=interval(counts['success'],n),
            collision_rate=counts['collision']/n,collision_95_interval=interval(counts['collision'],n),
            mean_duration_s=sum(r['duration_s'] for r in rows)/n,mean_reset_s=sum(r['reset_wall_s'] for r in rows)/n))
    updates=[json.loads(s) for s in (root/'updates.jsonl').read_text().splitlines()] if (root/'updates.jsonl').exists() else []
    status=read(root/'status.json') if (root/'status.json').exists() else None
    failure=read(root/'failure.json') if (root/'failure.json').exists() else None
    write(root/'morning-report.json',dict(schema='ppo-overnight-report/v1',status=status,failure=failure,outcomes=outcomes,
        updates=updates,targets=dict(success_rate=.9,collision_rate=.01),navigation_acceptance=False,
        world_selector=False,photo_slam_claim=False,latency_note='Training optimization pauses excluded from physical flight duration; continuous evaluation reported separately',
        validation=[read(p) for p in sorted(root.glob('evaluation-*.json'))]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();summarize(a.run)
