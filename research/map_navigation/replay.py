"""Deferred recorded-RGB replay; never called flight or deployment evidence."""
import json
from pathlib import Path
import time
import numpy as np
from episode_store import frames,verified_rgb_storage
from goal_io import load_goal
from .runtime import Navigator
from .common import write,digest


def replay(episode,package,map_folder,output,variant,window,perception_only=False):
    episode=Path(episode);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    verified_rgb_storage(episode/'observations');goal=load_goal(episode/'goal')
    views=[np.frombuffer(raw,np.uint8).reshape(480,640,3).copy() for raw in goal.rgb_views]
    navigator=Navigator(package,map_folder,views,variant);count=0
    try:
        with (output/'decisions.jsonl').open('x') as stream:
            for metadata,rgb in frames(episode/'observations'):
                if not window.remaining():break
                metadata=dict(metadata,received_monotonic=time.monotonic())
                decision=navigator.step(metadata,rgb,perception_only=perception_only)
                stream.write(json.dumps(decision,allow_nan=False)+'\n');count+=1
    finally:navigator.close()
    write(output/'receipt.json',dict(source_episode=str(episode.resolve()),checkpoint_sha256=digest(Path(package)/'model.pt'),package_sha256=digest(Path(package)/'package.json'),decisions_sha256=digest(output/'decisions.jsonl'),perception_only=perception_only,status='replay_finished',frames=count,flight_evidence=False,
          closed_loop=False,clock='original observation timestamps; wall inference timing measured during replay'))
