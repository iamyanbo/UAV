"""Runs inside the actual restricted container against the live RGB broker."""
import argparse
import json
from pathlib import Path
import socket
import time

from wire import BrokerClient

parser = argparse.ArgumentParser()
parser.add_argument('--episode-id', required=True)
parser.add_argument('--host-canary', required=True)
args = parser.parse_args()
for path in (args.host_canary, '/var/run/docker.sock', '/privileged', '/home/iamyanbo/uav-rgb-flight/scene-original'):
    if Path(path).exists():
        raise RuntimeError('Privileged path visible in inference namespace: ' + path)
with socket.socket() as stream:
    stream.settimeout(.2)
    if stream.connect_ex(('127.0.0.1',41451)) == 0:
        raise RuntimeError('Simulator RPC reachable from inference namespace')
channel = BrokerClient('/ipc/rgb.sock', args.episode_id)
for op in ('getMultirotorState','simGetVehiclePose','simGetImages','labels','reset'):
    try:
        channel.request(op)
    except RuntimeError:
        continue
    raise RuntimeError('Privileged operation accepted: ' + op)
first, rgb = channel.observe()
if len(rgb) != 640*480*3 or set(first) != {'episode_id','frame_id','sim_ns','received_monotonic','request_started_monotonic','calibration','command_history'}:
    raise RuntimeError('Unexpected RGB-only response schema')
goal_views=0
try:
    panorama=[]
    for index in range(4):
        metadata,goal_rgb=channel.goal_view(index)
        if len(goal_rgb)!=640*480*3 or metadata['index']!=index or 'panorama_sha256' not in metadata:
            raise RuntimeError('Unexpected coordinate-free goal schema')
        panorama.append(metadata['panorama_sha256']); goal_views+=1
    if len(set(panorama))!=1:
        raise RuntimeError('Goal views do not share a panorama identity')
except RuntimeError as error:
    if str(error)!='Goal panorama unavailable':
        raise
channel.command(first['frame_id'],[0,0,0,0])
time.sleep(.3)
last, _ = channel.observe(first['frame_id'])
if last['sim_ns'] <= first['sim_ns']:
    raise RuntimeError('Simulation did not advance')
Path('/output/isolation.json').write_text(json.dumps(dict(status='live_inference_boundary_passed',
    episode_id=args.episode_id, rgb_bytes=len(rgb), first_sim_ns=first['sim_ns'],last_sim_ns=last['sim_ns'],
    goal_views=goal_views,denied_operations=5, simulator_rpc_inaccessible=True, privileged_files_inaccessible=True),indent=2))
