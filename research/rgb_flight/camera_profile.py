"""Compare lossless camera transport under concurrent physical command/state RPC."""
import argparse
import io
import json
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import airsim
import numpy as np
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--expected-settings',type=Path,required=True)
args=parser.parse_args()
output=args.output/'engineering_only'
output.mkdir()
client=airsim.MultirotorClient(timeout_value=10)
vehicle='drone_1'
assert json.loads(client.getSettingsString())==json.loads(args.expected_settings.read_text())
assert not client.simIsPause()
time.sleep(5)
client.enableApiControl(True,vehicle)
client.armDisarm(True,vehicle)
client.takeoffAsync(timeout_sec=15,vehicle_name=vehicle).get()
client.moveByVelocityBodyFrameAsync(0,0,-1,5,vehicle_name=vehicle).get()
client.hoverAsync(vehicle_name=vehicle).get()
stop=threading.Event()
errors=[]


def commands():
    control=airsim.MultirotorClient(timeout_value=5)
    try:
        while not stop.is_set():
            started=time.monotonic()
            control.getMultirotorState(vehicle_name=vehicle)
            control.simGetCollisionInfo(vehicle_name=vehicle)
            control.moveByVelocityBodyFrameAsync(0,0,0,.15,vehicle_name=vehicle)
            stop.wait(max(0,.05-(time.monotonic()-started)))
    except Exception as error:
        errors.append(type(error).__name__+': '+str(error))


worker=threading.Thread(target=commands,daemon=True)
worker.start()
results=[]
try:
    for label,compressed in [('raw_before',False),('png',True),('raw_parallel',False),('raw_after',False)]:
        rows=[]
        deadline=time.monotonic()+15
        local=threading.local()
        def request_image():
            if not hasattr(local,'client'):
                local.client=airsim.MultirotorClient(timeout_value=10)
            started=time.monotonic()
            image=local.client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],vehicle_name=vehicle)[0]
            return started,time.monotonic(),image
        parallel=ThreadPoolExecutor(max_workers=2) if label=='raw_parallel' else None
        pending={parallel.submit(request_image) for _ in range(2)} if parallel else set()
        late=0
        while time.monotonic()<deadline:
            if parallel:
                done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                if not done:
                    raise RuntimeError('Pipelined camera timed out')
                responses=sorted((future.result() for future in done),key=lambda item:item[1])
                pending.update(parallel.submit(request_image) for _ in done)
            else:
                start=time.monotonic()
                response=client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,compressed)],vehicle_name=vehicle)[0]
                responses=[(start,time.monotonic(),response)]
            for start,received,response in responses:
                if rows and response.time_stamp<=rows[-1]['sim_ns']:
                    late+=1
                    continue
                if compressed:
                    image=Image.open(io.BytesIO(response.image_data_uint8)).convert('RGB')
                    rgb=image.tobytes()
                else:
                    rgb=bytes(response.image_data_uint8)
                if len(rgb)!=640*480*3 or (response.width,response.height)!=(640,480):
                    raise RuntimeError('Lossless RGB shape mismatch')
                if not rows:
                    Image.frombytes('RGB',(640,480),rgb).save(output/(label+'.png'))
                rows.append(dict(sim_ns=response.time_stamp,start=start,received=received,decoded=time.monotonic(),bytes=len(response.image_data_uint8)))
        if parallel:
            parallel.shutdown(wait=True,cancel_futures=True)
        dt=np.diff([r['sim_ns'] for r in rows])/1e9
        results.append(dict(mode=label,frames=len(rows),late_or_duplicate_responses=late,interval_p95_sim_seconds=float(np.quantile(dt,.95)),
                            interval_max_seconds=float(dt.max()),passed=bool(np.all(dt>0) and np.quantile(dt,.95)<=.05),
                            mean_payload_bytes=float(np.mean([r['bytes'] for r in rows]))))
        (output/(label+'.json')).write_text(json.dumps(rows))
finally:
    stop.set(); worker.join(timeout=7)
    receipt=dict(scope='camera transport diagnostic at hover; no navigation/model feasibility claim',results=results,errors=errors)
    (output/'camera-profile.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
raise SystemExit(0 if results and any(r['passed'] for r in results) and not errors else 2)
