"""Run actual frozen Qwen on recorded RGB after flight collection releases the GPU."""
import gc
import os
from pathlib import Path
import socket
import time
from PIL import Image
import torch
from photo_goal.common import read,write
from photo_goal.mission_storage import configure
from photo_goal.mission_contracts import city_config
from photo_goal.mission_resources import Resources,RunWindow
from photo_goal.mission_mode2 import FrozenCityQwen,pack_request

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native');configure(root)
window=RunWindow(3);resources=Resources(root,city_config(),'cuda')
preceding=os.environ.get('UAV_PRECEDING_PID')
while preceding and Path('/proc/'+preceding).exists():
    if not window.admits(120):raise RuntimeError('Preceding collection window has not finished')
    time.sleep(2)
while window.admits(120):
    try:
        with socket.create_connection(('127.0.0.1',8990),timeout=.5):pass
    except OSError:break
    time.sleep(10)
else:raise RuntimeError('Physical collection did not release the GPU inside admission window')
resources.check(growth_bytes=10*2**30,disk=False)
output=root/'runs/qwen-admission.json';receipt=dict(schema='photo-goal-qwen-admission/v1',passed=False)
try:
    data=read(root/'data/visual-bootstrap/clips.json')
    clips=[r for r in data['clips'] if r['split']=='train']
    current=clips[0];goal=next(r for r in clips if r['episode']!=current['episode'])
    paths=[root/'data/visual-bootstrap'/r['frames'][-1]['image'] for r in (current,goal)]
    images=[Image.open(path).convert('RGB') for path in paths]
    stamp=current['end_ns']/1e9
    references=[dict(id=name,source='image_region',image=0,frame_id=0,rgb_path=str(paths[0]),roi=roi,stamp=stamp)
        for name,roi in [('left',[0,0,0.5,1]),('center',[0.25,0,0.75,1]),('right',[0.5,0,1,1])]]
    model=FrozenCityQwen(root/'assets/models/qwen2.5-vl-3b')
    resources.check(growth_bytes=1024**3,disk=False)
    torch.cuda.reset_peak_memory_stats()
    response=model.generate(pack_request(images,references,dict(tried=[],localization='unknown',revision=0,remaining_s=300,purpose='recorded RGB component admission'),
                                         'recorded-admission',0,stamp,'visual-bootstrap'))
    receipt.update(passed=True,response=response,resources=resources.check(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),frozen=True,
                   navigation_performance_measured=False,live_flight_concurrency_measured=False)
except Exception as error:
    receipt.update(error=type(error).__name__+': '+str(error),raw=getattr(locals().get('model'),'last_generation',None),
                   resources=resources.check(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    raise
finally:
    write(output,receipt)
    print({k:v for k,v in receipt.items() if k!='response'},flush=True)
    gc.collect();torch.cuda.empty_cache()
