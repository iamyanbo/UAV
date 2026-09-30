"""First real renderer/control flight receipt; does not publish PPO qualification."""
import json
from pathlib import Path
import time
from PIL import Image
from photo_goal.mission_storage import configure
from photo_goal.mission_contracts import city_config
from photo_goal.mission_resources import Resources
from photo_goal.project_city import ProjectCity
from photo_goal.common import write

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native')
configure(root)
resources=Resources(root,city_config(),'cuda')
output=root/'runs/first-linux-flight'
receipt=dict(schema='photo-goal-linux-flight/v1',passed=False,navigation_success=False)
try:
    resources.check()
    with ProjectCity(root,output) as session:
        resources.check()
        before=session.state()
        print(json.dumps(dict(state=before)),flush=True)
        session.reset([0,0,-25],0)
        frames=[]
        for index in range(30):
            session.command([.5,0,0,5],duration=.3)
            rgb,frame=session.image()
            frame['path']=str(output/f'{index:04d}.png')
            Image.fromarray(rgb).save(frame['path'])
            frames.append(frame)
            if index==0:print(json.dumps(dict(first_frame={**frame,'texture_std':float(rgb.std())})),flush=True)
            time.sleep(.1)
        session.command([0,0,0,0],duration=1)
        time.sleep(1)
        after=session.state()
        receipt.update(passed=True,frames=frames,before=before,after=after,
                       commands=session.command_receipts,collisions=session.collision_events,
                       resources=resources.check(),scope='real rendered continuous-motion diagnostic; not navigation qualification')
except Exception as error:
    receipt['error']=type(error).__name__+': '+str(error)
    raise
finally:
    write(output/'flight-receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('frames','commands','collisions')}),flush=True)
