"""Run physical braking calibration through the production flight launcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

output=Path(os.environ['RGB_JOB_DIR'])/'safety-calibration';output.mkdir()
process=subprocess.run([sys.executable,str(Path(__file__).with_name('spark_launch.py')),'--probe','dynamics'],
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(output/'launcher.log').write_text(process.stdout)
launches=[]
for line in process.stdout.splitlines():
    try:value=json.loads(line)
    except ValueError:continue
    if isinstance(value,dict) and 'launch' in value:launches.append(Path(value['launch']))
if len(launches)!=1:raise RuntimeError('Missing braking launch identity')
measurement=launches[0]/'engineering_only'
receipt=json.loads((measurement/'dynamics.json').read_text())
profile=measurement/'safety-profile.json'
if not profile.is_file():raise RuntimeError('Physical braking profile unavailable: '+str(receipt))
shutil.copyfile(profile,output/'safety-profile.json')
result=dict(status='completed',accepted=False,measurement=str(measurement),dynamics=receipt,
            launcher_return_code=process.returncode)
(output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
