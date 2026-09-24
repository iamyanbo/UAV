"""Guard-owned offline data preparation; privileged mounts never enter flight."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path.home()/'uav-rgb-flight';job=Path(os.environ['RGB_JOB_DIR'])
image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
name='rgb-learning-data-'+str(os.getpid())
try:
    raise SystemExit(subprocess.run(['docker','run','--rm','--name',name,'--gpus','all','--network','none',
        '--label','rgb-flight.job='+str(job),'--cap-drop','ALL','--security-opt','no-new-privileges',
        '--user',f'{os.getuid()}:{os.getgid()}','--cpus','6',
        '-v',str(root)+':'+str(root)+':ro','-v',str(job)+':'+str(job),
        '-v',str(root/'assets/models')+':/models:ro',image,'python',
        str(Path(__file__).with_name('prepare_learning_data.py')),*sys.argv[1:]]).returncode)
finally:subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
