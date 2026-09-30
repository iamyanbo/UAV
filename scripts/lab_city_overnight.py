"""One bounded qualified city-training window; no retries or research supervisor."""
import argparse
from datetime import datetime,timezone
import json
import os
import resource
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
from photo_goal.common import read,write,digest
from photo_goal.mission_storage import configure
from photo_goal.mission_contracts import city_config
from photo_goal.native_full_training import checked_assets,implementation_identity

parser=argparse.ArgumentParser();parser.add_argument('--replace-service-pid',type=int,default=0)
args=parser.parse_args()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path('/mnt/hdd2/yanbocheng/photo-goal-native');configure(root)
active=root/'runs/active-city-window.json'
if active.exists():
    previous_window=Path('/proc')/str(read(active)['pid'])/'cmdline'
    if previous_window.exists() and b'lab_city_overnight.py' in previous_window.read_bytes():
        raise RuntimeError('A city window is already running; inspect active-city-window.json')
cfg=city_config();survey=root/'data/visual-bootstrap/rgb-atlas.json'
checked_assets(root,survey,cfg)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
job=root/'runs'/('city-window-'+stamp);job.mkdir()
snapshot=job/'source';shutil.copytree(root/'code/photo_goal',snapshot/'photo_goal',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(root/'code/sitecustomize.py',snapshot/'sitecustomize.py')
shutil.copyfile(root/'code/lab_training_metrics.py',snapshot/'lab_training_metrics.py')
env=dict(os.environ,PYTHONPATH=str(snapshot),UAV_WINDOW_STOP_FILE=str(job/'stop-requested'),
         UAV_GPU_FRACTION_CEILING='1.0',UAV_RESOURCE_LOG='1')
python=str(root/'env/bin/python');checkpoint=root/'city-training/latest.pt'
if not checkpoint.exists():checkpoint=root/'checkpoints/city-initialized.pt'
plan=dict(schema='photo-goal-lab-window/v1',status='starting',pid=os.getpid(),started_utc=stamp,
    maximum_hours=8,project_root=str(root),gpu=0,folder_ceiling_gib=128,
    qualification_sha256=digest(root/'city-qualification.json'),implementation_sha256=implementation_identity(),
    checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),source=str(snapshot))
plan['environment_startup_sha256']=digest(snapshot/'sitecustomize.py')
plan['resource_policy']=dict(gpu_fraction_ceiling=1.0,configured_default=cfg['resources']['gpu_fraction_ceiling'],
    authorization='User authorized the whole of GPU 0; GPU 1 remains untouched',
    resource_module_sha256=digest(snapshot/'photo_goal/mission_resources.py'),memory_logging=True)
plan['metrics_script_sha256']=digest(snapshot/'lab_training_metrics.py')
write(job/'launch.json',plan);write(root/'runs/active-city-window.json',dict(job=str(job),pid=os.getpid()))
qwen=None;trainer=None;metrics=None
def interrupted(signum,frame):raise KeyboardInterrupt('Window shutdown requested')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    previous=Path('/proc')/str(args.replace_service_pid)/'cmdline'
    if previous.exists():
        command=previous.read_bytes()
        if b'serve-city-qwen' not in command or str(root).encode() not in command:
            raise RuntimeError('Replacement PID is not this study\'s Qwen service')
        os.kill(args.replace_service_pid,signal.SIGTERM)
        until=time.monotonic()+20
        while previous.exists() and time.monotonic()<until:time.sleep(.5)
    qwen_log=(job/'qwen.log').open('w')
    qwen=subprocess.Popen([python,'-u','-m','photo_goal','--root',str(root),'serve-city-qwen',
        '--qwen-model',str(root/'assets/models/qwen2.5-vl-3b'),'--authfile',str(root/'campaign/qwen-auth.bin')],
        cwd=root,env=env,stdout=qwen_log,stderr=subprocess.STDOUT,start_new_session=True)
    until=time.monotonic()+180
    while time.monotonic()<until:
        if qwen.poll() is not None:raise RuntimeError('Qwen service failed; see '+str(job/'qwen.log'))
        # Inspect the listening socket without interrupting its authenticated handshake.
        if subprocess.check_output(['ss','-H','-ltn','sport','=',':48005'],text=True).strip():break
        time.sleep(1)
    else:raise RuntimeError('Qwen service readiness timed out')
    train_log=(job/'training.log').open('w')
    trainer=subprocess.Popen([python,'-u','-m','photo_goal','--root',str(root),'train-city',
        '--checkpoint',str(checkpoint),'--backbone',str(root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt'),
        '--survey',str(survey),'--qwen-auth',str(root/'campaign/qwen-auth.bin'),
        '--hours','7.9','--batches','244'],cwd=root,env=env,stdout=train_log,stderr=subprocess.STDOUT,start_new_session=True)
    plan.update(status='running',trainer_pid=trainer.pid,qwen_pid=qwen.pid);write(job/'launch.json',plan)
    metrics_env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    metrics_log=(job/'metrics.log').open('w')
    metrics=subprocess.Popen([python,'-u',str(snapshot/'lab_training_metrics.py'),'--watch','--hours','8'],
        cwd=root,env=metrics_env,stdout=metrics_log,stderr=subprocess.STDOUT,start_new_session=True)
    plan.update(metrics_pid=metrics.pid,metrics_dashboard=str(root/'runs/training-metrics/dashboard.html'))
    write(job/'launch.json',plan)
    print(json.dumps(plan),flush=True)
    next_size=0;last_counts=None
    while trainer.poll() is None:
        if qwen.poll() is not None:raise RuntimeError('Frozen guidance service ended during collection')
        if time.monotonic()>=next_size:
            size=int(subprocess.check_output(['du','-sb',str(root)],text=True,timeout=60).split()[0])
            if size>126*2**30:
                (job/'stop-requested').write_text('Project size reached checkpoint reserve below 128 GiB ceiling.\n')
            next_size=time.monotonic()+60
        ledger=root/'campaign/overnight.sqlite'
        if ledger.exists():
            try:
                with sqlite3.connect('file:'+str(ledger)+'?mode=ro',uri=True,timeout=1) as db:
                    counts=db.execute("SELECT COALESCE(SUM(actual),0),COALESCE(SUM(optimized),0) FROM batches WHERE id NOT LIKE 'qualification-%'").fetchone()
                if counts!=last_counts:
                    print(json.dumps(dict(physical_ppo_rows_observed=counts[0],accepted_ppo_rows=counts[1],root_bytes=size)),flush=True)
                    last_counts=counts
            except sqlite3.Error:pass
        time.sleep(10)
    plan.update(status='finished' if trainer.returncode==0 else 'failed',trainer_exit=trainer.returncode)
    if trainer.returncode:raise RuntimeError('Training ended with failure; checkpoint/evidence retained')
except BaseException as error:
    plan.update(status='failed',error=type(error).__name__+': '+str(error))
    raise
finally:
    if trainer and trainer.poll() is None:
        os.killpg(trainer.pid,signal.SIGINT)
        try:trainer.wait(timeout=120)
        except subprocess.TimeoutExpired:os.killpg(trainer.pid,signal.SIGTERM);trainer.wait(timeout=30)
    if qwen and qwen.poll() is None:
        os.killpg(qwen.pid,signal.SIGTERM);qwen.wait(timeout=30)
    write(job/'launch.json',plan)
    if metrics and metrics.poll() is None:
        metrics.terminate()
        try:metrics.wait(timeout=15)
        except subprocess.TimeoutExpired:metrics.kill();metrics.wait(timeout=5)
    # Capture the terminal checkpoint even if the observer missed the last update.
    if metrics:
        try:
            subprocess.run([python,'-u',str(snapshot/'lab_training_metrics.py')],cwd=root,env=metrics_env,
                stdout=metrics_log,stderr=subprocess.STDOUT,timeout=30,check=True)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
            plan['metrics_final_error']=str(error);write(job/'launch.json',plan)
