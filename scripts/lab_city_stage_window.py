"""Qualify then train one already-forked city phase within one eight-hour window."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from photo_goal.common import contained, read, write, digest
from photo_goal.mission_storage import configure
from photo_goal.mission_resources import Resources
from photo_goal.mission_contracts import city_config
from photo_goal.mission_space import guard, reserve_write, shutdown_writes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    args=parser.parse_args()
    if not 0<args.hours<=8:parser.error('Use an eight-hour maximum')
    root=args.root.resolve();source=contained(root,args.source);run=contained(root,args.run_dir)
    configure(root);Resources(root,city_config(run/'config.json'),'cpu').check()
    lock=(root/'runs/stage-window.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'active-job.lock').exists():raise RuntimeError('An existing flight job owns this project')
    guard().reconcile_quiescent()
    for port in (48005,8989,8990):
        if subprocess.check_output(['ss','-H','-ltn','sport','=',':'+str(port)],text=True).strip():
            raise RuntimeError('Existing endpoint is occupied; preserve its owner: '+str(port))
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    gpu=next(line.split(',') for line in memory.splitlines() if line.split(',')[0].strip()=='0')
    uuid=gpu[1].strip()
    # Existing desktop/Gazebo graphics may coexist; never replace a compute owner.
    if any(line.split(',')[0].strip()==uuid for line in compute.splitlines()):
        raise RuntimeError('GPU 0 has an existing compute workload; preserve it')
    if float(gpu[3])-float(gpu[2])<18*1024:raise RuntimeError('Insufficient measured GPU 0 headroom')
    started=time.time();deadline=started+args.hours*3600
    job=root/'runs'/('city-stage-window-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    job.mkdir();qualification=job/'qualification';stop_file=job/'stop-requested'
    env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
        UAV_WINDOW_DEADLINE_UNIX=str(deadline),UAV_WINDOW_STOP_FILE=str(stop_file),
        UAV_GPU_FRACTION_CEILING='1.0',UAV_RESOURCE_LOG='1')
    python=str(root/'env/bin/python');checkpoint=run/'latest.pt';config=run/'config.json'
    scene=root/'scene.json';tasks=root/'city-tasks.json';survey=root/'data/visual-bootstrap/rgb-atlas.json'
    backbone=root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt'
    receipt=dict(schema='photo-goal-stage-window/v1',status='starting',pid=os.getpid(),job=str(job),
        source=str(source),run_dir=str(run),checkpoint_sha256=digest(checkpoint),
        started_utc=datetime.now(timezone.utc).isoformat(),absolute_deadline_unix=deadline,
        gpu_inventory=memory,compute_inventory=compute,other_jobs_preserved=True)
    write(job/'launch.json',receipt);write(root/'runs/active-city-integration.json',dict(job=str(job),pid=os.getpid()))
    children=[];streams=[]
    def interrupted(signum,frame):raise KeyboardInterrupt('Requested bounded stage shutdown')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def launch(argv,name):
        log=job/(name+'.log');reserve_write(log,128*2**20)
        stream=log.open('w');streams.append(stream)
        process=subprocess.Popen(argv,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(process);return process
    def stop_owned(process):
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    def monitor(process,service=None):
        while process.poll() is None:
            if service is not None and service.poll() is not None:raise RuntimeError('Owned Qwen service ended')
            try:guard().observe()
            except RuntimeError:
                with shutdown_writes():write(stop_file,dict(reason='capacity'))
                raise
            if time.time()>=deadline-300:
                with shutdown_writes():write(stop_file,dict(reason='deadline'))
            if time.time()>=deadline:raise RuntimeError('Absolute eight-hour deadline reached')
            time.sleep(5)
        if process.returncode:raise RuntimeError('Owned child failed: '+str(process.returncode))
    try:
        qwen=launch([python,'-u','-m','photo_goal','--root',str(root),'serve-city-qwen','--config',str(config),
            '--qwen-model',str(root/'assets/models/qwen2.5-vl-3b'),'--authfile',str(root/'campaign/qwen-auth.bin')],'qwen')
        until=min(time.time()+180,deadline-300)
        while time.time()<until:
            if qwen.poll() is not None:raise RuntimeError('Owned Qwen startup failed')
            if subprocess.check_output(['ss','-H','-ltn','sport','=',':48005'],text=True).strip():break
            time.sleep(1)
        else:raise RuntimeError('Owned Qwen readiness deadline exhausted')
        shared=['--root',str(root),'--checkpoint',str(checkpoint),'--config',str(config),'--scene',str(scene),
            '--taskset',str(tasks),'--survey',str(survey),'--backbone',str(backbone)]
        qual=launch([python,'-u',str(source/'scripts/lab_city_qualify.py'),*shared,
            '--run-dir',str(qualification),'--qualification',str(qualification/'receipt.json'),
            '--hours',str(max(.1,(deadline-time.time())/3600))],'qualification')
        receipt.update(status='qualifying',qualification_pid=qual.pid,qwen_pid=qwen.pid)
        write(job/'launch.json',receipt);monitor(qual,qwen)
        if not read(qualification/'receipt.json').get('passed'):raise RuntimeError('Native qualification did not pass')
        stop_owned(qwen) # Only the Qwen child created by this wrapper.
        if time.time()>=deadline-600:raise RuntimeError('Insufficient bounded time for training')
        operator=launch([python,'-u',str(source/'scripts/lab_city_overnight.py'),*shared,
            '--run-dir',str(run),'--qualification',str(qualification/'receipt.json'),
            '--hours',str((deadline-time.time())/3600),'--batches','2'],'operator')
        receipt.update(status='training',operator_pid=operator.pid,qualification_passed=True)
        write(job/'launch.json',receipt);monitor(operator)
        receipt['status']='finished'
    except BaseException as error:
        receipt.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        for process in reversed(children):stop_owned(process)
        receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
        with shutdown_writes():write(job/'launch.json',receipt)
        for stream in streams:stream.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
