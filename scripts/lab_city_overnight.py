"""One explicit, HDD-only eight-hour city window; no automatic restart loop."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
from photo_goal.common import read, write, digest
from photo_goal.mission_storage import configure
from photo_goal.mission_space import guard, reserve_write, shutdown_writes
from photo_goal.mission_contracts import city_config, identity
from photo_goal.native_full_training import checked_assets, implementation_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/hdd2/yanbocheng/photo-goal-native'))
    for name in ('config','scene','taskset','qualification','survey','run-dir','checkpoint','backbone','ffmpeg'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--batches',type=int,default=2)
    parser.add_argument('--replace-service-pid',type=int,default=0)
    args=parser.parse_args()
    if not 0<args.hours<=8 or args.batches<=0:parser.error('Use <=8 hours and positive batch count')
    started=time.time();deadline=started+args.hours*3600
    root=args.root.resolve();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    storage=configure(root);(root/'runs').mkdir(exist_ok=True)
    lock=(root/'runs/operator.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another campaign operator owns this project')
    cfg=city_config(args.config)
    scene=args.scene or root/'scene.json';tasks=args.taskset or root/'city-tasks.json'
    qualification=args.qualification or root/'city-qualification.json'
    survey=args.survey or root/'data/visual-bootstrap/rgb-atlas.json'
    run=args.run_dir or root/'city-training'
    checkpoint=args.checkpoint or run/'latest.pt'
    backbone=args.backbone or root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt'
    if cfg.get('phase_id') and any(x is None for x in (args.config,args.run_dir,args.checkpoint,args.qualification)):
        parser.error('Repair windows require explicit config/run-dir/checkpoint/qualification')
    if not checkpoint.is_file():raise FileNotFoundError('No automatic fallback checkpoint: '+str(checkpoint))
    for path in (scene,tasks,qualification,survey,run,checkpoint,backbone):
        if not path.resolve().is_relative_to(root):raise ValueError('Window path escapes HDD root: '+str(path))
    if args.replace_service_pid:
        command=Path('/proc')/str(args.replace_service_pid)/'cmdline'
        if command.exists():
            value=command.read_bytes()
            if b'serve-city-qwen' not in value or str(root).encode() not in value:
                raise RuntimeError('Replacement PID is not this project Qwen service')
            os.kill(args.replace_service_pid,signal.SIGTERM);until=time.monotonic()+20
            while command.exists() and time.monotonic()<until:time.sleep(.5)
            if command.exists():raise RuntimeError('Previous Qwen service did not end')
    if (root/'active-job.lock').exists():raise RuntimeError('An existing trainer/qualification lock must be reconciled')
    guard().reconcile_quiescent()
    checked_assets(root,survey,cfg,scene,tasks,qualification)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job=root/'runs'/('city-window-'+stamp);job.mkdir()
    snapshot=job/'source';snapshot.mkdir()
    import photo_goal
    package=Path(photo_goal.__file__).parent
    reserve_write(snapshot,sum(p.stat().st_size for p in package.rglob('*') if p.is_file())+32*2**20)
    shutil.copytree(package,snapshot/'photo_goal',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('lab_city_overnight.py','lab_training_metrics.py','lab_city_status.py','lab_city_qualify.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    site=root/'code/sitecustomize.py'
    if site.exists():shutil.copyfile(site,snapshot/'sitecustomize.py')
    env=dict(os.environ,PYTHONPATH=str(snapshot),CUDA_VISIBLE_DEVICES='0',
        UAV_WINDOW_STOP_FILE=str(job/'stop-requested'),UAV_WINDOW_DEADLINE_UNIX=str(deadline),
        UAV_GPU_FRACTION_CEILING='1.0',UAV_RESOURCE_LOG='1',PYTHONDONTWRITEBYTECODE='1')
    if args.ffmpeg:env['UAV_FFMPEG']=str(args.ffmpeg.resolve(strict=True))
    python=str(root/'env/bin/python')
    plan=dict(schema='photo-goal-lab-window/v2',status='starting',pid=os.getpid(),
        started_at_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),
        absolute_deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        maximum_hours=args.hours,project_root=str(root),gpu=0,folder_ceiling_gib=256,
        admission_ceiling_gib=254,phase_id=cfg.get('phase_id'),config_sha256=identity(cfg),
        storage=storage,qualification_sha256=digest(qualification),implementation_sha256=implementation_identity(),
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),source=str(snapshot),run_dir=str(run),
        taskset=str(tasks),scene=str(scene),survey=str(survey),exit_reason=None,
        metrics_dashboard=str(run/'metrics/dashboard.html'),
        source_manifest={str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob('*') if p.is_file()})
    write(job/'launch.json',plan)
    write(root/'runs/active-city-window.json',dict(job=str(job),pid=os.getpid(),run_dir=str(run)))
    children=[];logs=[];trainer=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Operator shutdown requested')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def launch(argv,name,cpu=False):
        path=job/(name+'.log');reserve_write(path,128*2**20)
        stream=path.open('w');logs.append(stream)
        child_env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1') if cpu else env
        process=subprocess.Popen(argv,cwd=root,env=child_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(process);return process
    try:
        qwen=launch([python,'-u','-m','photo_goal','--root',str(root),'serve-city-qwen',
            '--qwen-model',str(root/'assets/models/qwen2.5-vl-3b'),'--authfile',str(root/'campaign/qwen-auth.bin')],'qwen')
        until=min(time.time()+180,deadline-300)
        while time.time()<until:
            if qwen.poll() is not None:raise RuntimeError('Frozen Qwen startup failed')
            if subprocess.check_output(['ss','-H','-ltn','sport','=',':48005'],text=True).strip():break
            time.sleep(1)
        else:raise RuntimeError('Qwen readiness or window deadline exhausted')
        argv=[python,'-u','-m','photo_goal','--root',str(root),'train-city','--checkpoint',str(checkpoint),
            '--backbone',str(backbone),'--survey',str(survey),'--qwen-auth',str(root/'campaign/qwen-auth.bin'),
            '--scene',str(scene),'--taskset',str(tasks),'--qualification',str(qualification),'--run-dir',str(run),
            '--hours',str(args.hours),'--batches',str(args.batches)]
        if args.config:argv.extend(['--config',str(args.config)])
        trainer=launch(argv,'training')
        metrics=launch([python,'-u',str(snapshot/'lab_training_metrics.py'),'--root',str(root),
            '--run-dir',str(run),'--taskset',str(tasks),'--scene',str(scene),'--watch','--hours',str(args.hours)],'metrics',True)
        plan.update(status='running',trainer_pid=trainer.pid,qwen_pid=qwen.pid,metrics_pid=metrics.pid)
        write(job/'launch.json',plan);print(json.dumps(plan),flush=True)
        while trainer.poll() is None:
            if qwen.poll() is not None:plan['exit_reason']='guidance_failure';raise RuntimeError('Frozen Qwen ended during training')
            try:bound=guard().observe()
            except RuntimeError as error:
                plan.update(exit_reason='resource_limit',resource_error=str(error));bound=guard().bound()
                with shutdown_writes():write(job/'stop-requested',dict(reason=plan['exit_reason']))
            if time.time()>=deadline-300:
                plan['exit_reason']=plan['exit_reason'] or 'deadline'
                with shutdown_writes():write(job/'stop-requested',dict(reason=plan['exit_reason']))
            if time.time()>=deadline:break
            print(json.dumps(dict(root_upper_bound_bytes=bound,deadline_unix=deadline)),flush=True)
            time.sleep(10)
        plan.update(trainer_exit=trainer.poll(),status='finished' if trainer.poll()==0 else 'stopping',
            exit_reason=plan['exit_reason'] or ('batch_budget' if trainer.poll()==0 else 'trainer_failure'))
    except BaseException as error:
        plan.update(status='failed',error=type(error).__name__+': '+str(error),
            exit_reason=plan['exit_reason'] or ('operator_interrupt' if isinstance(error,KeyboardInterrupt) else 'operator_failure'))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT if child is trainer else signal.SIGTERM)
                try:child.wait(timeout=min(120,max(5,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        if trainer:plan['trainer_exit']=trainer.returncode
        plan['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        if plan['status']=='stopping':
            plan['status']='finished' if trainer and trainer.returncode==0 else ('failed' if plan['exit_reason']=='trainer_failure' else 'stopped')
        with shutdown_writes():write(job/'launch.json',plan)
        for stream in logs:stream.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
