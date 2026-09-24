"""Bounded native-Linux job guard for Spark's shared CPU/GPU memory pool."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from resource_admission import ResourceLease


def memory():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
    return dict(total_bytes=values['MemTotal'],available_bytes=values['MemAvailable'],
                used_fraction=1-values['MemAvailable']/values['MemTotal'],swap_used_bytes=values['SwapTotal']-values['SwapFree'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--seconds',type=int,default=1800)
    parser.add_argument('--kind',choices=['exclusive','flight','gpu','cpu'],default='exclusive')
    parser.add_argument('--peak-gib',type=float,default=64)
    parser.add_argument('--artifact',help='Exclusive mutable artifact key')
    parser.add_argument('--source-directory',type=Path,help='Explicit original study source snapshot for exact resumption')
    parser.add_argument('--resume-dependencies',type=Path,help='Original guarded dependency manifest to match before resumption')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not 0 < args.peak_gib < 112:
        parser.error('Expected a positive, bounded peak-memory reservation')
    if not 0<args.seconds<=8*3600 or not args.name.replace('-','').replace('_','').isalnum():
        parser.error('Invalid bounded job')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    root=Path.home()/'uav-rgb-flight'
    if not command:
        parser.error('A command is required')
    run=root/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+args.name+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    snapshot=run/'source'
    snapshot.mkdir()
    hashes={}
    submission=(args.source_directory or Path(__file__).resolve().parent).resolve()
    if not submission.is_relative_to(root.resolve()) or not submission.is_dir():
        raise ValueError('Source snapshot must remain inside the study workspace')
    for source in submission.rglob('*'):
        if source.is_file() and source.suffix in ('.py','.json','.txt','.toml','.lock') and '__pycache__' not in source.parts:
            data=source.read_bytes()
            relative=source.relative_to(submission)
            (snapshot/relative).parent.mkdir(parents=True,exist_ok=True)
            (snapshot/relative).write_bytes(data)
            hashes[str(relative)]=hashlib.sha256(data).hexdigest()
    command=[arg.replace('{job}',str(run)) for arg in command]
    command=[str(snapshot/Path(arg).name) if Path(arg).suffix=='.py' and Path(arg).resolve().parent==submission else arg for arg in command]
    (run/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    # Bind external model/source dependencies to this attempt before executing.
    # Only explicit provenance receipts are copied; credentials are never read.
    dependency_hashes={}
    dependencies=run/'dependencies'
    for path in [*(root/'receipts').glob('model-*.json'),
                 root/'receipts/splat-native-port.json',root/'receipts/splat-native.patch',
                 *(root/'assets/models/receipts').glob('*.json')]:
        if path.is_file():
            relative=path.relative_to(root)
            target=dependencies/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            data=path.read_bytes()
            target.write_bytes(data)
            dependency_hashes[str(relative)]=hashlib.sha256(data).hexdigest()
    (run/'dependency_hashes.json').write_text(json.dumps(dependency_hashes,indent=2))
    (run/'guard_bootstrap_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.resume_dependencies:
        previous=args.resume_dependencies.resolve()
        if not previous.is_relative_to(root.resolve()) or json.loads(previous.read_text())!=dependency_hashes:
            raise ValueError('Asset/dependency receipts changed; exact survey resumption is not valid')
    receipt=dict(command=command,seconds=args.seconds,resource_policy='12 GiB available floor; 4 GiB checkpoint headroom; measured peak admission; 80 GiB disk reserve',kind=args.kind,peak_gib=args.peak_gib,status='running')
    (run/'request.json').write_text(json.dumps(receipt,indent=2))
    # Publish identity even when admission fails before a worker can start.
    print(json.dumps({'run':str(run),'guard_pid':os.getpid()}),flush=True)
    child=None
    lease=ResourceLease(root,run,args.kind,args.peak_gib,args.artifact)
    log=None
    checkpoint_requested=None
    exit_code=2
    started=time.monotonic()
    def check():
        nonlocal checkpoint_requested
        sample=dict(wall_time=time.time(),**memory(),free_disk_bytes=shutil.disk_usage(root).free)
        with (run/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(sample)+'\n')
        if sample['available_bytes']<12*1024**3:
            raise RuntimeError('Unified-memory reserve breached')
        if sample['free_disk_bytes']<80*1024**3:
            raise RuntimeError('Disk reserve breached')
        if (root/'STOP').exists() or (run/'STOP').exists():
            raise RuntimeError('Operator stop marker')
        if time.monotonic()-started>=args.seconds:
            raise RuntimeError('Job deadline reached')
        if child is not None and checkpoint_requested is None and (sample['available_bytes']<16*1024**3 or time.monotonic()-started>=args.seconds-min(120,args.seconds*.1)):
            checkpoint_requested=time.monotonic()
            (run/'CHECKPOINT_REQUEST').write_text('Stop after atomic checkpoint: resource headroom or deadline')
            receipt['checkpoint_requested']=True
        if checkpoint_requested is not None and time.monotonic()-checkpoint_requested>=120:
            raise RuntimeError('Checkpoint grace period expired')
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,interrupt)
    try:
        log=(run/'stdout.log').open('w')
        check()
        lease.acquire(memory()['available_bytes'])
        environment=dict(os.environ,RGB_JOB_DIR=str(run),RGB_CHECKPOINT_REQUEST=str(run/'CHECKPOINT_REQUEST'))
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
        (run/'process.json').write_text(json.dumps(dict(pid=child.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
        print(json.dumps({'run':str(run),'pid':child.pid}),flush=True)
        while child.poll() is None:
            check()
            time.sleep(1)
        receipt.update(status='finished',return_code=child.returncode)
        exit_code=0 if child.returncode==0 else 2
    except (RuntimeError,KeyboardInterrupt,BlockingIOError) as error:
        receipt.update(status='stopped',reason=str(error))
        exit_code=2
    finally:
        if child is not None:
            # Reap owned descendants even if the direct child already exited.
            try:
                os.killpg(child.pid,signal.SIGTERM)
                grace=time.monotonic()+10
                while time.monotonic()<grace:
                    child.poll()
                    os.killpg(child.pid,0)
                    time.sleep(.1)
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            # Docker workloads are daemon-owned, outside the client's process
            # group. Stop only containers explicitly labeled with THIS job.
            if shutil.which('docker'):
                try:
                    containers=subprocess.check_output(['docker','ps','-q','--filter','label=rgb-flight.job='+str(run)],text=True,timeout=10).split()
                    for container in containers:
                        subprocess.run(['docker','stop','-t','5',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=True)
                except (subprocess.SubprocessError,OSError) as error:
                    receipt['container_cleanup_error']=type(error).__name__
                    receipt['status']='cleanup_failed'
                    exit_code=2
        receipt['elapsed_seconds']=time.monotonic()-started
        (run/'result.json').write_text(json.dumps(receipt,indent=2))
        if log is not None:
            log.close()
        lease.release()
        print(json.dumps(receipt),flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
