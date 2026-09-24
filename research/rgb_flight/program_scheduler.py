"""Remote, receipt-driven dependency execution; runs independently of Windows.

Each invocation is an at-most-eight-hour resumable window. A completed worker
is never scientifically accepted without a separately supplied acceptance
receipt. Changed inputs/source/spec require a new round directory.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending'); temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def process_identity(pid):
    try:
        # comm may contain spaces; starttime is field 22 after its closing ).
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':
            return None
        return dict(pid=pid,start_ticks=stat[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):
        return None


def resolve(value, job, round_dir):
    return value.replace('{job}',str(job)).replace('{round}',str(round_dir)).replace('{source}',str(Path(__file__).parent))


def reconcile(stage, row, round_dir):
    log=Path(row['log'])
    if not row.get('job') and log.exists():
        for line in log.read_text().splitlines():
            if line.startswith('{"run":'):
                row['job']=json.loads(line)['run'];break
    job=Path(row['job']) if row.get('job') else None
    if job and (job/'result.json').exists():
        guard=json.loads((job/'result.json').read_text())
        row['guard_receipt']=dict(path=str(job/'result.json'),sha256=digest(job/'result.json'))
        if guard.get('status')=='stopped' and any(word in guard.get('reason','') for word in
            ('admission','owns measured','workers admitted','reservations','mutable artifact')):
            row.update(status='ready',reason='Waiting for resource admission: '+guard['reason']);return
        receipt=Path(resolve(stage['receipt'],job,round_dir))
        if not receipt.exists():
            row.update(status='failed',reason='Guard ended without worker receipt; inspect '+str(job/'stdout.log'));return
        outcome=json.loads(receipt.read_text())
        row['worker_receipt']=dict(path=str(receipt),sha256=digest(receipt))
        row['progress']=outcome
        if outcome.get('status')=='blocked':
            row.update(status='blocked',reason=outcome.get('reason','Missing supervision'));return
        artifacts=[]
        for item in stage.get('outputs',[]):
            path=Path(resolve(item,job,round_dir))
            if not path.is_file():
                if outcome.get('status')=='checkpointed':continue
                row.update(status='failed',reason='Missing output artifact: '+str(path));return
            artifacts.append(dict(path=str(path),sha256=digest(path)))
        row['artifacts']=artifacts
        if outcome.get('status')=='checkpointed':
            checkpoint=stage.get('checkpoint')
            if checkpoint:
                path=Path(resolve(checkpoint,job,round_dir))
                if not path.is_file():
                    row.update(status='failed',reason='Checkpoint receipt lacks resumable artifact');return
                row['checkpoint']=dict(path=str(path),sha256=digest(path))
            row.update(status='checkpointed',reason='Resume from verified checkpoint in a new guarded attempt');return
        if guard.get('return_code')!=0 or outcome.get('status') not in stage.get('completed_statuses',['completed']):
            detail=outcome.get('reason') or outcome.get('error') or 'Worker or resource guard failed'
            row.update(status='failed',reason=str(detail)+'; preserved '+str(receipt));return
        row.update(status='completed',reason='Worker completed; acceptance not inferred')
        acceptance=stage.get('acceptance')
        if acceptance:
            path=Path(resolve(acceptance,job,round_dir))
            if path.exists():
                evidence=json.loads(path.read_text())
                row['validation']=dict(path=str(path),sha256=digest(path),result=evidence)
                if evidence.get('accepted') is True:
                    if evidence.get('artifacts') != artifacts:
                        row.update(status='failed',reason='Acceptance receipt does not bind exact output hashes');return
                    row.update(status='accepted',reason='Artifact-bound acceptance receipt passed')
        return
    if process_identity(row['process']['pid'])!=row['process']:
        row.update(status='failed',reason='Remote process identity no longer exists and no guard receipt; inspect retained log')


def run(spec_path, round_dir, hours):
    spec=json.loads(spec_path.read_text())
    stages=spec['stages']; ids=[s['id'] for s in stages]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate stage ID')
    visited=set()
    for stage in stages:
        if any(d['stage'] not in visited or d['status'] not in ('completed','accepted') for d in stage.get('depends_on',[])):
            raise ValueError('Stages must be topologically ordered with explicit completed/accepted dependencies')
        visited.add(stage['id'])
    source=Path(__file__).parent
    hashes={p.name:digest(p) for p in sorted(source.iterdir()) if p.suffix in ('.py','.json','.toml','.lock')}
    contract=dict(spec_sha256=digest(spec_path),source_hashes=hashes,
        inputs={str(Path(p).resolve()):digest(p) for s in stages for p in s.get('inputs',[]) if '{' not in p})
    round_dir.mkdir(parents=True,exist_ok=True); state_path=round_dir/'state.json'
    import fcntl
    lock=(round_dir/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(schema='training-round/v1',
        contract=contract,round_id=round_dir.name,stages={s['id']:dict(status='ready',attempts=[]) for s in stages})
    if state['contract']!=contract:raise ValueError('Inputs, objective or source changed; create an explicit new round')
    def render(value):
        def substitute(match):
            row=state['stages'][match.group(1)]
            if not row.get('job') or row['status'] not in ('completed','accepted'):
                raise ValueError('Dependency has no completed job artifact: '+match.group(1))
            return row['job']
        # The guard creates the attempt directory. Keep its placeholder until
        # that directory exists rather than turning job-relative paths into /.
        return re.sub(r'\{stage:([A-Za-z0-9_-]+):job\}',substitute,resolve(value,'{job}',round_dir))
    deadline=time.monotonic()+hours*3600
    active_child=None
    while time.monotonic()<deadline and not (round_dir/'STOP').exists():
        running=False; launched=False; waiting=False
        for stage in stages:
            row=state['stages'][stage['id']]
            if row['status']=='running':
                reconcile(stage,row,round_dir)
                running |= row['status']=='running'
            if row['status']=='checkpointed' and stage.get('resume_command') and row.get('checkpoint'):
                if digest(row['checkpoint']['path'])!=row['checkpoint']['sha256']:
                    row.update(status='failed',reason='Checkpoint changed after verification')
                else:
                    row.update(status='ready',reason='Verified checkpoint queued for continuation')
            if row['status']=='ready' and not running:
                unmet=[d for d in stage.get('depends_on',[]) if
                    state['stages'][d['stage']]['status'] not in
                    (('completed','accepted') if d['status']=='completed' else ('accepted',))]
                if unmet:
                    failed=[d['stage'] for d in unmet if state['stages'][d['stage']]['status'] in ('failed','blocked')]
                    if failed:
                        row.update(status='failed',reason='Dependency failed: '+'; '.join(
                            name+': '+state['stages'][name].get('reason','Unknown failure') for name in failed))
                        continue
                    row['reason']='Waiting for '+', '.join(d['stage']+'='+d['status'] for d in unmet);continue
                # Avoid failed launches while a physical flight owns timing.
                active=[]
                for path in (Path.home()/'uav-rgb-flight/admission').glob('*.json'):
                    entry=json.loads(path.read_text())
                    identity=process_identity(entry['pid'])
                    if identity and identity['boot_id']==entry['boot_id']:active.append(entry)
                if active and (stage['kind'] in ('flight','exclusive') or any(x['kind'] in ('flight','exclusive') for x in active)
                               or (stage['kind']=='gpu' and sum(x['kind']=='gpu' for x in active)>=2)):
                    row['reason']='Waiting for active flight/exclusive resource lease';waiting=True;continue
                remaining=int(deadline-time.monotonic())
                if remaining<150:break
                template=stage['resume_command'] if row.get('checkpoint') else stage['command']
                command=[render(x).replace('{checkpoint}',row.get('checkpoint',{}).get('path','')) for x in template]
                if any('{job}' in p for p in stage.get('inputs',[])):
                    raise ValueError('Inputs must precede this attempt; use an explicit dependency artifact')
                stage_inputs={render(p):digest(render(p)) for p in stage.get('inputs',[])}
                if row.get('input_hashes') is not None and row['input_hashes']!=stage_inputs:
                    row.update(status='failed',reason='Stage input hashes changed; create a new training round');continue
                row['input_hashes']=stage_inputs
                log=round_dir/(stage['id']+f'-{len(row["attempts"]):03d}.log')
                guard=['python3',str(source/'spark_guard.py'),'--name',stage['id'],
                    '--seconds',str(min(remaining,int(stage.get('seconds',28800)))),
                    '--kind',stage['kind'],'--peak-gib',str(stage['peak_gib']),'--',*command]
                with log.open('x') as stream:
                    active_child=subprocess.Popen(guard,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(active_child.pid)
                if identity is None:raise RuntimeError('Guard exited during launch; inspect '+str(log))
                row.update(status='running',process=identity,log=str(log),command=guard,
                    started_unix=time.time(),reason='Owned remote guard running',job=None)
                row['attempts'].append(dict(process=identity,log=str(log)))
                running=launched=True
        state['updated_unix']=time.time();atomic(state_path,state)
        if not running and not launched and not waiting:break
        if active_child is not None:active_child.poll()
        time.sleep(1)
    for stage in stages:
        row=state['stages'][stage['id']]
        if row['status']=='running':
            reconcile(stage,row,round_dir)
            if row['status']=='running' and row.get('job'):
                (Path(row['job'])/'CHECKPOINT_REQUEST').touch()
    atomic(state_path,state)
    print(json.dumps({key:dict(status=row['status'],reason=row.get('reason')) for key,row in state['stages'].items()}))
    return 0 if all(row['status'] in ('completed','accepted') for row in state['stages'].values()) else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--round',type=Path,required=True);parser.add_argument('--hours',type=float,default=8)
    args=parser.parse_args()
    if not 0<args.hours<=8:parser.error('Window must be in (0,8] hours')
    raise SystemExit(run(args.spec,args.round,args.hours))
