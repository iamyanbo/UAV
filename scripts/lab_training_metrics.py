"""CPU-only metrics snapshots from actual flight receipts and accepted checkpoints.

Does not launch flights, run optimizers, or alter qualification/budget/checkpoints.
Earlier overwritten loss reports are unavailable, never reconstructed as measurements.
"""
import argparse
import collections
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time
import psutil
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/mnt/hdd2/yanbocheng/photo-goal-native'))
parser.add_argument('--run-dir',type=Path);parser.add_argument('--taskset',type=Path);parser.add_argument('--scene',type=Path)
parser.add_argument('--watch', action='store_true')
parser.add_argument('--hours', type=float, default=8)
parser.add_argument('--interval', type=float, default=30)
args = parser.parse_args()
if not 0 < args.hours <= 8 or args.interval < 15:
    raise ValueError('Metrics windows must be <=8 hours; interval must be >=15 seconds')
root = args.root.resolve()
run_root=(args.run_dir or root/'city-training').resolve()
out = run_root/'metrics'
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.pending')
    from photo_goal.mission_space import reserve_write
    text=json.dumps(value, allow_nan=False, indent=2)+'\n'
    reserve_write(path,len(text.encode())+4096)
    temporary.write_text(text,encoding='utf-8')
    temporary.replace(path)

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def distance(a, b):
    return math.dist(a, b)

cache = read(out/'episodes.json') if (out/'episodes.json').exists() else {}
taskset=args.taskset or root/'city-tasks.json'
task_data=read(taskset)
if task_data.get('schema')=='photo-goal-taskset/v2':
    os.environ['UAV_PROJECT_ROOT']=str(root)
    from photo_goal.mission_task_catalog import load_catalog
    from photo_goal.common import digest
    task_rows,_=load_catalog(taskset,digest(args.scene or root/'scene.json'))
else:task_rows=task_data['tasks']
tasks = {row['id']: row for row in task_rows}
last_checkpoint_stamp = None

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>UAV training progress</title>
<style>body{font:16px system-ui;margin:28px;background:#f6f8fa;color:#18212b;max-width:1200px}h1{margin-bottom:6px}p{line-height:1.5}.cards,.plots{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:18px}.card,section{background:white;border:1px solid #dce2e8;border-radius:10px;padding:18px}canvas{width:100%;height:240px}small{color:#53616e}table{border-collapse:collapse;width:100%}td,th{padding:7px;text-align:left;border-bottom:1px solid #eee}.warn{color:#8c3418}select{padding:6px}pre{white-space:pre-wrap;font-size:12px}</style></head>
<body><h1>UAV training progress</h1><p id="stamp"></p><p class="warn" id="state"></p>
<div class="cards" id="cards"></div><p>Flight curves: <select id="scope"><option value="mission">Regular missions</option><option value="support">Near-goal practice</option></select> <small>Trailing 25 completed training flights; these are training outcomes, not held-out evaluation.</small></p>
<div class="plots"><section><h3>Episode reward</h3><canvas id="reward"></canvas></section>
<section><h3>Successful / false stops (%)</h3><canvas id="success"></canvas></section>
<section><h3>Net movement and goal progress (m)</h3><canvas id="movement"></canvas></section>
<section><h3>Flight duration (seconds)</h3><canvas id="duration"></canvas></section>
<section><h3>PPO losses by accepted batch</h3><canvas id="ppo"></canvas></section>
<section><h3>PPO KL and critic explained variance</h3><canvas id="kl"></canvas></section>
<section><h3>World-model loss (last update in snapshot)</h3><canvas id="world"></canvas></section>
<section><h3>Visual-bootstrap loss history</h3><canvas id="bootstrap"></canvas></section></div>
<h2>Recorded flight outcomes by distance</h2><div id="bands"></div><h2>Latest optimizer report</h2><pre id="report"></pre>
<p>Metrics capture was added after three accepted PPO batches. Earlier overwritten PPO/world loss reports are missing.
World points show the latest update loss retained in each checkpoint, not the mean across every world update.
Qwen is frozen; it has no training-loss curve. Camera/command telemetry and guidance logs retain finer raw evidence.</p>
<p>This file is an offline snapshot. The lab observer refreshes its copy every 30 seconds while its job runs; reload a newly downloaded copy for later progress.</p>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
$('stamp').textContent='Snapshot: '+d.checked_utc;
$('state').textContent='Window status: '+d.window.status+(d.failure?' — '+d.failure:'');
$('cards').innerHTML=[['Accepted PPO transitions',d.counts.accepted_transitions],['World updates',d.counts.world_updates],['Completed training flights',d.episodes.length],['Accepted Qwen guidance responses',d.guidance.accepted]].map(([k,v])=>'<div class="card"><small>'+k+'</small><h2>'+v+'</h2></div>').join('');
function chart(id,series){const c=$(id),ctx=c.getContext('2d');c.width=540;c.height=240;ctx.clearRect(0,0,540,240);const points=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));if(!points.length){ctx.fillStyle='#53616e';ctx.fillText('No measured history available yet',42,120);return}let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax===xmin)xmax=xmin+1;if(ymax===ymin){ymin-=.1;ymax+=.1}const X=x=>48+(x-xmin)/(xmax-xmin)*474,Y=y=>196-(y-ymin)/(ymax-ymin)*153;ctx.strokeStyle='#dce2e8';ctx.strokeRect(48,43,474,153);ctx.font='12px system-ui';ctx.fillStyle='#53616e';ctx.fillText(ymax.toPrecision(3),2,49);ctx.fillText(ymin.toPrecision(3),2,197);ctx.fillText(xmin,48,218);ctx.fillText(xmax,488,218);series.forEach((s,i)=>{ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.fillText(s.name,48+i*160,23);ctx.beginPath();let begun=false;for(const p of s.points){if(!Number.isFinite(p[1]))continue;if(!begun){ctx.moveTo(X(p[0]),Y(p[1]));begun=true}else ctx.lineTo(X(p[0]),Y(p[1]));}ctx.stroke();for(const p of s.points)if(Number.isFinite(p[1])){ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),2.5,0,7);ctx.fill();}});c.title='Horizontal axis: completed flight number, or update/batch number for losses';}
const colors=['#1765aa','#d05c27','#238047'];
function plotFlights(){let rows=d.episodes.filter(r=>r.support===($('scope').value==='support'));function roll(fn){return rows.map((r,i)=>{const a=rows.slice(Math.max(0,i-24),i+1);return [i+1,a.reduce((s,v)=>s+fn(v),0)/a.length]})}chart('reward',[{name:'Mean reward',color:colors[0],points:roll(r=>r.reward)}]);chart('success',[{name:'Successful stop',color:colors[2],points:roll(r=>100*(r.event==='success'))},{name:'False stop',color:colors[1],points:roll(r=>100*(r.event==='false_stop'))}]);chart('movement',[{name:'Net movement',color:colors[0],points:roll(r=>r.net_movement_m)},{name:'Goal progress',color:colors[2],points:roll(r=>r.goal_progress_m)}]);chart('duration',[{name:'Mean duration',color:colors[0],points:roll(r=>r.elapsed_s)}]);}
$('scope').onchange=plotFlights;plotFlights();
const snapshots=d.optimizers, pp=key=>snapshots.filter(r=>r.report.ppo).map(r=>[r.counts.city_batches,r.report.ppo[key]]);
chart('ppo',['policy_loss','value_loss','stop_loss'].map((k,i)=>({name:k,color:colors[i],points:pp(k)})));
chart('kl',['final_rollout_kl','explained_variance'].map((k,i)=>({name:k,color:colors[i],points:pp(k)})));
chart('world',[{name:'Latest loss',color:colors[0],points:snapshots.filter(r=>r.report.world_last).map(r=>[r.counts.world_updates,r.report.world_last.loss])}]);
chart('bootstrap',['loss','development_loss'].map((k,i)=>({name:k,color:colors[i],points:d.bootstrap.map(r=>[r.update,r[k]])})));
$('bands').innerHTML='<table><tr><th>Group</th><th>Flights</th><th>Success</th><th>False stop</th><th>Mean reward</th><th>Mean goal progress</th></tr>'+d.groups.map(r=>'<tr><td>'+r.group+'</td><td>'+r.flights+'</td><td>'+r.success+'</td><td>'+r.false_stop+'</td><td>'+r.mean_reward.toFixed(3)+'</td><td>'+r.mean_goal_progress_m.toFixed(2)+' m</td></tr>').join('')+'</table>';
$('report').textContent=JSON.stringify(d.latest_optimizer,null,2);
</script></body></html>'''

def snapshot():
    global last_checkpoint_stamp
    for path in run_root.glob('city-*.json'):
        if path.stem in cache:
            continue
        flight = read(path)
        if flight.get('controller') != 'learner' or flight.get('infrastructure_cut'):
            continue
        telemetry = Path(flight['telemetry'])
        observations = []
        with telemetry.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('kind') == 'observation':
                    observations.append(row['state']['position'])
        if not observations:
            continue
        goal = tasks[flight['task_id']]['goal']
        initial = distance(observations[0], goal)
        final = distance(observations[-1], goal)
        cache[path.stem] = dict(attempt=flight['attempt'],task_id=flight['task_id'],
            support=flight['support'],event=flight['event'],reward=flight['reward'],
            task_class=flight.get('task_class','support' if flight['support'] else 'regular'),
            elapsed_s=flight['elapsed_s'],ended_timestamp=path.stat().st_mtime,
            distance_m=distance(flight['start'],goal),initial_distance_m=initial,final_distance_m=final,
            goal_progress_m=initial-final,net_movement_m=distance(observations[0],observations[-1]),
            path_length_m=sum(distance(a,b) for a,b in zip(observations,observations[1:])),
            policy_bundles=flight['policy_bundles'],telemetry=str(telemetry))
    atomic(out/'episodes.json', cache)
    pointer = run_root/'latest.json'
    current = read(pointer)
    checkpoint = run_root/'latest.pt'
    stamp = (checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
    if stamp != last_checkpoint_stamp and psutil.virtual_memory().available >= 14*2**30:
        with checkpoint.open('rb') as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        report = clean(saved.get('last_report', {}))
        counts = saved['counts']
        record = dict(checked_utc=datetime.now(timezone.utc).isoformat(),counts=counts,report=report,
                      scope='Latest accepted checkpoint report; earlier overwritten history unavailable')
        if report:
            atomic(out/f"optimizer-batch-{counts['city_batches']:04d}.json",record)
        del saved
        gc.collect()
        last_checkpoint_stamp = stamp
    records = [read(p) for p in sorted(out.glob('optimizer-batch-*.json'))]
    episodes = sorted(cache.values(), key=lambda row: row['ended_timestamp'])
    groups = collections.defaultdict(list)
    for row in episodes:
        band = 'Near-goal practice' if row['support'] else ('50–100 m' if row['distance_m']<=100 else '100–200 m' if row['distance_m']<=200 else '200–300 m')
        if row['task_class']=='intermediate':band='10-50 m'
        groups[band].append(row)
    grouped = [dict(group=name,flights=len(rows),success=sum(r['event']=='success' for r in rows),
                    false_stop=sum(r['event']=='false_stop' for r in rows),
                    mean_reward=sum(r['reward'] for r in rows)/len(rows),
                    mean_goal_progress_m=sum(r['goal_progress_m'] for r in rows)/len(rows),
                    mean_net_movement_m=sum(r['net_movement_m'] for r in rows)/len(rows)) for name,rows in groups.items()]
    def lines(path):
        if not path.exists():return []
        result=[]
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:continue # A concurrently appended final line may be incomplete.
        return result
    bootstrap = lines(root/'data/visual-bootstrap/loss.jsonl')
    guidance = lines(run_root/'guidance.jsonl')
    plan = read(Path(read(root/'runs/active-city-window.json')['job'])/'launch.json')
    status_path = run_root/'status.json'
    status = read(status_path) if status_path.exists() else {}
    result = clean(dict(checked_utc=datetime.now(timezone.utc).isoformat(),window=plan,
        failure=status.get('error') if plan['status']=='failed' else None,counts=current['counts'],
        pending=current.get('pending'),episodes=episodes,groups=grouped,bootstrap=bootstrap,
        optimizers=records,latest_optimizer=records[-1] if records else None,
        guidance=dict(accepted=sum(bool(r.get('selected')) for r in guidance),
                      discarded=sum(r.get('discarded')=='invalid_or_obsolete_guidance' for r in guidance)),
        limits=['Missing earlier overwritten PPO/world loss history',
                'World loss points are latest-update snapshots, not averages',
                'Training outcomes only; no held-out evaluation', 'Qwen frozen; no adaptation loss']))
    atomic(out/'summary.json', result)
    payload=json.dumps(result,allow_nan=False).replace('<','\\u003c')
    html=out/'dashboard.html'
    pending=out/'dashboard.html.pending'
    text=HTML.replace('__DATA__',payload)
    from photo_goal.mission_space import reserve_write
    reserve_write(html,len(text.encode())+4096)
    pending.write_text(text,encoding='utf-8')
    pending.replace(html)
    print(json.dumps(dict(counts=current['counts'],groups=grouped,latest_optimizer=result['latest_optimizer'],
                         failure=result['failure'],dashboard=str(html))),flush=True)

deadline = time.monotonic()+args.hours*3600
owner = read(root/'runs/active-city-window.json')['pid']
while True:
    snapshot()
    if not args.watch or time.monotonic() >= deadline or not psutil.pid_exists(owner):
        break
    time.sleep(args.interval)
