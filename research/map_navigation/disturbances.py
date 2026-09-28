"""Fixed-command disturbance interventions; no claim of moving-state cloning."""
from bisect import bisect_right
from pathlib import Path
import math
from .common import read,write,digest
from .collection_protocol import MISSIONS,LABELS


class CommandReplay:
    def __init__(self,path):
        self.spec=read(path)
        if self.spec.get('schema')!='photo-map-command-intervention/v1':raise ValueError('Invalid command intervention')
        self.rows=self.spec['commands'];self.times=[r['seconds'] for r in self.rows]
        if not self.times or self.times[0]!=0 or any(b<=a for a,b in zip(self.times,self.times[1:])):raise ValueError('Unordered intervention')
        for row in self.rows:
            values=row['values']
            if len(values)!=4 or not all(math.isfinite(x) for x in values) or math.hypot(*values[:2])>3 or abs(values[2])>1 or abs(values[3])>45:
                raise ValueError('Intervention command outside campaign limits')
        if not self.times[-1]<=self.spec['duration_s']:raise ValueError('Invalid intervention duration')

    def at(self,seconds):
        if seconds>=self.spec['duration_s']:return None
        return list(self.rows[max(0,bisect_right(self.times,seconds)-1)]['values'])


def matched_dispatch_prefix(arms):
    """Compare piecewise held commands, including refresh gaps and watchdogs.

    Similar transition times alone cannot establish a matching constant-command
    interval. Stop at the first unsupported interval, including a 250 ms gap.
    """
    histories=[];ends=[]
    for frames,commands in arms:
        start=frames[0]['state_sim_ns'];end=(frames[-1]['state_sim_ns']-start)/1e9
        transitions=[];previous=None;last=None
        preceding=[row for row in commands if row.get('dispatch_sim_ns',row.get('sim_ns'))<start]
        if preceding:
            before=preceding[-1]
            if start-before.get('dispatch_sim_ns',before.get('sim_ns'))<=250_000_000:
                commands=[dict(before,dispatch_sim_ns=start),*[r for r in commands if r.get('dispatch_sim_ns',r.get('sim_ns'))>=start]]
        for row in commands:
            t=(row.get('dispatch_sim_ns',row.get('sim_ns'))-start)/1e9
            if t<0:continue
            if t>end:break
            if (last is None and t>.0125) or (last is not None and t-last>.25):
                end=min(end,(last+.25) if last is not None else 0.);break
            if row.get('watchdog_override'):
                end=min(end,t);break
            if previous!=row['values']:
                transitions.append((t,row['values']));previous=row['values']
            last=t
        end=min(end,last+.25 if last is not None else 0.)
        histories.append(transitions);ends.append(end)
    prefix=min(ends)
    a,b=histories
    for i in range(max(len(a),len(b))):
        if i>=min(len(a),len(b)):
            extra=a[i:] or b[i:]
            return max(0.,min(prefix,extra[0][0]))
        x,y=a[i],b[i]
        if x[0]>=prefix and y[0]>=prefix:break
        if abs(x[0]-y[0])>.0125 or x[1]!=y[1]:
            return max(0.,min(prefix,x[0],y[0]))
    return max(0.,prefix)


def prepare(registry,flights,output):
    from dataclasses import replace
    from .annotations import lines
    from .data import yaw
    from goal_io import load_goal,write_goal
    scenes={r['scene_id']:r for r in read(registry)['scenes']};out=Path(output)
    public=[];private=[];excluded=[]
    for result_path in sorted(Path(flights).rglob('episode/result.json')):
        result=read(result_path);episode=result_path.parent
        if result.get('split') not in ('train','validation'):continue
        scene=scenes[result['scene_id']];qualification=read(scene['qualification'])
        wind=qualification.get('disturbance_capabilities',{}).get('wind',{})
        if not wind.get('measured') or not wind.get('supported_ned_mps'):
            excluded.append(dict(episode=str(episode),reason='wind capability unqualified'));continue
        labels=lines(episode/'training_labels/frames.jsonl');commands=lines(episode/'training_labels/commands.jsonl')
        if not labels or not commands:continue
        first=labels[0]
        if sum(x*x for x in first['true_velocity_ned_mps'])>.01 or sum(x*x for x in first['true_angular_velocity_radps'])>.0025:continue
        start=first['state_sim_ns'];end=labels[-1]['state_sim_ns']
        selected=[r for r in commands if start<=r.get('dispatch_sim_ns',r.get('sim_ns',-1))<end]
        if not selected:continue
        rows=[dict(seconds=0.,values=[0.]*4)]
        for row in selected:
            t=(row.get('dispatch_sim_ns',row['sim_ns'])-start)/1e9
            if t>rows[-1]['seconds']:rows.append(dict(seconds=t,values=row['values']))
        group=digest(episode/'training_labels/commands.jsonl')[:16]
        script=out/'commands'/(group+'.json')
        write(script,dict(schema='photo-map-command-intervention/v1',commands=rows,duration_s=(end-start)/1e9,
            source_sha256=digest(episode/'training_labels/commands.jsonl'),start_state=first))
        CommandReplay(script)
        task=read(episode/'evaluator_labels/episode.json')
        for suffix,value in [('calm',[0.,0.,0.]),('wind',wind['supported_ned_mps'])]:
            ident=group+'-'+suffix;goal_path=out/'goals'/ident
            write_goal(goal_path,replace(load_goal(episode/'goal'),episode_id=ident))
            public.append(dict(episode_id=ident,scene_id=scene['scene_id'],split=scene['split'],map_sha256=scene['map_sha256'],
                goal_views=1,goal_path=str(goal_path.resolve()),goal_sha256=digest(goal_path/'goal.json'),timeout_s=(end-start)/1e9+10,
                collection_source='manoeuvre',collection_stream='disturbance_branch',branch_group=group,
                branch_protocol='settled-fixed-commands/v1',command_replay=str(script.resolve()),command_replay_sha256=digest(script),
                conditions=dict(wind_ned_mps=value,lighting='nominal',weather='nominal',sensor='nominal',latency='native')))
            private.append(dict(task,episode_id=ident,start_ned_m=first['true_position_ned_m'],
                start_yaw_degrees=math.degrees(yaw(first['true_quaternion_xyzw'])),timeout_s=(end-start)/1e9+10))
    write(out/'excluded.json',excluded)
    for split in ('train','validation'):
        write(out/(split+'.json'),dict(schema=MISSIONS,registry_sha256=digest(registry),episodes=[r for r in public if r['split']==split]))
        write(out/'evaluator_labels'/(split+'.json'),dict(schema=LABELS,episodes=[r for r in private if r['split']==split]))
    if not public:raise ValueError('No qualified wind intervention states; see excluded.json')


def qualify_pairs(flights,output):
    from collections import defaultdict
    import numpy as np
    from .annotations import lines
    groups=defaultdict(list);reports=[]
    for path in Path(flights).rglob('episode/result.json'):
        result=read(path)
        if result.get('collection_stream')=='disturbance_branch':groups[result['branch_group']].append((path.parent,result))
    for group,arms in groups.items():
        reasons=[];prefix=0.
        if len(arms)!=2:reasons.append('two completed arms required')
        else:
            (a,ra),(b,rb)=arms
            fa=lines(a/'training_labels/frames.jsonl');fb=lines(b/'training_labels/frames.jsonl')
            ca=lines(a/'training_labels/commands.jsonl');cb=lines(b/'training_labels/commands.jsonl')
            if not fa or not fb or not ca or not cb:reasons.append('missing state or dispatch records')
            else:
                if not ra.get('command_replay_sha256') or ra['command_replay_sha256']!=rb.get('command_replay_sha256'):reasons.append('missing or different scheduled commands')
                wa,wb=ra.get('wind_ned_mps'),rb.get('wind_ned_mps')
                if wa is None or wb is None or wa==wb or [0.,0.,0.] not in (wa,wb):reasons.append('calm and wind arms required')
                for key,tolerance in [('true_position_ned_m',.05),('true_velocity_ned_mps',.02),('true_angular_velocity_radps',.02)]:
                    if np.linalg.norm(np.asarray(fa[0][key])-fb[0][key])>tolerance:reasons.append('initial '+key)
                if abs(float(np.dot(fa[0]['true_quaternion_xyzw'],fb[0]['true_quaternion_xyzw'])))<.9999:reasons.append('initial attitude')
                prefix=matched_dispatch_prefix([(fa,ca),(fb,cb)])
                if prefix<4:reasons.append('less than four seconds of matched dispatched commands')
        reports.append(dict(group=group,qualified=not reasons,matched_prefix_s=prefix,reasons=reasons,
                            comparison_scope='observed settled state and dispatched commands; not hidden simulator state'))
    write(output,dict(schema='photo-map-intervention-audit/v1',pairs=reports,qualified_pairs=sum(r['qualified'] for r in reports)))
