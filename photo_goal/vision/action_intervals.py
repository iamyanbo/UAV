"""Executed 200 ms action intervals and identical Mode 2 control-slot layout."""
import numpy as np

SLOT_NS=50_000_000


def application_slots(commands, start_ns, constant_epochs=()):
    """Duration-weighted commands in four slots; missing application time masks.

    Submission timestamps are not application timestamps. Old command files
    remain useful for perception but cannot claim exact world action targets.
    """
    for epoch in constant_epochs:
        if epoch.get('schema')!='confirmed-constant-command/v1':
            raise ValueError('Unknown constant-command proof')
        if epoch['begin_sim_ns']<=start_ns and start_ns+4*SLOT_NS<=epoch['end_sim_ns']:
            relevant=[r for r in commands if epoch['begin_sim_ns']<=r['sim_ns']<epoch['end_sim_ns']]
            if not relevant or any(r['values']!=epoch['values'] for r in relevant):
                raise ValueError('Constant-command proof contradicts command log')
            return dict(values=[list(epoch['values']) for _ in range(4)],valid=[True]*4,
                        reason='confirmed_constant_command_epoch')
    if not commands or any(r.get('application_sim_ns') is None for r in commands):
        return dict(values=None,valid=[False]*4,reason='command_application_times_unobserved')
    events=sorted(commands,key=lambda r:r['application_sim_ns'])
    values=[];valid=[]
    for slot in range(4):
        begin=start_ns+slot*SLOT_NS;end=begin+SLOT_NS
        past=[r for r in events if r['application_sim_ns']<=begin]
        if not past:
            values.append([0.]*4);valid.append(False);continue
        cursor=begin;current=np.asarray(past[-1]['values'],dtype=float);total=np.zeros(4)
        for row in events:
            stamp=row['application_sim_ns']
            if begin<stamp<end:
                total+=(stamp-cursor)*current;cursor=stamp;current=np.asarray(row['values'],dtype=float)
        total+=(end-cursor)*current
        values.append((total/SLOT_NS).tolist());valid.append(True)
    return dict(values=values,valid=valid,reason=None)


def proposed_slots(action):
    """Differentiable expansion of each constant Mode 2 action into four slots."""
    if action.shape[-1]!=4:raise ValueError('Expected four body command channels')
    return action.unsqueeze(-2).expand(*action.shape[:-1],4,4)


ACTION_SEMANTICS='post-safety-dispatch/50ms-v3'


def executed_slots(commands,start_ns,constant_epochs=()):
    """Commands dispatched after safety; application latency is learned dynamics.

    Legacy sim_ns is a simulator clock sampled immediately before dispatch.
    Its uncertainty is retained, never called an application timestamp.
    Unknown coverage, nonmonotonic dispatches and >250ms gaps are masked.
    """
    events=[]
    for row in commands:
        stamp=row.get('dispatch_sim_ns',row.get('sim_ns'))
        if stamp is None or len(row.get('values',[]))!=4:continue
        if not np.isfinite(row['values']).all():continue
        if row.get('submitted_monotonic',row.get('issued_monotonic')) is None:continue
        events.append(dict(row,dispatch_sim_ns=stamp))
    if any(a['dispatch_sim_ns']>=b['dispatch_sim_ns'] for a,b in zip(events,events[1:])):
        return dict(values=None,valid=[False]*4,reason='nonmonotonic_dispatch_clock')
    values=[];valid=[]
    for slot in range(4):
        begin=start_ns+slot*SLOT_NS;end=begin+SLOT_NS
        past=[r for r in events if r['dispatch_sim_ns']<=begin]
        if not past:
            values.append([0.]*4);valid.append(False);continue
        current=past[-1];cursor=begin;total=np.zeros(4);qualified=True
        for row in [r for r in events if begin<r['dispatch_sim_ns']<end]:
            qualified &= row['dispatch_sim_ns']-current['dispatch_sim_ns']<=250_000_000
            total+=(row['dispatch_sim_ns']-cursor)*np.asarray(current['values']);cursor=row['dispatch_sim_ns'];current=row
        qualified &= end-current['dispatch_sim_ns']<=250_000_000
        # Need a recorded right boundary; never extrapolate beyond capture.
        qualified &= bool(events and events[-1]['dispatch_sim_ns']>=end)
        total+=(end-cursor)*np.asarray(current['values'])
        values.append((total/SLOT_NS).tolist());valid.append(bool(qualified))
    return dict(values=values,valid=valid,reason=None if all(valid) else 'dispatch_coverage_missing',
                action_semantics=ACTION_SEMANTICS,application_timestamps_observed=False)
