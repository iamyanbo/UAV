"""Auditable gates; self-reported Qwen confidence is never a quality grade."""
import argparse
from collections import Counter
from pathlib import Path
import time
from .common import read,write,digest

AUDIT_CLASSES=('direct','detour','multiple_decisions','useful_climb','unnecessary_climb')


def quality_gate(audit,grades):
    a=read(audit);g=read(grades)
    if g.get('audit_sha256')!=digest(audit) or not g.get('reviewer') or g.get('grader')=='qwen':
        raise ValueError('Independent review bound to the exact audit required')
    rows=a['examples'];by_id={r['id']:r for r in g['examples']}
    if len(rows)!=30 or len(by_id)!=30 or len({r['current_sha256'] for r in rows})!=30:
        raise ValueError('Thirty distinct contexts and grades required')
    if Counter(r['category'] for r in rows)!=Counter({k:6 for k in AUDIT_CLASSES}):raise ValueError('Audit category coverage incomplete')
    if not any(r.get('ambiguous') for r in rows):raise ValueError('Audit must include insufficient/ambiguous evidence')
    grounded=actionable=hazardous=0
    for row in rows:
        for name in ('current','goal'):
            if digest(row[name])!=row[name+'_sha256']:raise ValueError('Audit image changed')
        grade=by_id[row['id']]
        if not grade.get('rationale'):raise ValueError('Every independent grade requires a rationale')
        if any(type(grade[k]) is not bool for k in ('grounded','actionable','hazardous')):raise ValueError('Boolean review labels required')
        grounded+=grade['grounded'] and not row.get('error');actionable+=grade['actionable'] and not row.get('error');hazardous+=grade['hazardous']
    review_s=g.get('review_elapsed_s')
    if not isinstance(review_s,(float,int)) or not 0<=review_s<=900:raise ValueError('Record independent review duration')
    passed=grounded>=27 and actionable>=24 and hazardous==0 and a['elapsed_s']+review_s<=900
    return dict(passed=passed,grounded=grounded,actionable=actionable,hazardous=hazardous,
        audit_sha256=digest(audit),grades_sha256=digest(grades),model_identity=a['model_identity'])


def audit(contexts,output,model_path):
    """Produce raw proposals, annotated inputs and a blank independent review form."""
    from PIL import Image
    import numpy as np
    from .compute import ComputeLane
    from .ppo_guidance import RegionProposer,request_context
    spec=read(contexts);rows=spec['examples']
    if len(rows)!=30 or Counter(r['category'] for r in rows)!=Counter({k:6 for k in AUDIT_CLASSES}):
        raise ValueError('Supply six examples in each of the five audit classes')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    # Loading is reported separately; the 15-minute audit clock begins warm.
    loaded=time.monotonic();proposer=RegionProposer(ComputeLane(maximum_slow_slice_s=900),path=model_path);load_s=time.monotonic()-loaded
    started=time.monotonic();result=[]
    for i,row in enumerate(rows):
        if time.monotonic()-started>=900:break
        record=dict(row,current_sha256=digest(row['current']),goal_sha256=digest(row['goal']))
        with Image.open(row['current']) as image:
            if image.size!=(640,480):raise ValueError('Audit requires an actual calibrated full camera image')
            rgb=np.asarray(image.convert('RGB')).tobytes()
        current=dict(id=i,stamp=float(i),rgb=rgb,path=row['current'])
        images,refs,spatial,_=request_context(current,Image.open(row['goal']).convert('RGB'),[],0)
        images[0].save(root/f'{i:02d}-references.png')
        try:
            goal,raw=proposer.region_proposal(images,refs,spatial,i,float(i))
            from dataclasses import asdict
            record.update(raw=raw,proposal=asdict(goal),references=refs)
        except Exception as error:record['error']=str(error)
        result.append(record)
        write(root/'audit.json',dict(schema='ppo-qwen-audit/v1',examples=result,load_s=load_s,
            elapsed_s=time.monotonic()-started,model_identity=proposer.base_identity))
    write(root/'grades-template.json',dict(audit_sha256=digest(root/'audit.json'),reviewer='',grader='independent',review_elapsed_s=None,
        examples=[dict(id=r['id'],grounded=None,actionable=None,hazardous=None,rationale='') for r in result]))


def workload_gate(receipt,identity,workers):
    r=read(receipt)
    if r.get('schema')!='ppo-workload-admission/v1' or r.get('identity')!=identity:raise ValueError('Workload admission does not bind this exact run')
    trials=r['trials'];one=next(t for t in trials if t['workers']==1)
    selected=next(t for t in trials if t['workers']==workers)
    for t in (one,selected):
        raw=read(t['evidence'])
        if digest(t['evidence'])!=t['evidence_sha256'] or raw['summary']!=t['summary']:raise ValueError('Workload evidence changed')
        s=t['summary']
        checks=(s['reset_starts']>=20,s['reset_exhausted']==0,s['reset_mean_s']<=30,
            s['pause_resume_passed'],
            s['source_age_p99_s']<.25,s['active_watchdog_faults']==0,s['fresh_proposal_fraction']>=.8,
            s['recording_failures']==0,.95<=s['sim_wall_ratio']<=1.05,
            s['peak_memory_bytes']+s['measured_growth_bytes']<=s['total_memory_bytes'],
            s['disk_free_bytes']>=s['projected_recording_bytes'],s['actor_qwen_recording_concurrent'])
        if not all(checks):raise ValueError('Concurrent workload failed admission')
    if workers==2 and selected['summary']['valid_transitions_per_wall_s']<1.1*one['summary']['valid_transitions_per_wall_s']:
        raise ValueError('Two workers do not improve measured throughput by ten percent')
    return selected


def assemble_workload(paths,output):
    trials=[];identity=None
    for path in paths:
        r=read(path)
        if r.get('failed') or r.get('schema')!='ppo-workload-trial/v1':raise ValueError('Failed/malformed workload trial')
        if identity is None:identity=r['identity']
        if r['identity']!=identity:raise ValueError('Workload comparison changed policy/tasks/source')
        trials.append(dict(workers=r['workers'],evidence=str(Path(path).resolve()),evidence_sha256=digest(path),
            summary=r['summary'],model_identity=r['model_identity']))
    if len({t['workers'] for t in trials})!=len(trials):raise ValueError('Duplicate worker trial')
    write(output,dict(schema='ppo-workload-admission/v1',identity=identity,trials=trials))
    for t in trials:workload_gate(output,identity,t['workers'])


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('audit');a.add_argument('--contexts',required=True);a.add_argument('--output',required=True);a.add_argument('--model-path',default='/models/qwen2.5-vl-3b')
    a=sub.add_parser('review');a.add_argument('--audit',required=True);a.add_argument('--grades',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('workload');a.add_argument('--trial',action='append',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.stage=='audit':audit(a.contexts,a.output,a.model_path)
    elif a.stage=='review':write(a.output,quality_gate(a.audit,a.grades))
    else:assemble_workload(a.trial,a.output)
