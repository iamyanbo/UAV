"""Explicit source migration using verified completed stages, never fake resume."""
import argparse
import copy
import json
from pathlib import Path
import re

from program_scheduler import digest,run


def connected_state(cycle):
    spec=json.loads((cycle/'programme-spec.json').read_text())
    state=json.loads((cycle/'execution/state.json').read_text())
    if spec.get('continuation_of'):
        prior=connected_state(Path(spec['continuation_of']))
        state['stages']={**prior['stages'],**state['stages']}
    return state


def continuation(cycle,start):
    spec=json.loads((cycle/'programme-spec.json').read_text())
    state=connected_state(cycle);stages=spec['stages']
    offset=[s['id'] for s in stages].index(start)
    prefix={s['id'] for s in stages[:offset]};refs=[];replacements={}
    for stage in stages[:offset]:
        row=state['stages'][stage['id']]
        if row['status']!='completed':raise ValueError('Cannot reuse incomplete stage: '+stage['id'])
        for ref in [row['worker_receipt'],*row.get('artifacts',[])]:
            if digest(ref['path'])!=ref['sha256']:raise ValueError('Changed completed artifact: '+ref['path'])
            refs.append(ref)
        for output in stage.get('outputs',[]):
            if output.startswith('{round}/'):
                directory=str(Path(output).parent)
                replacements[directory]=directory.replace('{round}',str(cycle/'execution'))
    def render(value):
        if isinstance(value,list):return [render(x) for x in value]
        if isinstance(value,dict):return {k:render(v) for k,v in value.items()}
        if not isinstance(value,str):return value
        def replace(match):
            name=match.group(1)
            return state['stages'][name]['job'] if name in prefix else match.group(0)
        value=re.sub(r'\{stage:([A-Za-z0-9_-]+):job\}',replace,value)
        for old,new in sorted(replacements.items(),key=lambda x:-len(x[0])):
            if value==old or value.startswith(old+'/'):value=new+value[len(old):]
        return value
    remaining=copy.deepcopy(stages[offset:])
    for stage in remaining:
        stage['depends_on']=[d for d in stage.get('depends_on',[]) if d['stage'] not in prefix]
    return dict(schema='training-dependencies/v1',continuation_of=str(cycle),
        reused_artifacts=refs,source_migration='New immutable source; optimizer resumes only inside matching dataset/objective',
        stages=render(remaining))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',type=Path,required=True)
    parser.add_argument('--start-at',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8);args=parser.parse_args()
    if not 0<args.hours<=8:parser.error('At most eight-hour windows')
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'programme-spec.json'
    if not path.exists():path.write_text(json.dumps(continuation(args.cycle.resolve(),args.start_at),indent=2))
    raise SystemExit(run(path,args.output/'execution',args.hours))
