"""Connect complete-flight outcomes to Qwen tuning, preferences and PPO."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from goal_io import load_goal
from .common import read,write,digest
from .data import lines,rgb_at
from .contracts import Subgoal,snapshot_from_dict
from .maps import MapPrior


def build(dataset,output,component,behavior_checkpoint=None):
    spec=read(dataset)
    if spec['schema']!='photo-map-dataset/v6':raise ValueError('Temporal dataset required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[];pairs=defaultdict(list);by_episode=defaultdict(list)
    for index,w in enumerate(spec['windows']):by_episode[w['episode']].append((index,w))
    behavior=digest(behavior_checkpoint) if behavior_checkpoint else None
    if component=='ppo' and behavior is None:raise ValueError('PPO needs packaged behavior model identity')
    for ei,ep in enumerate(spec['episodes']):
        root=Path(ep['path']);result=read(root/'result.json')
        path=root/'photo-controller/runtime/decisions.jsonl'
        if not path.exists():continue
        decisions=lines(path);by_frame={d['frame_id']:d for d in decisions}
        images=lines(root/'observations/frames.jsonl');image_index={r['frame_id']:r for r in images}
        artifacts={str(path):digest(path),str(root/'result.json'):digest(root/'result.json')}
        command_path=root/'training_labels/commands.jsonl'
        artifacts[str(command_path)]=digest(command_path)
        dispatched=defaultdict(list);events=lines(command_path)
        for a,b in zip(events,events[1:]):
            if a.get('decision_id') and 0<b['dispatch_sim_ns']-a['dispatch_sim_ns']<=250_000_000:
                dispatched[a['decision_id']].append(dict(a,exposure_ns=b['dispatch_sim_ns']-a['dispatch_sim_ns']))
        collided=result.get('termination') in ('collision','geometry_collision')
        if component=='ppo':
            for index,w in by_episode[ei]:
                decision=by_frame.get(images[w['frame']]['frame_id'])
                if not decision or decision.get('sampled_latent') is None:continue
                exposure=dispatched.get(decision.get('decision_id'),[])
                if not any(not d['watchdog_override'] for d in exposure):continue
                if w.get('decision_id')!=decision['decision_id']:raise ValueError('Behavior context/dispatch join mismatch')
                if decision['behavior_sha256']!=behavior:raise ValueError('Mixed behavior checkpoints')
                row=dict(id=f'{ei}-{index}',split=ep['split'],scene_id=ep['scene_id'],artifacts=artifacts,
                    window_index=index,sampled_latent=decision['sampled_latent'],behavior_logprob=decision['behavior_logprob'],
                    decision_id=decision['decision_id'],dispatch_exposure=exposure,
                    behavior_sha256=behavior,behavior_subgoal=decision['context']['subgoal'],
                    proposed_action=decision['proposed_command'],safety_action=decision['command'],
                    reward_return=float(bool(result.get('success')))-.001*max(0.,result.get('elapsed_sim_seconds',0)-decision['observed_s']+result.get('start_sim_seconds',images[0]['sim_ns']/1e9)),
                    cost_return=float(collided)+.01*float(decision['braked'])+.01*sum(d['watchdog_override'] for d in exposure))
                rows.append(row)
            continue
        used=set()
        # Only controlled branch interventions establish proposal attribution.
        if not result.get('branch_protocol') or result.get('branch_protocol')!='settled-restart/v2':continue
        if not result.get('branch_context_sha256') or not result.get('branch_continuation_sha256'):continue
        for decision in decisions:
            if component=='preferences' and (used or not result.get('initial_subgoal')):break
            slow=decision.get('slow') or {}
            selected=slow.get('selected')
            if result.get('initial_subgoal') and not used:
                selected=decision.get('selected_subgoal');source=decision['frame_id']
            elif selected:source=slow['source_frame']
            else:continue
            if selected is None or decision.get('warmup'):continue
            if source in used or source not in image_index:continue
            used.add(source);goal=Subgoal(**selected)
            source_decision=by_frame.get(source)
            if source_decision is None:continue
            spatial=snapshot_from_dict(source_decision['context']['spatial'])
            if not goal.valid(image_index[source]['sim_ns']/1e9,spatial):continue
            ident=f'{ei}-{source}';paths=[]
            def save_image(name,raw):
                p=out/(ident+'-'+name+'.png');Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(p);paths.append(str(p.resolve()))
            save_image('current',rgb_at(root,image_index[source]));save_image('goal',load_goal(root/'goal').rgb_views[0])
            p=out/(ident+'-map.png');MapPrior(ep['map']).annotated(spatial.map_lookup).save(p);paths.append(str(p.resolve()))
            for key,frame,_ in spatial.keyframes:
                if frame in image_index:save_image(key,rgb_at(root,image_index[frame]))
            proposal=dict(intention=goal.intention,target_reference=goal.target_reference,target_source=goal.target_source,
                          altitude=goal.altitude,confidence=goal.confidence,horizon_s=goal.expires_s-goal.source_s)
            answer=json.dumps(dict(reasoning='Supported by recorded complete-flight outcome.',proposals=[proposal]))
            row=dict(id=ident,split=ep['split'],scene_id=ep['scene_id'],images=paths,spatial=asdict(spatial),answer=answer,
                success=int(bool(result.get('success'))),cost=int(collided),artifacts=dict(artifacts,**{p:digest(p) for p in paths}))
            timeout=read(root/'evaluator_labels/episode.json')['timeout_s']
            row['duration_s']=result.get('elapsed_sim_seconds',timeout)
            row['intervention_rate']=sum(bool(d.get('braked')) for d in decisions)/max(1,len(decisions))
            row['utility']=(100. if row['success'] else -100. if collided else -25.)-min(row['duration_s']/timeout,1.)-.01*row['intervention_rate']
            if result.get('branch_group'):
                pairs[(result['branch_group'],ep['split'],result['seed'],result['branch_context_sha256'],
                       result['branch_continuation_sha256'])].append(row)
    if component in ('preferences','configurator'):
        for key,group in pairs.items():
            if len(group)<2:continue
            best=max(group,key=lambda r:r['utility']);worst=min(group,key=lambda r:r['utility'])
            if best['utility']<=worst['utility']:continue
            # Only initial branch proposals with no target can transfer across
            # independent map initialization; do not transplant obsolete IDs.
            a=json.loads(best['answer'])['proposals'][0];b=json.loads(worst['answer'])['proposals'][0]
            if a['target_reference'] is not None or b['target_reference'] is not None:continue
            if component=='configurator':
                if best['success'] and not best['cost']:rows.append(best)
                continue
            rows.append(dict(best,chosen=best['answer'],rejected=worst['answer'],
                chosen_utility=best['utility'],rejected_utility=worst['utility'],
                chosen_success=best['success'],rejected_success=worst['success'],chosen_cost=best['cost'],rejected_cost=worst['cost'],
                artifacts=dict(best['artifacts'],**worst['artifacts']),branch_group=key[0]))
    if not rows:raise ValueError('No eligible causal adaptation examples')
    write(out/'dataset.json',dict(schema='temporal-adaptation/v2',rows=rows,component=component,
        base_dataset=str(Path(dataset).resolve()),base_dataset_sha256=digest(dataset),
        behavior_checkpoint=str(Path(behavior_checkpoint).resolve()) if behavior_checkpoint else None,
        behavior_checkpoint_sha256=behavior,cost_multiplier=2.,
        train_scenes=sorted({e['scene_id'] for e in spec['episodes'] if e['split']=='train'}),
        validation_scenes=sorted({e['scene_id'] for e in spec['episodes'] if e['split']=='validation'})))
