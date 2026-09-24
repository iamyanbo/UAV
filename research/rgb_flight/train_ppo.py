"""Bounded optimization of one fresh physical batch, followed by recollection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from learning_models import RecurrentPolicy
from ppo_learning import advantages,recurrent_ppo_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('/dataset'))
    parser.add_argument('--output',type=Path,default=Path('/output/ppo'));parser.add_argument('--policy',required=True)
    parser.add_argument('--goal-checkpoint',required=True)
    parser.add_argument('--resume');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--integration-only',action='store_true');parser.add_argument('--updates',type=int)
    args=parser.parse_args()
    if args.integration_only and args.updates!=1:parser.error('Integration requires exactly one update')
    root=args.dataset.resolve();manifest_path=root/'ppo.json';raw=manifest_path.read_bytes();manifest=json.loads(raw)
    if not manifest['actual_continuous_physics'] or not manifest['all_failures_retained']:
        raise ValueError('PPO accepts only retained physical simulator transitions')
    if manifest.get('reward_privileged_distance_progress_training_only') is not True:
        raise ValueError('PPO reward provenance must separate privileged progress from observations')
    if hashlib.sha256(Path(args.goal_checkpoint).read_bytes()).hexdigest()!=manifest.get('goal_encoder_checkpoint_sha256'):
        raise ValueError('PPO goal encoder differs from rollout feature provenance')
    behavior_hash=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    if manifest.get('behavior_policy_sha256')!=behavior_hash or not all(manifest.get(key) is True for key in
        ('behavior_frozen_during_collection','safety_filter_active','slow_planner_disabled')):
        raise ValueError('PPO requires a frozen matching behavior checkpoint, safety enabled and planner disabled')
    completed=manifest['completed_episodes']
    if not completed or len({row['attempt_id'] for row in completed})!=len(completed):
        raise ValueError('Unique completed-episode outcomes required for the collision constraint')
    if any(row['termination'] not in ('success','collision','geometry_collision','timeout','out_of_bounds','failure','initialization_timeout','tracking_recovery_timeout') for row in completed):
        raise ValueError('Collection truncations cannot count as completed episodes')
    paths=[];transition_ids=set()
    for item in manifest['shards']:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Modified PPO shard')
        paths.append(path)
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['behavior_policy_sha256']!=behavior_hash:
            raise ValueError('Mixed behavior checkpoints in one PPO collection batch')
        for key in ('transition_ids','latent_action','sampled_stop','executed_command','old_logprob','terminated','truncated'):
            if key not in data:raise ValueError('Missing PPO attribution: '+key)
        ids=data['transition_ids']
        if len(ids)!=int(data['valid'].sum()) or len(set(ids))!=len(ids) or transition_ids.intersection(ids):
            raise ValueError('Physical transition IDs duplicated or absent; warm-up context must not be recounted')
        transition_ids.update(ids)
    if not transition_ids or transition_ids.intersection(manifest.get('previously_consumed_transition_ids',[])):
        raise ValueError('Empty/reused physical rollout batch')
    args.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    policy=RecurrentPolicy(args.goal_checkpoint).cuda()
    initialization=torch.load(args.policy,map_location='cpu',weights_only=True)
    if manifest.get('perception_artifacts_sha256')!=initialization.get('perception_artifacts_sha256') or not initialization.get('perception_artifacts_sha256'):
        raise ValueError('Fresh rollouts must bind the behavior policy perception snapshots')
    policy.load_state_dict(initialization['model'],strict=True)
    dual=torch.nn.Parameter(torch.tensor(float(manifest.get('initial_collision_dual',0.)),device='cuda'))
    optimizer=torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),lr=3e-5)
    dual_optimizer=torch.optim.Adam([dual],lr=1e-3)
    transitions=len(transition_ids);updates=exposures=0;manifest_hash=hashlib.sha256(raw).hexdigest()
    schedule=[int(i) for _ in range(4) for i in rng.permutation(len(paths))]
    if args.updates is not None:
        if args.updates<1:raise ValueError('Positive bounded update budget required')
        schedule=schedule[:args.updates]
    early_stop=False;dual_updated=False
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved['manifest_sha256']!=manifest_hash: raise ValueError('PPO resume data changed')
        policy.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);dual.data.copy_(saved['dual'].cuda())
        dual_optimizer.load_state_dict(saved['dual_optimizer']);updates=saved['updates'];rng.bit_generator.state=saved['sample_rng']
        if saved['unique_physics_transitions']!=transitions:raise ValueError('Resume transition count changed')
        schedule=saved['schedule'];exposures=saved['optimizer_exposures'];early_stop=saved['kl_early_stop'];dual_updated=saved['dual_updated']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def checkpoint(name):
        state=dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),dual=dual.detach().cpu(),dual_optimizer=dual_optimizer.state_dict(),
                   module='policy',objective_version=initialization.get('objective_version'),
                   belief_version=initialization.get('belief_version'),action_semantics=initialization.get('action_semantics'),
                   training_method='fresh-physical-ppo/v2',update=initialization['update']+updates,
                   perception_artifacts_sha256=initialization['perception_artifacts_sha256'],
                   goal_encoder_checkpoint_sha256=initialization.get('goal_encoder_checkpoint_sha256'),
                   world_checkpoint_sha256=initialization.get('world_checkpoint_sha256'),critic=initialization.get('critic'),
                   unique_physics_transitions=transitions,updates=updates,optimizer_exposures=exposures,
                   manifest_sha256=manifest_hash,sample_rng=rng.bit_generator.state,schedule=schedule,
                   kl_early_stop=early_stop,dual_updated=dual_updated,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        pending=args.output/(name+'.pending');torch.save(state,pending);pending.replace(args.output/(name+'.pt'))
    outcomes=torch.tensor([bool(row['collision']) for row in completed],device='cuda')
    observed_cost=outcomes.float().mean()
    from parameter_evidence import ParameterEvidence
    evidence=ParameterEvidence(dict(policy=policy))
    while updates<len(schedule) and not early_stop and not Path('/output/CHECKPOINT_REQUEST').exists():
        batch=torch.load(paths[schedule[updates]],map_location='cuda',weights_only=True)
        if set(batch['runtime']) - {'image','current_tokens','goal_context','state','memory_context','task','previous_command','goal_tokens','target_context'}:
            raise ValueError('Privileged label leaked into PPO observation tensors')
        adv,returns=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminated'],batch['truncated'],dt=batch['delta_seconds'],valid=batch['valid'])
        cost_adv,cost_returns=advantages(batch['collision_cost'],batch['cost_value'],batch['next_cost_value'],batch['terminated'],batch['truncated'],dt=batch['delta_seconds'],valid=batch['valid'])
        batch.update({'advantage':adv,'return':returns,'collision_cost_advantage':cost_adv,
                      'collision_cost_return':cost_returns,'completed_episode_collision':outcomes,
                      'expected_behavior_policy_sha256':behavior_hash})
        optimizer.zero_grad(set_to_none=True);dual_optimizer.zero_grad(set_to_none=True)
        objective,dual_loss,observed_cost,kl=recurrent_ppo_loss(policy,batch,collision_lagrange=dual)
        if not torch.isfinite(objective+dual_loss): raise RuntimeError('Nonfinite constrained PPO objective')
        if not torch.isfinite(kl):raise RuntimeError('Nonfinite PPO KL')
        if float(kl)>.02:
            early_stop=True;break
        objective.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);evidence.observe_gradients();optimizer.step()
        exposures+=int(batch['valid'][:,20:].sum());updates+=1
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=updates,unique_physics_transitions=transitions,
                                         optimizer_exposures=exposures,approximate_kl=float(kl),policy_loss=float(objective.detach()),
                                         collision_cost=float(observed_cost),collision_dual=float(dual.detach())))+'\n')
        if updates%50==0: checkpoint('latest')
    finished=updates==len(schedule) or early_stop
    if finished and not dual_updated:
        dual_optimizer.zero_grad(set_to_none=True)
        (-(dual*(observed_cost-.01))).backward();dual_optimizer.step()
        with torch.no_grad():dual.clamp_(0,100)
        dual_updated=True
    checkpoint('final' if finished else 'latest')
    (args.output/'consumed-transitions.json').write_text(json.dumps(sorted(transition_ids)))
    result=dict(status='completed' if finished else 'checkpointed',accepted=False,
                integration_only=args.integration_only,gradient_evidence=evidence.finish() if updates else None,
                unique_physics_transitions=transitions,optimizer_exposures=exposures,updates=updates,
                max_epochs=4,kl_early_stop=early_stop,requires_fresh_physical_rollouts=finished,
                completed_episode_collision_rate=float(observed_cost),constrained_collision_cost=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__': main()
