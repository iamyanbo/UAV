"""Resumable physical-trajectory training. No synthetic or privileged inputs.

Bundle manifest: foundation receipt, disjoint train/validation episode records,
and window shards. Each shard contains runtime tensors, separately nested
training_labels, causal source/latest-observation ns, and its episode ID.
World windows contain >=10 context +20 prediction intervals at 0.2 s. Policy
windows contain >=20 context +20 learning frames at 20 Hz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from learning_models import WorldModel, RecurrentPolicy, PrimitiveCritic, bootstrapped_world_loss


def objective_version(module):
    return 'masked-dispatched-state-fields/v4' if module=='world' else 'masked-dispatched-sequences/v3'


class TrajectoryBundle:
    def __init__(self, root, module, integration_only=False):
        self.integration_only=integration_only
        self.root = Path(root).resolve()
        path = self.root / 'manifest.json'
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = json.loads(path.read_text())
        self.goal_encoder_sha256 = self.manifest.get('goal_encoder_checkpoint_sha256')
        if not isinstance(self.goal_encoder_sha256, str) or len(self.goal_encoder_sha256) != 64:
            raise ValueError('Trajectory windows require a versioned frozen goal encoder')
        foundation = self.manifest['foundation']
        if not foundation.get('training_ready',False) or not foundation['visual_goal_runtime']:
            raise ValueError('Training requires valid causal data and disjoint development episodes')
        self.episodes = {x['episode_id']: x for x in self.manifest['episodes']}
        groups = {}
        for episode in self.episodes.values():
            for field in ('goal_region_id', 'start_goal_pair_id'):
                key = field, episode[field]
                if key in groups and groups[key] != episode['split']:
                    raise ValueError('Goal region or start-goal pair crosses dataset splits')
                groups[key] = episode['split']
        self.records = {split: [x for x in self.manifest['windows'] if x['module'] == module and self.episodes[x['episode_id']]['split'] == split]
                        for split in ('train', 'validation')}
        if module=='world':
            self.records={split:[x for x in rows if x.get('action_timing_valid',False)] for split,rows in self.records.items()}
        if not self.records['train'] or not self.records['validation']:
            raise ValueError('Both training and validation trajectory windows are required')
        self.module = module
        self.exposed_windows = set()
        self.exposed_episodes = set()

    def load(self, record):
        path = (self.root / record['path']).resolve()
        if not path.is_relative_to(self.root) or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Window path/checksum mismatch')
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['episode_id'] != record['episode_id']:
            raise ValueError('Window episode mismatch')
        allowed_teachers={'observed_frontier_local_expert','observed-exploration/v3'}
        if self.module == 'policy' and data.get('teacher_source') not in allowed_teachers:
            raise ValueError('Search imitation requires observation-conditioned target selection before local expert execution')
        runtime = data['runtime']
        allowed = {'z', 'state', 'belief', 'memory', 'memory_valid', 'action', 'task', 'goal_tokens',
                   'target_context', 'image', 'previous_command',
                   'current_tokens','goal_context',
                   'maximum_speed_mps',
                   'sim_ns', 'latest_observation_ns', 'visual_available'}
        if set(runtime) - allowed:
            raise ValueError('Unexpected runtime fields; labels must remain in training_labels')
        if not bool((runtime['latest_observation_ns'] <= runtime['sim_ns']).all()):
            raise ValueError('Future observations in a historical input')
        if self.module=='world' and (runtime['action'].shape[-2:]!=(4,4) or
                not bool(data['training_labels']['action_interval_valid'][:30].all())):
            raise ValueError('World training requires four timestamp-qualified executed command slots per interval')
        delta = runtime['sim_ns'][1:] - runtime['sim_ns'][:-1]
        if self.module=='world' and not bool((delta==200000000).all()):
            raise ValueError('World windows require a declared 200 ms prediction grid')
        if self.module=='policy' and not bool(((delta>0)&(delta<=250000000)).all()):
            raise ValueError('Policy windows require fresh consecutive observations within 75 ms; never duplicate RGB')
        if self.module=='world':
            for name in ('z','state','motion','collision','visibility','information','goal_match','time_to_goal'):
                if name+'_valid' not in data['training_labels']:raise ValueError('Missing explicit supervision mask: '+name)
        required = 31 if self.module == 'world' else 40
        if len(runtime['sim_ns']) < required:
            raise ValueError('Insufficient recurrent warm-up and rollout coverage')
        return data

    def batch(self, indices, split):
        if split == 'train':
            self.exposed_windows.update(int(i) for i in indices)
            self.exposed_episodes.update(self.records[split][int(i)]['episode_id'] for i in indices)
        examples = [self.load(self.records[split][i]) for i in indices]
        runtime = {k: torch.stack([x['runtime'][k] for x in examples]).cuda() for k in examples[0]['runtime']}
        labels = {k: torch.stack([x['training_labels'][k] for x in examples]).cuda() for k in examples[0]['training_labels']}
        masks = []
        for example in examples:
            digest = hashlib.sha256(example['episode_id'].encode()).digest()
            mask = [bool(digest[h] & 1) for h in range(3)]
            if not any(mask):
                mask[digest[3] % 3] = True
            masks.append(mask)
        return runtime, labels, torch.tensor(masks, device='cuda').T


def world_objective(model, runtime, labels, masks):
    batch = runtime['z'].shape[0]
    hidden = runtime['z'].new_zeros(batch, 256)
    # During warm-up the slow representation is delivered only when available.
    latest = runtime['z'][:, 0]
    with torch.no_grad():
        for step in range(10):
            fresh = runtime['visual_available'][:, step, None, None].bool()
            latest = torch.where(fresh, runtime['z'][:, step], latest)
            prediction = model(latest, runtime['state'][:, step], hidden, runtime['memory'][:, step],
                               runtime['memory_valid'][:, step], runtime['action'][:, step], runtime['task'][:, step],
                               runtime['goal_tokens'], runtime['target_context'][:, step])
            hidden = prediction['belief'].mean(0)
    z = torch.where(runtime['visual_available'][:, 10, None, None].bool(), runtime['z'][:, 10], latest)
    state = runtime['state'][:, 10]
    # The full imagined future sees only memory available at its origin.
    memory, valid = runtime['memory'][:, 10], runtime['memory_valid'][:, 10]
    total = z.new_zeros(())
    for offset in range(20):
        step = 10 + offset
        prediction = model(z, state, hidden, memory, valid, runtime['action'][:, step], runtime['task'][:, 10],
                           runtime['goal_tokens'], runtime['target_context'][:, 10])
        target = dict(z=runtime['z'][:, step + 1], state=runtime['state'][:, step + 1],
                      **{k: labels[k][:, step] for k in ('motion', 'collision', 'visibility', 'information',
                                                        'goal_match', 'time_to_goal')})
        target.update({name+'_valid':labels[name+'_valid'][:,step] for name in
                       ('z','state','motion','collision','visibility','information','goal_match','time_to_goal')})
        target['time_censored']=labels['time_censored'][:,step]
        weight = 2 if offset in (4, 9, 14, 19) else 1
        total = total + weight * bootstrapped_world_loss(prediction, target, masks, model.feature_scale, model.state_scale, model.motion_scale)
        z, state, hidden = prediction['z'].mean(0), prediction['state'].mean(0), prediction['belief'].mean(0)
    return total / 24


def policy_objective(policy, critic, runtime, labels):
    hidden = policy.gru.weight_hh.new_zeros(len(runtime['state']), 256)
    losses = []
    for step in range(40):
        memory = runtime['memory'][:, step, :, :256]
        mask = runtime['memory_valid'][:, step, :, None]
        context = (memory * mask).sum(1) / mask.sum(1).clamp_min(1)
        with torch.set_grad_enabled(step >= 20 and policy.training):
            if 'current_tokens' in runtime:
                distribution,value,hidden=policy.forward_features(runtime['current_tokens'][:,step],runtime['goal_context'][:,step],
                    runtime['state'][:,step],context,runtime['task'][:,step],runtime['previous_command'][:,step],hidden,
                    runtime['target_context'][:,step])
            else:
                distribution, value, hidden = policy(runtime['image'][:, step], runtime['state'][:, step], context,
                                                runtime['task'][:, step], runtime['previous_command'][:, step], hidden,
                                                runtime['goal_tokens'], runtime['target_context'][:, step])
        if step < 20:
            hidden = hidden.detach()
            continue
        speed = runtime['maximum_speed_mps'][:,step]
        limits = torch.stack((speed,speed,torch.ones_like(speed),torch.full_like(speed,45)),1)
        trustworthy=labels['imitation_valid'][:,step].bool()
        error=F.smooth_l1_loss(policy.command(distribution.mean,speed) / limits,
                             labels['expert_command'][:, step].detach() / limits,reduction='none').mean(-1)
        action_loss=(error*trustworthy).sum()/trustworthy.sum().clamp_min(1)
        stop_loss=F.binary_cross_entropy_with_logits(policy.stop_head(hidden).squeeze(-1),
            labels['explicit_stop'][:,step].detach().float(),reduction='none')
        stop_loss=(stop_loss*trustworthy).sum()/trustworthy.sum().clamp_min(1)
        # The planner's recurrent world belief and the fast policy's GRU are
        # different representations. Train the terminal critic on the former.
        remaining = critic(runtime['z'][:, step], runtime['state'][:, step], runtime['belief'][:, step],
                           runtime['task'][:, step], runtime['goal_tokens'],
                           runtime['target_context'][:, step])
        primitive_error=F.smooth_l1_loss(remaining/critic.primitive_scale,
            labels['primitive_return'][:,step].detach()/critic.primitive_scale,reduction='none')
        primitive_mask=labels['primitive_valid'][:,step].bool()
        primitive_loss=(primitive_error*primitive_mask).sum()/primitive_mask.sum().clamp_min(1)
        cost_loss=F.smooth_l1_loss(policy.collision_value(hidden).squeeze(-1),labels['collision_return'][:,step].detach())
        # Entropy is deliberately left to PPO; imitation updates the mean only.
        losses.append(action_loss+stop_loss+.1*F.smooth_l1_loss(value,labels['fixed_return'][:,step].detach())+
                      primitive_loss+.1*cost_loss)
    return torch.stack(losses).mean()


def save_checkpoint(path, model, critic, optimizer, scheduler, update, best, rng, bundle):
    state = dict(model=model.state_dict(), critic=critic.state_dict() if critic else None,
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), update=update,
                 best_validation=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                 sample_rng=rng.bit_generator.state, python_rng=random.getstate(), manifest_sha256=bundle.digest)
    state['exposed_train_window_indices'] = sorted(bundle.exposed_windows)
    state['exposed_train_episode_ids'] = sorted(bundle.exposed_episodes)
    state.update(module=bundle.module,objective_version=objective_version(bundle.module),
        action_semantics='post-safety-dispatch/50ms-v3',belief_version='masked-map-dispatch-state/v3',teacher_version='observed-exploration/v3',
        goal_encoder_checkpoint_sha256=bundle.goal_encoder_sha256,
        world_checkpoint_sha256=bundle.manifest.get('world_checkpoint_sha256'),
        projection_sha256=bundle.manifest.get('projection_sha256'),
        navigation_checkpoint_set_sha256=bundle.manifest.get('navigation_checkpoint_set_sha256'))
    state['perception_artifacts_sha256']=bundle.manifest.get('perception_artifacts_sha256')
    temporary = path.with_suffix('.pending')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', choices=['world', 'policy'], required=True)
    parser.add_argument('--dataset', default='/dataset')
    parser.add_argument('--output', default='/output/training')
    parser.add_argument('--resume')
    parser.add_argument('--initialize-from')
    parser.add_argument('--goal-checkpoint', help='Frozen shared spatial goal checkpoint for policy imitation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--integration-only',action='store_true')
    parser.add_argument('--updates',type=int)
    args = parser.parse_args()
    if args.integration_only and args.updates!=1:parser.error('Unqualified integration rounds permit exactly one update')
    bundle = TrajectoryBundle(args.dataset, args.module,args.integration_only)
    config = json.loads((Path(__file__).parent / 'campaign.json').read_text())['training']
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    is_world = args.module == 'world'
    if not is_world and not args.goal_checkpoint:
        parser.error('Policy imitation requires --goal-checkpoint')
    if not is_world and hashlib.sha256(Path(args.goal_checkpoint).read_bytes()).hexdigest() != bundle.goal_encoder_sha256:
        raise ValueError('Policy goal checkpoint differs from trajectory-token provenance')
    target = config['world']['updates'] if is_world else config['policy']['imitation_updates']
    programme_target=target
    if args.updates is not None:
        if args.updates<1 or args.updates>target:parser.error('Invalid update budget')
        target=args.updates
    reference = config['reference_budgets']['world_updates' if is_world else 'imitation_updates']
    effective_batch = 16 if is_world else 32
    if args.integration_only:effective_batch=2
    model = (WorldModel() if is_world else RecurrentPolicy(args.goal_checkpoint)).cuda()
    critic = None if is_world else PrimitiveCritic().cuda()
    normalizer_path = (bundle.root / bundle.manifest['normalization']['path']).resolve()
    if not normalizer_path.is_relative_to(bundle.root) or hashlib.sha256(normalizer_path.read_bytes()).hexdigest() != bundle.manifest['normalization']['sha256']:
        raise ValueError('Missing or modified training-fitted normalization artifact')
    normalization = torch.load(normalizer_path, weights_only=True, map_location='cpu')
    if normalization['fit_split'] != 'train' or any(bundle.episodes[episode]['split'] != 'train' for episode in normalization['episode_ids']):
        raise ValueError('Normalization contains held-out data')
    with torch.no_grad():
        for name in ('state_mean', 'state_scale', 'motion_scale', 'feature_scale') if is_world else ('primitive_scale',):
            destination = getattr(model if is_world else critic, name)
            source = normalization[name]
            if source.shape != destination.shape or not torch.isfinite(source).all() or (name != 'state_mean' and not (source > 0).all()):
                raise ValueError('Invalid training normalization: ' + name)
            destination.copy_(source)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += [] if critic is None else list(critic.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, programme_target, eta_min=1e-6)
    update, best = 0, float('inf')
    if args.initialize_from:
        if args.resume:raise ValueError('Initialization is a new round, not exact resume')
        initial=torch.load(args.initialize_from,map_location='cpu',weights_only=True)
        if initial.get('module')!=args.module or initial.get('perception_artifacts_sha256')!=bundle.manifest.get('perception_artifacts_sha256'):
            raise ValueError('Initialization module or perception provenance differs')
        model.load_state_dict(initial['model'],strict=True)
        if critic:critic.load_state_dict(initial['critic'],strict=True)
        with torch.no_grad():
            for name in ('state_mean','state_scale','motion_scale','feature_scale') if is_world else ('primitive_scale',):
                getattr(model if is_world else critic,name).copy_(normalization[name])
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved['manifest_sha256'] != bundle.digest or saved.get('objective_version')!=objective_version(args.module):
            raise ValueError('Resume dataset changed; create an explicit new training round')
        model.load_state_dict(saved['model'], strict=True)
        if critic:
            critic.load_state_dict(saved['critic'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        update, best = saved['update'], saved['best_validation']
        rng.bit_generator.state = saved['sample_rng']
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        bundle.exposed_windows.update(saved['exposed_train_window_indices'])
        bundle.exposed_episodes.update(saved['exposed_train_episode_ids'])
    started = time.monotonic()
    resumed_update=update
    from parameter_evidence import ParameterEvidence
    evidence=ParameterEvidence(dict(model=model,**({'critic':critic} if critic else {})))
    def objective(indices, split):
        runtime, labels, mask = bundle.batch(indices, split)
        # Recurrent warm-up uses no_grad. Autocast's weight cache otherwise
        # reuses detached casts in the learning segment of this same context.
        with torch.autocast('cuda', dtype=torch.bfloat16,cache_enabled=False):
            return world_objective(model, runtime, labels, mask) if is_world else policy_objective(model, critic, runtime, labels)
    while update < target:
        if Path('/output/CHECKPOINT_REQUEST').exists():
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = rng.integers(len(bundle.records['train']), size=effective_batch)
        training_loss = 0.
        # Accumulation limits activation peaks without changing effective batch.
        for chunk in np.array_split(indices, effective_batch // 2):
            loss = objective(chunk, 'train') / (effective_batch // 2)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite learning objective; last checkpoint retained')
            loss.backward()
            training_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        evidence.observe_gradients()
        optimizer.step()
        scheduler.step()
        update += 1
        validation = None
        if (update % 2000 == 0 or update == target) and bundle.records['validation']:
            model.eval()
            values = []
            with torch.no_grad():
                for offset in range(0, len(bundle.records['validation']), 2):
                    indices = list(range(offset, min(offset + 2, len(bundle.records['validation']))))
                    values.append((float(objective(indices, 'validation')), len(indices)))
            validation = float(sum(value*count for value,count in values)/sum(count for _,count in values))
            if validation < best:
                best = validation
                save_checkpoint(output / 'best.pt', model, critic, optimizer, scheduler, update, best, rng, bundle)
        if update % 200 == 0 or update in (reference, target):
            save_checkpoint(output / 'latest.pt', model, critic, optimizer, scheduler, update, best, rng, bundle)
        if update == reference:
            save_checkpoint(output / 'reference-budget.pt', model, critic, optimizer, scheduler, update, best, rng, bundle)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(update=update, training_loss=training_loss, validation_loss=validation,
                                         window_exposures=update * effective_batch, unique_train_windows=len(bundle.records['train']),
                                         exposed_unique_train_windows=len(bundle.exposed_windows),
                                         exposed_unique_train_episodes=len(bundle.exposed_episodes),
                                         dataset_train_simulated_hours=sum(x['simulated_seconds'] for x in bundle.manifest.get('attempts',list(bundle.episodes.values())) if x['split']=='train')/3600,
                                         elapsed_seconds=time.monotonic() - started)) + '\n')
    save_checkpoint(output / ('final.pt' if update >= target else 'latest.pt'), model, critic, optimizer, scheduler, update, best, rng, bundle)
    result = dict(status='completed' if update >= target else 'checkpointed', updates=update,accepted=False,
                  objective_version=objective_version(args.module),
                  integration_only=args.integration_only,gradient_evidence=evidence.finish(require_update=update>resumed_update),
                  resumed_from_update=resumed_update,optimizer_states=len(optimizer.state),
                  foundation_source=bundle.manifest['foundation'], dataset_manifest_sha256=bundle.digest,
                  scope='world training' if is_world else 'policy imitation and primitive critic; PPO not executed')
    reloaded=torch.load(output/('final.pt' if update>=target else 'latest.pt'),map_location='cpu',weights_only=True)
    result['checkpoint_reload_verified']=all(torch.equal(value.detach().cpu(),reloaded['model'][key]) for key,value in model.state_dict().items())
    result['optimizer_reload_states']=len(reloaded['optimizer']['state'])
    if not result['checkpoint_reload_verified']:raise RuntimeError('Saved checkpoint differs from updated model')
    del reloaded
    result['initialization']=dict(sha256=hashlib.sha256(Path(args.initialize_from).read_bytes()).hexdigest(),
        update=initial['update']) if args.initialize_from else None
    intended=[name for name,row in result['gradient_evidence'].items() if row['trainable'] and name!='model.log_std']
    missing=[name for name in intended if not result['gradient_evidence'][name]['nonzero_gradient_tensors']
             or not result['gradient_evidence'][name]['changed_tensors']]
    result['gradient_gate_passed']=not missing
    if missing and update>resumed_update:
        result.update(status='failed',reason='Missing intended gradient or parameter update',missing_groups=missing)
    if is_world and update:
        model.eval();runtime,_,_=bundle.batch([0],'train')
        action=runtime['action'][:,10].detach().clone().requires_grad_(True)
        prediction=model(runtime['z'][:,10],runtime['state'][:,10],runtime['z'].new_zeros(1,256),
            runtime['memory'][:,10],runtime['memory_valid'][:,10],action,runtime['task'][:,10],
            runtime['goal_tokens'],runtime['target_context'][:,10])
        gradient=torch.autograd.grad(prediction['motion'].square().sum()+prediction['z'].square().mean(),action)[0]
        result['candidate_action_gradient']=dict(finite=bool(torch.isfinite(gradient).all()),norm=float(gradient.norm()))
        if not result['candidate_action_gradient']['finite'] or not result['candidate_action_gradient']['norm']:
            raise RuntimeError('Candidate actions do not affect predicted futures')
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if result['status']=='failed':raise SystemExit(2)


if __name__ == '__main__':
    main()
