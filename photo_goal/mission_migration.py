"""Explicit, idempotent phase forks. Ordinary resume remains strict."""
import math
from pathlib import Path
import torch
from .common import read, write, digest, FlightLock
from .mission_contracts import city_config, identity
from .mission_checkpoint import load_components, save_bundle, cpu_copy, restore_rng


MOMENTS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')
PHASES = {'stop': 'A', 'motion': 'B', 'tasks': 'C', 'reward': 'D'}


def clear_slice(optimizer, parameter, index):
    for key in MOMENTS:
        value = optimizer.state.get(parameter, {}).get(key)
        if value is not None:
            value[index].zero_()


def optimizer_snapshot(model,optimizer):
    return {name:cpu_copy(optimizer.state.get(parameter,{})) for name,parameter in model.named_parameters()}


def verify_optimizer(model,optimizer,before,removed=(),slices=None):
    slices=slices or {}
    for name,parameter in model.named_parameters():
        after=optimizer.state.get(parameter,{})
        if name in removed:
            if after:raise RuntimeError('Reset critic retained optimizer state: '+name)
            continue
        old=before[name]
        if set(old)!=set(after):raise RuntimeError('Optimizer state keys changed: '+name)
        for key,value in old.items():
            if torch.is_tensor(value):
                expected=value.clone()
                if name in slices and key in MOMENTS:expected[slices[name]].zero_()
                if not torch.equal(expected,after[key].cpu()):raise RuntimeError('Unexpected optimizer migration: '+name+'/'+key)
            elif value!=after[key]:raise RuntimeError('Unexpected optimizer metadata change: '+name+'/'+key)


def run(args):
    parent, child = city_config(args.parent_config), city_config(args.config)
    letter = PHASES[args.phase]
    if child.get('phase_id') != 'city-repair-'+letter:
        raise ValueError('Phase/config mismatch')
    expected_parent = {'B': 'A', 'C': 'B', 'D': 'C'}.get(letter)
    if expected_parent and parent.get('phase_id') != 'city-repair-'+expected_parent:
        raise ValueError('Repair phases must proceed A -> B -> C -> D')
    allowed = {'phase_id', 'stop_class_balance', 'stop_class_weight_cap', 'control_step_s',
               'stop_reference_step_s', 'motor_control', 'task_sampling', 'world_updates_per_batch',
               'potential_normalization', 'potential_distance_scale_m'}
    allowed |= {'A': {'stop_prior','resources'}, 'B': {'step_s', 'motor_control'},
                'C': {'task_sampling', 'task_start_weights', 'max_route_length_m',
                      'task_height_headroom_m', 'task_roi_padding_m', 'geometry_tile_size_m',
                      'geometry_resolution_m', 'route_cache_max_bytes', 'route_cache_max_entries',
                      'extra_frame_queue_capacity', 'intermediate_deadline_s', 'support_deadline_s'},
                'D': {'time_cost_per_mission', 'failure_remaining_time_charge',
                      'potential_normalization', 'potential_distance_scale_m'}}[letter]
    changes = {k for k in set(parent)|set(child) if parent.get(k) != child.get(k)}
    if changes-allowed:
        raise ValueError('Unsupported fork configuration changes: '+str(sorted(changes-allowed)))
    if 'resources' in changes and child['resources']!=dict(parent['resources'],gpu_fraction_ceiling=1.):
        raise ValueError('Only the authorized full-GPU allowance may change resource config')
    output = Path(args.run_dir).resolve()
    if not output.is_relative_to(Path(args.root).resolve()):
        raise ValueError('Fork output must be inside project root')
    key = identity(dict(parent=digest(args.checkpoint), config=identity(child), phase=args.phase))
    receipt_path, checkpoint = output/'migration.json', output/'latest.pt'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['fork_id'] != key or digest(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('Existing fork identity differs or was already trained')
        return receipt
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fork directory is not empty; reconcile interrupted publication explicitly')
    with FlightLock(args.root, 'city-phase-fork'):
        actor, optimizer, world, world_optimizer, meta = load_components(args.checkpoint, args.backbone, parent, args.device)
        if expected_parent and meta.get('pending'):
            raise ValueError('Finish the previous phase before migrating a partial batch')
        if expected_parent and meta.get('phase_accepted_batches',0) != 2:
            raise ValueError('Previous repair phase must have exactly two accepted batches')
        if meta.get('world_pending'):
            raise ValueError('Complete pending independent world work before a fork')
        restore_rng(meta)
        before_actor, before_world = cpu_copy(actor.state_dict()), cpu_copy(world.state_dict())
        actor_moments=optimizer_snapshot(actor,optimizer);world_moments=optimizer_snapshot(world,world_optimizer)
        actor_groups=cpu_copy(optimizer.state_dict()['param_groups']);world_groups=cpu_copy(world_optimizer.state_dict()['param_groups'])
        changed_actor, changed_world = set(), set()
        with torch.no_grad():
            if letter == 'A':
                bias = actor.actor.action[-1].bias
                if bias.shape != (5,) or parent['stop_prior'] != .01:
                    raise ValueError('Expected the inherited five-output actor and 0.01 reference prior')
                bias[4] += math.log(.00025/(1-.00025))-math.log(.01/(1-.01))
                clear_slice(optimizer, bias, 4)
                changed_actor.add('actor.action.2.bias')
            if letter == 'D':
                for prefix in ('value', 'execution_value'):
                    layer = getattr(actor,prefix)[-1]
                    for suffix, parameter in layer.named_parameters():
                        parameter.zero_(); optimizer.state.pop(parameter,None)
                        name = next(n for n,p in actor.named_parameters() if p is parameter)
                        changed_actor.add(name)
                layer = world.outcomes[-1]
                for parameter in (layer.weight,layer.bias):
                    parameter[0].zero_(); clear_slice(world_optimizer,parameter,0)
                    changed_world.add(next(n for n,p in world.named_parameters() if p is parameter))
        # Name the actual terminal layer, rather than assuming a Sequential index.
        if letter == 'A':
            changed_actor = {next(n for n,p in actor.named_parameters() if p is actor.actor.action[-1].bias)}
        for name,value in actor.state_dict().items():
            if name not in changed_actor and not torch.equal(value.cpu(),before_actor[name]):
                raise RuntimeError('Unexpected actor migration: '+name)
        for name,value in world.state_dict().items():
            if name not in changed_world and not torch.equal(value.cpu(),before_world[name]):
                raise RuntimeError('Unexpected world migration: '+name)
        if letter == 'A' and not torch.equal(actor.actor.action[-1].bias[:4].cpu(),
                                             before_actor[next(iter(changed_actor))][:4]):
            raise RuntimeError('Motor bias was changed during stop migration')
        for name in changed_world:
            if not torch.equal(world.state_dict()[name][1:].cpu(),before_world[name][1:]):
                raise RuntimeError('Nonreward world rows changed')
        verify_optimizer(actor,optimizer,actor_moments,
            removed=changed_actor if letter=='D' else (),
            slices={name:4 for name in changed_actor} if letter=='A' else {})
        verify_optimizer(world,world_optimizer,world_moments,slices={name:0 for name in changed_world})
        if actor_groups!=optimizer.state_dict()['param_groups'] or world_groups!=world_optimizer.state_dict()['param_groups']:
            raise RuntimeError('Optimizer parameter ownership or hyperparameters changed')
        output.mkdir(parents=True,exist_ok=True)
        meta.update(parent_checkpoint_sha256=digest(args.checkpoint), fork_id=key,
                    phase_id=child['phase_id'], phase_accepted_batches=0, pending=None,
                    task_sampler_state=None, world_pending=None, world_ranking_qualified=False,
                    stop_outcomes_qualified=False, asset_identity=None)
        meta.pop('last_report',None)
        meta['qwen_snapshot']=None
        save_bundle(checkpoint,actor,optimizer,world,world_optimizer,meta,child)
        write(output/'config.json',child)
        receipt = dict(schema='photo-goal-city-migration/v1',fork_id=key,phase=args.phase,
                       parent_checkpoint=str(Path(args.checkpoint).resolve()),
                       parent_checkpoint_sha256=digest(args.checkpoint),config_sha256=identity(child),
                       checkpoint_sha256=digest(checkpoint),changed_config=sorted(changes),
                       changed_actor=sorted(changed_actor),changed_world=sorted(changed_world),
                       optimizer_preservation_verified=True,
                       reference_pending_preserved=letter=='A',counts=meta['counts'])
        write(receipt_path,receipt)
        return receipt
