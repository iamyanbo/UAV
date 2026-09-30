"""Atomic full-system checkpoints; legacy optimizer migration is explicit."""
import random
from pathlib import Path
import numpy as np
import torch
from .common import digest, write
from .mission_contracts import BUNDLE_SCHEMA, identity
from .ppo_core import ActorCritic
from .mission_policy import CityActorCritic, owned_optimizer
from .mission_world import CityWorld


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def encoder_identity(actor):
    import hashlib
    hasher = hashlib.sha256()
    for name, value in actor.encoder.state_dict().items():
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def migrate_optimizer(actor, optimizer, legacy, backbone):
    """Retain historical Adam moments, padding new mission-input columns with zero."""
    if not legacy:
        return dict(restored=False)
    reference = ActorCritic(backbone)
    names = [name for name, _ in reference.named_parameters()]
    ids = [ident for group in legacy['param_groups'] for ident in group['params']]
    if len(ids) != len(names):
        raise ValueError('Legacy optimizer parameter ownership cannot be identified')
    old = dict(zip(names, ids))
    restored = []
    for name, parameter in actor.named_parameters():
        if not parameter.requires_grad or name not in old or old[name] not in legacy['state']:
            continue
        state = {}
        for key, value in legacy['state'][old[name]].items():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                if value.ndim == 2 and value.shape[1] == 576 and parameter.shape[1] == 640:
                    value = torch.nn.functional.pad(value, (0, 64))
                else:
                    raise ValueError('Unsupported legacy optimizer tensor: '+name)
            state[key] = value.to(parameter.device) if torch.is_tensor(value) else value
        optimizer.state[parameter] = state
        restored.append(name)
    return dict(restored=True, tensor_owners=restored)


def load_components(path, backbone, cfg, device='cuda'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    actor = CityActorCritic(backbone, cfg['stop_prior']).to(device).eval()
    optimizer = owned_optimizer(actor, cfg['learning_rate'])
    world = CityWorld().to(device).eval()
    world_optimizer = torch.optim.AdamW(world.parameters(), lr=cfg['world']['learning_rate'])
    if saved.get('schema') == BUNDLE_SCHEMA:
        if saved['config_sha256'] != identity(cfg) or saved['backbone_sha256'] != digest(backbone):
            raise ValueError('Resume configuration/backbone differs from serving bundle')
        actor.load_state_dict(saved['actor'], strict=True)
        optimizer.load_state_dict(saved['actor_optimizer'])
        world.load_state_dict(saved['world'], strict=True)
        world_optimizer.load_state_dict(saved['world_optimizer'])
        if saved['encoder_basis_sha256'] != encoder_identity(actor):
            raise ValueError('Checkpoint visual basis changed')
        # Never reset learned stop logits while resuming.
        return actor, optimizer, world, world_optimizer, saved
    if 'model' not in saved or 'counts' not in saved:
        raise ValueError('A native actor checkpoint or full serving bundle is required')
    actor.migrate(saved['model'], initialize_stop=True, stop_prior=cfg['stop_prior'])
    migration = migrate_optimizer(actor, optimizer, saved.get('optimizer'), backbone)
    # The deliberately reset stop-output row must not inherit its old momentum.
    for parameter in (actor.actor.action[-1].weight, actor.actor.action[-1].bias):
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            state = optimizer.state.get(parameter, {}).get(key)
            if state is not None:
                state[4].zero_()
    metadata = dict(schema=BUNDLE_SCHEMA, config_sha256=identity(cfg),
                    backbone_sha256=digest(backbone), encoder_basis_sha256=encoder_identity(actor),
                    source_checkpoint_sha256=digest(path), optimizer_migration=migration,
                    historical_counts=saved['counts'], counts=dict(city_batches=0, accepted_transitions=0, world_updates=0),
                    qwen_adapter=None, world_ranking_qualified=False, pending=None)
    metadata['rng'] = dict(python=saved.get('python_rng', random.getstate()),
                           numpy=saved.get('numpy_rng', np.random.get_state()),
                           torch=saved.get('torch_rng', torch.get_rng_state()),
                           cuda=saved.get('cuda_rng'))
    return actor, optimizer, world, world_optimizer, metadata


def restore_rng(saved):
    state = saved['rng']
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG topology changed')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_bundle(path, actor, optimizer, world, world_optimizer, metadata, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata['encoder_basis_sha256'] != encoder_identity(actor):
        raise ValueError('Frozen encoder changed before publication')
    saved = dict(metadata, schema=BUNDLE_SCHEMA, config_sha256=identity(cfg),
                 actor=cpu_copy(actor.state_dict()), actor_optimizer=cpu_copy(optimizer.state_dict()),
                 world=cpu_copy(world.state_dict()), world_optimizer=cpu_copy(world_optimizer.state_dict()),
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    pending = path.with_suffix('.pending')
    from .mission_space import reserve_write
    import os
    def tensor_bytes(value):
        if torch.is_tensor(value): return value.numel()*value.element_size()
        if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(list,tuple)): return sum(tensor_bytes(v) for v in value)
        return 0
    reserve_write(path,int(tensor_bytes(saved)*1.1)+32*2**20,os.environ.get('UAV_SHUTDOWN_WRITES')=='1')
    torch.save(saved, pending)
    with pending.open('rb') as stream:os.fsync(stream.fileno())
    pending.replace(path)
    if os.name!='nt':
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    write(path.with_suffix('.json'), dict(schema=BUNDLE_SCHEMA, checkpoint_sha256=digest(path),
          counts=saved['counts'], encoder_basis_sha256=saved['encoder_basis_sha256'],
          phase_id=saved.get('phase_id'),phase_accepted_batches=saved.get('phase_accepted_batches'),
          world_work_remaining=(saved.get('world_pending') or {}).get('remaining',0),
          asset_identity=saved.get('asset_identity'),
          pending=(dict(batch_id=saved['pending']['batch_id'], rows=len(saved['pending']['rows']),
                        policy_sha256=saved['pending']['policy_sha256']) if saved['pending'] else None),
          world_ranking_qualified=saved['world_ranking_qualified'],
          qwen_adapter=saved.get('qwen_adapter'), no_cross_module_gradients=True))
    return digest(path)
