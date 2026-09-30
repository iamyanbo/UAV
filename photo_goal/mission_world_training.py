"""Train the physical visual model from canonical, training-only flight shards."""
from pathlib import Path
import random
from .common import FlightLock
from .mission_contracts import city_config
from .mission_checkpoint import load_components, save_bundle, restore_rng
from .mission_data import load_shard
from .mission_world_data import from_shard
from .mission_world import world_update
from .mission_policy import require_disjoint
from .mission_resources import Resources, RunWindow
from .ppo_budget import Budget


def run(args):
    cfg, root = city_config(args.config), Path(args.root).resolve()
    resources = Resources(root, cfg, args.device)
    resources.check()
    actor, optimizer, world, world_optimizer, meta = load_components(args.checkpoint, args.backbone, cfg, args.device)
    if meta.get('pending'):
        raise ValueError('Finish or explicitly archive the pending on-policy batch before fitting a serving world')
    require_disjoint(optimizer, world_optimizer)
    restore_rng(meta)
    paths = sorted(Path(args.replay).glob('records/*.json'))
    if not paths:
        raise ValueError('No canonical recorded flight shards')
    window = RunWindow(args.hours)
    completed, budget = 0, None
    cached = None
    with FlightLock(root, 'city-world'):
        try:
            budget = Budget(root/'campaign', cfg)
            while completed < args.updates and window.admits(60):
                resources.check()
                if meta['counts']['world_updates'] >= cfg['world']['updates']:
                    break
                # One shard in RAM; validate bytes once per shard switch.
                if cached is None or completed % 100 == 0:
                    data, labels, source = load_shard(random.choice(paths))
                    cached = (data['transitions'], labels, source)
                records, labels, source = cached
                if not records:
                    raise ValueError('Empty replay shard')
                ids = random.sample(range(len(records)), min(cfg['world']['batch_size'], len(records)))
                batch = from_shard(actor, [records[i] for i in ids], [labels[i] for i in ids], source, args.device)
                budget.reserve_updates('world', 1, cfg['world']['updates'])
                report = world_update(world, world_optimizer, batch, cfg['world']['auxiliary_weight'])
                meta['counts']['world_updates'] += 1
                completed += 1
                meta['last_world_report'] = report
                # Training loss does not qualify world ranking or stop outcomes.
                meta['world_ranking_qualified'] = False
                meta['stop_outcomes_qualified'] = False
                if completed % 100 == 0:
                    save_bundle(args.output, actor, optimizer, world, world_optimizer, meta, cfg)
        finally:
            save_bundle(args.output, actor, optimizer, world, world_optimizer, meta, cfg)
            if budget:
                budget.close()
