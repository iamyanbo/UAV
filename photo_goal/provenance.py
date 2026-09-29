"""Content identities bind learned assessment/calibration to its actual actor."""
import hashlib

ACTOR_PREFIXES=('encoder.','goal.','arrival.','policy.','subgoal_encoder.','reference_geometry.')


def state_identity(state,prefixes=None):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if prefixes is not None and not name.startswith(prefixes):continue
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def identities(state):
    return dict(actor=state_identity(state,ACTOR_PREFIXES),world=state_identity(state,('world.',)),
                perception=state_identity(state,('encoder.','goal.','arrival.','camera_projection.','map_projection.','registration.')))


def verify_release(saved,world=False,perception_only=False):
    current=identities(saved['model']);calibration=saved.get('calibration') or {}
    binding=calibration.get('model_identities',{})
    required=['perception'] if perception_only else ['perception','actor']
    if any(binding.get(key)!=current[key] for key in required):raise ValueError('Calibration stale for current model')
    if world:
        if saved.get('world_actor_identity')!=current['actor']:raise ValueError('Reassess world model after actor changes')
        if binding.get('world')!=current['world'] or 'score_scales' not in calibration:raise ValueError('World score calibration stale')
