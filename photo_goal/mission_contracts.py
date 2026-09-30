"""Strict RGB runtime boundary and the versioned city-training configuration."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
from .common import read

RUN_SCHEMA = 'photo-goal-city-run/v1'
BUNDLE_SCHEMA = 'photo-goal-full-bundle/v1'
WORLD_SCHEMA = 'photo-goal-city-world/v1'
CONTEXT_WIDTH = 16


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def city_config(path=None):
    """Leave the historical endpoint/campaign configurations unchanged."""
    base = read(Path(__file__).with_name('ppo_endpoint.json'))
    new = read(path or Path(__file__).with_name('city_campaign.json'))
    known=set(base)|set(read(Path(__file__).with_name('city_campaign.json')))|{
        'experiment','mode2_enabled','endpoint_pilot','phase_id','stop_class_balance','stop_class_weight_cap',
        'stop_reference_step_s','control_step_s','motor_control','task_sampling','world_updates_per_batch',
        'task_start_weights','max_route_length_m','task_height_headroom_m','task_roi_padding_m',
        'geometry_tile_size_m','geometry_resolution_m','route_cache_max_bytes','route_cache_max_entries',
        'extra_frame_queue_capacity','intermediate_deadline_s','support_deadline_s',
        'potential_normalization','potential_distance_scale_m','failure_remaining_time_charge'}
    if set(new)-known:raise ValueError('Unknown city config fields: '+str(sorted(set(new)-known)))
    if new.get('schema') != RUN_SCHEMA:
        raise ValueError('Expected the city full-run configuration')
    base.update(new)
    if base['profile'] != 'full' or base['rollout_steps'] != 8192 or not base['fixed_batch']:
        raise ValueError('City training requires full fixed 8192-row PPO batches')
    if base['max_start_goal_distance_m'] != 300:
        raise ValueError('This study is scoped to 300 m')
    bands, weights, deadlines = (base[k] for k in
                                ('distance_bands_m', 'distance_band_weights', 'mission_deadlines_s'))
    if (len(bands) != len(weights) or len(weights) != len(deadlines) or
            not math.isclose(sum(weights), 1.) or any(w <= 0 for w in weights) or
            any(not 0 < lo < hi <= 300 for lo, hi in bands) or
            any(not 0 < t <= base['episode_s'] for t in deadlines)):
        raise ValueError('Invalid city task bands/weights/deadlines')
    if not 0 < base['stop_prior'] < 1 or not 0 <= base['support_fraction'] < 1:
        raise ValueError('Invalid stop initialization or support mixture')
    if base['gradient_routing'] != dict(ppo='actor-critic', world='world-only',
                                       qwen='lora-only', cross_module_backpropagation=False):
        raise ValueError('Cross-module gradients are not admitted')
    for key in ('gamma_time_constant_s', 'gae_time_constant_s'):
        if not math.isfinite(base[key]) or base[key] <= 0:
            raise ValueError('Invalid duration discount')
    base['experiment'] = 'city-photo-goal-separate-gradients'
    base['mode2_enabled'] = True
    base['endpoint_pilot'] = False
    if 'phase_id' in base:
        if base['phase_id'] not in ('city-repair-A','city-repair-B','city-repair-C','city-repair-D'):
            raise ValueError('Unknown repair phase')
        if base.get('stop_reference_step_s') != .05 or base.get('control_step_s') != .05:
            raise ValueError('The reference hazard/control clock must remain 50 ms')
        if base.get('stop_class_balance') != 'rollout' or base.get('stop_class_weight_cap') != 20:
            raise ValueError('Invalid repair stop supervision')
        if base.get('task_sampling') == 'quota_v2' and base.get('task_start_weights') != dict(regular=.5,intermediate=.3,support=.2):
            raise ValueError('Invalid v2 mission-start mixture')
        if base.get('potential_normalization') == 'route_metres' and base.get('task_sampling') != 'quota_v2':
            raise ValueError('Route reward requires a route-validated v2 catalog')
        letter=base['phase_id'][-1]
        expected=dict(step_s=.05 if letter=='A' else .15,
                      motor_control='collector_limited' if letter=='A' else 'dispatcher_target',
                      task_sampling='legacy' if letter in ('A','B') else 'quota_v2',
                      potential_normalization='route_metres' if letter=='D' else 'initial_distance',
                      time_cost_per_mission=.2 if letter=='D' else 2)
        if any(base.get(k)!=v for k,v in expected.items()) or base['stop_prior']!=.00025:
            raise ValueError('Repair configuration combines changes from different phases')
        if bool(base.get('failure_remaining_time_charge',False))!=(letter=='D'):
            raise ValueError('Failure-time charge is exclusive to reward phase D')
        if base.get('world_updates_per_batch')!=819:raise ValueError('Repair requires 819 independent world updates per batch')
    return base


def task_band(distance, cfg):
    for i, (lo, hi) in enumerate(cfg['distance_bands_m']):
        if lo <= distance <= hi:
            return i
    raise ValueError('Task outside qualified city bands')


@dataclass(frozen=True)
class RuntimeObservation:
    rgb: bytes
    rgb_path: str
    frame: int
    sim_s: float
    source_wall: float
    preceding_command: tuple
    mission_id: str
    remaining_s: float
    deadline_s: float
    calibration_id: str

    @classmethod
    def from_environment(cls, obs, mission_id, deadline_s, calibration_id):
        # Copy named fields only. Rich environment state never crosses inference.
        value = cls(rgb=bytes(obs['rgb']), rgb_path=str(obs['rgb_path']),
                    frame=int(obs['frame']), sim_s=float(obs['sim_s']),
                    source_wall=float(obs['source_wall']),
                    preceding_command=tuple(float(x) for x in obs['preceding_command']),
                    mission_id=str(mission_id),
                    remaining_s=max(0., deadline_s-float(obs['elapsed_s'])),
                    deadline_s=float(deadline_s), calibration_id=str(calibration_id))
        value.validate()
        return value

    def validate(self):
        if len(self.rgb) != 640*480*3 or len(self.preceding_command) != 4:
            raise ValueError('Invalid RGB/command payload')
        values = (self.sim_s, self.source_wall, self.remaining_s, self.deadline_s,
                  *self.preceding_command)
        if (not all(math.isfinite(x) for x in values) or self.deadline_s <= 0 or
                not 0 <= self.remaining_s <= self.deadline_s):
            raise ValueError('Invalid runtime time/commands')
        if not self.mission_id or not self.calibration_id:
            raise ValueError('Missing mission/calibration identity')
        return self

    def as_scheduler_observation(self):
        return dict(rgb=self.rgb, rgb_path=self.rgb_path, frame=self.frame,
                    sim_s=self.sim_s, source_wall=self.source_wall,
                    preceding_command=list(self.preceding_command),
                    remaining_s=self.remaining_s, deadline_s=self.deadline_s,
                    mission_id=self.mission_id, calibration_id=self.calibration_id)


@dataclass(frozen=True)
class StrategicTarget:
    id: str
    intention: str
    altitude: str
    source: str
    reference: str | None
    confidence: float
    generation_frame: int
    generation_s: float
    expires_s: float

    @classmethod
    def from_proposal(cls, row, frame, stamp, references):
        from .contracts import INTENTIONS, ALTITUDES
        allowed = {'intention', 'altitude', 'target_source', 'target_reference',
                   'confidence', 'horizon_s'}
        if set(row) != allowed:
            raise ValueError('Proposals cannot include motor commands/coordinates/stop')
        source, ref = row['target_source'], row['target_reference']
        if (row['intention'] not in INTENTIONS or row['altitude'] not in ALTITUDES or
                source not in ('none', 'map', 'keyframe', 'image_region') or
                not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1):
            raise ValueError('Invalid strategic proposal')
        if source == 'none':
            if ref is not None:
                raise ValueError('Absent source must have no reference')
        elif ref not in references or references[ref]['source'] != source:
            raise ValueError('Reference was not supplied with this source')
        horizon = float(row['horizon_s'])
        maximum = 5. if source == 'image_region' else 30.
        if not math.isfinite(horizon) or not 0 < horizon <= maximum:
            raise ValueError('Invalid strategic horizon')
        key = identity(dict(row=row, frame=frame, stamp=stamp))[:24]
        return cls(key, row['intention'], row['altitude'], source, ref,
                   float(row['confidence']), frame, stamp, stamp+horizon)

    def valid(self, stamp):
        return self.generation_s <= stamp < self.expires_s

    def vector(self, stamp):
        from .contracts import INTENTIONS, ALTITUDES, SOURCES
        if not self.valid(stamp):
            from .contracts import Subgoal, SpatialSnapshot
            return Subgoal().vector(stamp, SpatialSnapshot())
        return ([float(self.intention == x) for x in INTENTIONS] +
                [float(self.altitude == x) for x in ALTITUDES] +
                [float(self.source == x) for x in SOURCES] +
                [self.confidence, min(1., (self.expires_s-stamp)/5.)])
