"""The three shared, versioned records for collection, learning and inference.

Times use the broker's simulator-second clock, except host publication_wall.
Map references are hypotheses; only observed geometry has local support.
"""
from dataclasses import dataclass, asdict, replace
import math

INTENTIONS = ('goal', 'inspect', 'approach', 'search', 'recover', 'hold')
ALTITUDES = ('maintain', 'gain', 'lose')
SOURCES = ('none', 'keyframe', 'geometry', 'map')


@dataclass(frozen=True)
class SpatialSnapshot:
    schema: str = 'spatial-snapshot/v1'
    observed_s: float = 0.
    publication_wall: float = 0.
    revision: int = 0
    tracking: bool = False
    poses: tuple = ()                 # estimated camera-to-map transforms
    scale_status: str = 'unknown'
    meters_per_unit: float | None = None
    scale_relative_sigma: float | None = None
    keyframes: tuple = ()             # (id, source_frame, observation_s)
    geometry: tuple = ()              # (id, x,y,z, support_count); map units
    map_references: tuple = ()        # coarse hypotheses, never clearance
    map_hypotheses: tuple = ()        # (id, coarse height, quantization); not geometry

    def references(self):
        return {'none': (), 'keyframe': tuple(r[0] for r in self.keyframes),
                'geometry': tuple(r[0] for r in self.geometry if r[4] >= 2),
                'map': self.map_references}

    def validate(self):
        if self.schema != 'spatial-snapshot/v1' or self.scale_status not in ('unknown', 'metric', 'lost'):
            raise ValueError('Incompatible spatial snapshot')
        if len(self.keyframes) > 3 or len(self.geometry) > 256 or len(self.map_references)>8:
            raise ValueError('Unbounded spatial snapshot')
        if self.scale_status == 'metric' and not (
                self.meters_per_unit is not None and self.meters_per_unit > 0 and
                self.scale_relative_sigma is not None and math.isfinite(self.scale_relative_sigma)):
            raise ValueError('Metric scale needs uncertainty')
        return self

    def vector(self, now):
        return [float(self.tracking),float(self.scale_status=='metric'),
                min(1.,self.scale_relative_sigma or 1.),len(self.geometry)/256,
                min(5.,max(0.,now-self.observed_s))/5,len(self.keyframes)/3,
                math.log1p(self.meters_per_unit or 0.),float(self.scale_status=='lost')]


@dataclass(frozen=True)
class Subgoal:
    schema: str = 'subgoal/v1'
    intention: str = 'goal'
    target_reference: str | None = None
    target_source: str = 'none'
    altitude: str = 'maintain'
    confidence: float = 0.
    source_observation: int = -1
    source_s: float = 0.
    map_revision: int = 0
    expires_s: float = 0.

    def valid(self, now, spatial):
        if (self.schema != 'subgoal/v1' or self.intention not in INTENTIONS or
                self.altitude not in ALTITUDES or self.target_source not in SOURCES or
                not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1 or
                not 0 <= now-self.source_s <= 5 or not now < self.expires_s <= self.source_s+5 or
                self.map_revision != spatial.revision):
            return False
        if self.target_source in ('geometry','keyframe') and not spatial.tracking:
            return False
        return ((self.target_reference is None and self.target_source == 'none') or
                self.target_reference in spatial.references()[self.target_source])

    def vector(self, now, spatial):
        """No ID hashing or coordinate generation. Same encoding on every path."""
        valid = self.valid(now, spatial)
        intention, altitude, source = ((self.intention, self.altitude, self.target_source)
                                      if valid else ('goal', 'maintain', 'none'))
        return ([float(intention == x) for x in INTENTIONS] +
                [float(altitude == x) for x in ALTITUDES] +
                [float(source == x) for x in SOURCES] +
                [self.confidence if valid else 0., max(0., self.expires_s-now)/5 if valid else 0.])

    @classmethod
    def proposal(cls, row, frame, now, spatial):
        allowed = {'intention', 'target_reference', 'target_source', 'altitude', 'confidence', 'horizon_s'}
        if set(row) != allowed:
            raise ValueError('Subgoals must use the structured contract, never vehicle commands')
        horizon = float(row['horizon_s'])
        if not math.isfinite(horizon) or not 0 < horizon <= 5:
            raise ValueError('Subgoal horizon outside (0,5]')
        goal = cls(intention=row['intention'], target_reference=row['target_reference'],
                   target_source=row['target_source'], altitude=row['altitude'],
                   confidence=float(row['confidence']), source_observation=frame,
                   source_s=now, map_revision=spatial.revision, expires_s=now+horizon)
        if not goal.valid(now, spatial):
            raise ValueError('Unsupported subgoal')
        return goal


@dataclass(frozen=True)
class ObservationContext:
    schema: str
    frame_ids: tuple
    timestamps: tuple
    features: object                 # B,4,64,256; padding is masked
    commands: object                 # B,4,4; preceding dispatched commands
    valid: object                    # B,4; no duplicated history
    goal_grid: object                # full spatial grid, cached for mission
    goal_context: object             # B,256
    spatial: SpatialSnapshot
    source_wall: float

    def validate(self):
        if self.schema != 'observation-context/v1' or not 1 <= len(self.frame_ids) <= 4:
            raise ValueError('Incompatible observation context')
        if len(set(self.frame_ids)) != len(self.frame_ids) or any(b <= a for a,b in zip(self.timestamps,self.timestamps[1:])):
            raise ValueError('Repeated or unordered observations')
        if len(self.timestamps)!=len(self.frame_ids) or not all(math.isfinite(t) for t in self.timestamps):
            raise ValueError('Invalid observation timestamps')
        if tuple(self.features.shape[1:]) != (4,64,256) or tuple(self.commands.shape[1:]) != (4,4):
            raise ValueError('Invalid bounded history')
        self.spatial.validate()
        return self


def snapshot_from_dict(value):
    value = dict(value)
    for name in ('poses', 'keyframes', 'geometry', 'map_references', 'map_hypotheses'):
        value[name] = tuple(tuple(r) if isinstance(r,list) else r for r in value.get(name,()))
    return SpatialSnapshot(**value).validate()


def context_record(context, subgoal):
    """Features are derived from referenced RGB, never serialized into JSON logs."""
    return dict(schema=context.schema, frame_ids=context.frame_ids, timestamps=context.timestamps,
                preceding_commands=context.commands[0,-len(context.frame_ids):].detach().cpu().tolist(),
                spatial=asdict(context.spatial), subgoal=asdict(subgoal) if subgoal else None)
