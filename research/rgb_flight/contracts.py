"""Runtime-only records. Simulator labels belong in a different process/store.

These types enforce a data boundary; they are not an OS/process sandbox.
Conventions: seconds on simulation clock, body forward/right/down, yaw deg/s.
"""
from dataclasses import dataclass
import hashlib
import math


def finite(values):
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite runtime quantity")


@dataclass(frozen=True, slots=True)
class Command:
    forward_mps: float
    right_mps: float
    down_mps: float
    yaw_dps: float

    def __post_init__(self):
        finite((self.forward_mps, self.right_mps, self.down_mps, self.yaw_dps))
        # The active curriculum chooses a lower limit (3, 4.5, or 6 m/s).
        # This structural record enforces the absolute aircraft envelope.
        if math.hypot(self.forward_mps, self.right_mps) > 6 or abs(self.down_mps) > 1 or abs(self.yaw_dps) > 45:
            raise ValueError("Command exceeds fixed flight envelope")


@dataclass(frozen=True, slots=True)
class CommandSample:
    sim_seconds: float
    command: Command

    def __post_init__(self):
        finite((self.sim_seconds,))
        if self.sim_seconds < 0 or not isinstance(self.command, Command):
            raise ValueError("Invalid command sample")


@dataclass(frozen=True, slots=True)
class Calibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    camera_to_body_rotation: tuple[float, ...] | None = None
    camera_origin_body_m: tuple[float, ...] | None = None

    def __post_init__(self):
        finite((self.fx, self.fy, self.cx, self.cy))
        if (self.width, self.height) != (640, 480) or min(self.fx, self.fy) <= 0:
            raise ValueError("Invalid calibrated RGB camera")
        if not 0 <= self.cx < self.width or not 0 <= self.cy < self.height:
            raise ValueError("Principal point outside configured image")
        if (self.camera_to_body_rotation is None) != (self.camera_origin_body_m is None):
            raise ValueError('Camera extrinsics must provide both rotation and translation')
        if self.camera_to_body_rotation is not None:
            object.__setattr__(self, 'camera_to_body_rotation', tuple(self.camera_to_body_rotation))
            object.__setattr__(self, 'camera_origin_body_m', tuple(self.camera_origin_body_m))
            if len(self.camera_to_body_rotation) != 9 or len(self.camera_origin_body_m) != 3:
                raise ValueError('Invalid camera extrinsic dimensions')
            finite((*self.camera_to_body_rotation, *self.camera_origin_body_m))
            rows = [self.camera_to_body_rotation[i:i+3] for i in (0,3,6)]
            if any(abs(sum(a*b for a,b in zip(rows[i], rows[j])) - int(i==j)) > 1e-5 for i in range(3) for j in range(3)):
                raise ValueError('Camera extrinsic rotation must be orthonormal')
            a,b,c=rows
            determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
            if abs(determinant-1)>1e-5:
                raise ValueError('Camera extrinsics require a proper rotation, not a reflection')


@dataclass(frozen=True, slots=True)
class GoalObservation:
    """The complete runtime destination description.

    View order is the fixed panorama order (0, 90, 180, 270 degrees).  No
    camera/world pose or destination coordinate is representable here.
    """
    episode_id: str
    rgb_views: tuple[bytes, bytes, bytes, bytes]
    calibration: Calibration
    captured_sim_seconds: tuple[float, float, float, float]

    def __post_init__(self):
        if not self.episode_id or not isinstance(self.calibration, Calibration):
            raise ValueError('Invalid goal observation identity/calibration')
        if not isinstance(self.rgb_views, tuple) or len(self.rgb_views) != 4:
            raise ValueError('A visual goal requires exactly four ordered RGB views')
        if any(not isinstance(view, bytes) or len(view) != 640 * 480 * 3 for view in self.rgb_views):
            raise ValueError('Goal views must be immutable uncompressed RGB24')
        if not isinstance(self.captured_sim_seconds, tuple) or len(self.captured_sim_seconds) != 4:
            raise ValueError('Every goal view needs a capture timestamp')
        finite(self.captured_sim_seconds)
        if any(value < 0 for value in self.captured_sim_seconds):
            raise ValueError('Invalid goal capture time')

    @property
    def content_sha256(self):
        digest = hashlib.sha256()
        for view in self.rgb_views:
            digest.update(view)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Non-privileged episode identity and visual task configuration."""
    episode_id: str
    split: str
    phase: int
    goal: GoalObservation
    maximum_horizontal_speed_mps: float

    def __post_init__(self):
        if not self.episode_id or self.goal.episode_id != self.episode_id:
            raise ValueError('Cross-episode goal record')
        if self.split not in ('train', 'validation', 'test') or self.phase not in (1, 2, 3):
            raise ValueError('Invalid episode split/phase')
        if self.maximum_horizontal_speed_mps not in (3., 4.5, 6.):
            raise ValueError('Episode speed is outside the declared curriculum')


@dataclass(frozen=True, slots=True)
class Observation:
    episode_id: str
    frame_id: int
    rgb: bytes
    calibration: Calibration
    sim_seconds: float
    received_monotonic_seconds: float
    command_history: tuple[CommandSample, ...]

    def __post_init__(self):
        finite((self.sim_seconds, self.received_monotonic_seconds))
        if not self.episode_id or self.frame_id < 0 or min(self.sim_seconds, self.received_monotonic_seconds) < 0:
            raise ValueError("Invalid observation identity/time")
        if not isinstance(self.calibration, Calibration) or not isinstance(self.rgb, bytes) or len(self.rgb) != 640 * 480 * 3:
            raise ValueError("Expected immutable, uncompressed 640x480 RGB bytes")
        if not isinstance(self.command_history, tuple) or not all(isinstance(x, CommandSample) for x in self.command_history):
            raise ValueError("Command history must be immutable typed samples")
        times = [x.sim_seconds for x in self.command_history]
        if times != sorted(times) or any(t > self.sim_seconds for t in times):
            raise ValueError("Noncausal command history")


@dataclass(frozen=True, slots=True)
class BeliefSnapshot:
    episode_id: str
    sim_seconds: float
    estimated_state: tuple[float, ...]
    uncertainty: tuple[float, ...]
    tracking_confidence: float
    memory_version: int
    memory_latest_observation_seconds: float
    feature_age_seconds: float
    map_status: str = "initializing"
    input_validity: tuple[bool, ...] = ()
    initialization_elapsed_seconds: float = 0.
    gauge_version: int = 0
    hazard_version: int = 0

    def __post_init__(self):
        if not isinstance(self.estimated_state, tuple) or not isinstance(self.uncertainty, tuple):
            raise ValueError("Belief state and uncertainty must be immutable tuples")
        finite((*self.estimated_state, *self.uncertainty, self.sim_seconds, self.tracking_confidence,
                self.memory_latest_observation_seconds, self.feature_age_seconds))
        if not self.episode_id or not 0 <= self.tracking_confidence <= 1 or self.memory_version < 0:
            raise ValueError("Invalid belief identity/confidence/version")
        if any(x < 0 for x in self.uncertainty) or self.feature_age_seconds < 0:
            raise ValueError("Invalid belief uncertainty/age")
        if min(self.sim_seconds, self.memory_latest_observation_seconds) < 0 or self.memory_latest_observation_seconds > self.sim_seconds:
            raise ValueError("Memory includes future observations")


@dataclass(frozen=True, slots=True)
class VisualTaskConfig:
    episode_id: str
    target_id: str | None
    grounded_kind: str
    intention: str
    goal_weight: float
    time_weight: float
    information_weight: float
    additional_caution: float
    confidence: float
    valid_until_sim_seconds: float
    deliberate_immediately: bool

    def __post_init__(self):
        weights = (self.goal_weight, self.time_weight, self.information_weight, self.additional_caution)
        finite((*weights, self.confidence, self.valid_until_sim_seconds))
        if not self.episode_id or self.valid_until_sim_seconds < 0:
            raise ValueError("Invalid task identity/time")
        if any(not .25 <= x <= 4 for x in weights):
            raise ValueError("Task multiplier outside permitted range")
        abstain=self.target_id is None and self.grounded_kind=='none' and self.confidence==0 and self.intention=='search'
        if not abstain and (self.grounded_kind not in ("goal_match", "observed_frontier") or not self.target_id):
            raise ValueError("Configuration requires an observed visual match or frontier ID")
        if self.intention not in ("approach", "inspect", "search", "stop") or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid task intention/confidence")
        if type(self.deliberate_immediately) is not bool:
            raise ValueError('Deliberation trigger must be boolean')

    def validate_grounding(self, episode_id, observed_ids, now):
        grounded=(not observed_ids) if self.target_id is None else self.target_id in observed_ids
        if episode_id != self.episode_id or not grounded or now > self.valid_until_sim_seconds:
            raise ValueError("Ungrounded, cross-episode or expired configuration")


@dataclass(frozen=True, slots=True)
class Plan:
    episode_id: str
    based_on_sim_seconds: float
    created_monotonic_seconds: float
    commands: tuple[Command, ...]
    predicted_primitive_costs: tuple[float, ...]
    valid_until_sim_seconds: float
    memory_version: int
    interval_seconds: float = .2
    gauge_version: int = 0
    hazard_version: int = 0

    def __post_init__(self):
        if not isinstance(self.predicted_primitive_costs, tuple) or not self.episode_id or self.memory_version < 0:
            raise ValueError("Invalid plan identity/version or mutable costs")
        finite((*self.predicted_primitive_costs, self.based_on_sim_seconds, self.created_monotonic_seconds,
                self.valid_until_sim_seconds, self.interval_seconds))
        if not isinstance(self.commands, tuple) or len(self.commands) != 20 or not all(isinstance(u, Command) for u in self.commands):
            raise ValueError("A plan needs 20 bounded command intervals")
        if self.interval_seconds != .2 or not self.based_on_sim_seconds < self.valid_until_sim_seconds <= self.based_on_sim_seconds + 4:
            raise ValueError("Invalid plan validity/horizon")


@dataclass(frozen=True, slots=True)
class Transition:
    """Runtime history only. Training labels join by episode/frame outside inference."""
    observation: Observation
    command: Command
    next_observation: Observation

    def __post_init__(self):
        if not isinstance(self.observation, Observation) or not isinstance(self.next_observation, Observation) or not isinstance(self.command, Command):
            raise ValueError("Transition requires typed observations and command")
        if self.observation.episode_id != self.next_observation.episode_id or self.next_observation.sim_seconds <= self.observation.sim_seconds:
            raise ValueError("Transition crosses episode or time boundary")
