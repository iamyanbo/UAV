"""Runtime contracts: map coordinates are not aircraft or goal coordinates."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Hypothesis:
    position: tuple
    yaw: float
    probability: float
    sigma_m: float
    tile_id: int

    def __post_init__(self):
        if len(self.position) != 3 or not all(math.isfinite(v) for v in (*self.position, self.yaw, self.probability, self.sigma_m)):
            raise ValueError('Invalid pose hypothesis')
        if not 0 <= self.probability <= 1 or self.sigma_m <= 0:
            raise ValueError('Invalid pose confidence')


@dataclass(frozen=True)
class LocalizationBelief:
    hypotheses: tuple
    observed_s: float
    alignment_version: int
    aligned: bool

    @property
    def best(self):
        return self.hypotheses[0] if self.hypotheses else None


@dataclass(frozen=True)
class GoalBelief:
    candidates: tuple
    unresolved: bool
    observed_s: float


@dataclass(frozen=True)
class NavigationTask:
    episode_id: str
    scene_id: str
    map_sha256: str
    goal_sha256: str
    timeout_s: float


@dataclass(frozen=True)
class RoutePlan:
    waypoints: tuple
    alignment_version: int
    created_s: float
    target_id: str
    estimated_seconds: float
    family: str = "direct"


