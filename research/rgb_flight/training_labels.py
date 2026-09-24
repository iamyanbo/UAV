"""Privileged records for collection/training/evaluation only.

Runtime modules must never import this file.  Keeping destination coordinates,
simulator pose/depth and collision labels here makes accidental construction of
a privileged runtime Observation impossible.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path


def _finite(values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite privileged label')


@dataclass(frozen=True, slots=True)
class EvaluatorEpisodeLabels:
    episode_id: str
    split: str
    phase: int
    start_ned_m: tuple[float, float, float]
    start_yaw_degrees: float
    goal_ned_m: tuple[float, float, float]
    reference_path_ned_m: tuple[tuple[float, float, float], ...]
    goal_region_id: str

    def __post_init__(self):
        if not self.episode_id or self.split not in ('train', 'validation', 'test') or self.phase not in (1, 2, 3):
            raise ValueError('Invalid evaluator episode identity')
        if len(self.start_ned_m) != 3 or len(self.goal_ned_m) != 3 or len(self.reference_path_ned_m) < 2:
            raise ValueError('Invalid evaluator geometry')
        _finite((*self.start_ned_m, self.start_yaw_degrees, *self.goal_ned_m,
                 *(value for point in self.reference_path_ned_m for value in point)))


@dataclass(frozen=True, slots=True)
class TrainingFrameLabels:
    episode_id: str
    frame_id: int
    sim_seconds: float
    true_position_ned_m: tuple[float, float, float]
    true_velocity_ned_mps: tuple[float, float, float]
    true_quaternion_xyzw: tuple[float, float, float, float]
    goal_distance_m: float
    airsim_collision: bool
    geometry_collision: bool

    def __post_init__(self):
        if not self.episode_id or self.frame_id < 0 or self.sim_seconds < 0:
            raise ValueError('Invalid training label identity')
        _finite((self.sim_seconds, *self.true_position_ned_m, *self.true_velocity_ned_mps,
                 *self.true_quaternion_xyzw, self.goal_distance_m))
        if type(self.airsim_collision) is not bool or type(self.geometry_collision) is not bool:
            raise ValueError('Collision labels must be boolean')


def write_jsonl(path, rows):
    """Atomic label write under a privileged-only directory."""
    path = Path(path)
    if path.parent.name not in ('training_labels', 'evaluator_labels', 'engineering_only'):
        raise ValueError('Privileged labels require a clearly separated directory')
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
    temporary.replace(path)
