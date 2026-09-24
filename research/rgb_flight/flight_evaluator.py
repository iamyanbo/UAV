"""Privileged continuous-flight evaluator; never import from inference code."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EvaluationState:
    sim_seconds: float
    position: tuple
    velocity: tuple
    collided: bool
    geometry_collided: bool = False


class FlightEvaluator:
    def __init__(self, goal, boundary, start_seconds, reference_length):
        if not 20 <= reference_length <= 300:
            raise ValueError('Reference path outside visual-goal phase bounds')
        self.goal, self.boundary = goal, boundary
        self.start = self.previous_time = start_seconds
        self.reference_length = reference_length
        self.previous_position = None
        self.distance = 0.
        self.stopped_since = None
        self.result = None

    def update(self, state, stop_requested, tracking_lost=False):
        if self.result is not None:
            return self.result
        values = (*state.position, *state.velocity, state.sim_seconds)
        if len(state.position) != 3 or len(state.velocity) != 3 or not all(math.isfinite(x) for x in values):
            raise ValueError('Invalid evaluator sample')
        if state.sim_seconds < self.previous_time:
            raise ValueError('Evaluator time regressed')
        if self.previous_position is not None:
            self.distance += math.dist(self.previous_position, state.position)
        self.previous_position, self.previous_time = state.position, state.sim_seconds
        elapsed = state.sim_seconds - self.start
        in_goal = math.dist(state.position[:2], self.goal[:2]) <= 3 and abs(state.position[2] - self.goal[2]) <= 2
        speed = math.sqrt(sum(v*v for v in state.velocity))
        reason = None
        if state.collided or state.geometry_collided:
            reason = 'geometry_collision' if state.geometry_collided and not state.collided else 'collision'
        elif any(not self.boundary[0][i] <= state.position[i] <= self.boundary[1][i] for i in range(3)):
            reason = 'boundary_exit'
        elif tracking_lost:
            reason = 'tracking_loss'
        elif elapsed >= 180:
            reason = 'timeout'
        elif stop_requested and in_goal and speed < .5:
            if self.stopped_since is None:
                self.stopped_since = state.sim_seconds
            if state.sim_seconds - self.stopped_since >= 1:
                reason = 'success'
        else:
            self.stopped_since = None
        if reason:
            success = reason == 'success'
            # Boundary/tracking failures are task failures, recorded separately;
            # use the fixed noncollision failure reward, never invented success.
            outcome = 100 if success else -100 if reason in ('collision', 'geometry_collision') else -25
            horizontal_error = math.dist(state.position[:2], self.goal[:2])
            vertical_error = abs(state.position[2] - self.goal[2])
            self.result = dict(termination=reason, success=success, elapsed_sim_seconds=elapsed,
                               traveled_distance_m=self.distance, final_navigation_error_m=math.dist(state.position,self.goal),
                               final_horizontal_error_m=horizontal_error, final_vertical_error_m=vertical_error,
                               path_efficiency=float(success)*self.reference_length/max(self.reference_length,self.distance),
                               fixed_outcome_reward=outcome-10*min(elapsed/180,1),
                               stop_requested=bool(stop_requested), stopped_speed_mps=speed,
                               airsim_collision=bool(state.collided), geometry_collision=bool(state.geometry_collided))
        return self.result
