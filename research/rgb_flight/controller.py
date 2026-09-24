"""Asynchronous ownership and validity coordination; no command averaging."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

from contracts import Command


@dataclass(frozen=True)
class PlanContext:
    plan: object
    task_version: int
    initial_position: tuple
    expected_positions: tuple
    maximum_collision_probability: float
    initial_state: tuple
    expected_states: tuple


class Coordinator:
    def __init__(self, policy, planner, confidence_floor, state_tolerance_m, risk_ceiling,
                 velocity_tolerance_mps, rotation_tolerance_radians, maximum_log_scale_std):
        self.policy, self.planner = policy, planner
        self.confidence_floor = confidence_floor
        self.state_tolerance_m = state_tolerance_m
        self.risk_ceiling = risk_ceiling
        self.velocity_tolerance_mps = velocity_tolerance_mps
        self.rotation_tolerance_radians = rotation_tolerance_radians
        self.maximum_log_scale_std = maximum_log_scale_std
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='deliberation')
        self.pending = self.active = None
        self.last_request = -math.inf
        self.task_version = None

    def step(self, observation, belief, task, task_version):
        now = observation.sim_seconds
        if (belief.episode_id != observation.episode_id or belief.tracking_confidence < self.confidence_floor
                or len(belief.estimated_state) != 32 or belief.estimated_state[13] > self.maximum_log_scale_std
                or belief.sim_seconds > now or now - belief.sim_seconds > .25
                or task.episode_id != observation.episode_id or now > task.valid_until_sim_seconds):
            self.active = None
            return Command(0, 0, 0, 0), 'braking'
        fast = self.policy(observation, belief, task)
        if self.pending and self.pending.done():
            try:
                candidate = self.pending.result()
                self.active = candidate if self.valid(candidate, observation, belief, task_version) else None
            except (RuntimeError, ValueError):
                self.active = None
            finally:
                self.pending = None
        changed = task_version != self.task_version
        if self.pending is None and (changed or now - self.last_request >= .5):
            self.pending = self.executor.submit(self.planner, observation, belief, task, task_version, fast)
            self.last_request, self.task_version = now, task_version
        if self.active and self.valid(self.active, observation, belief, task_version):
            index = int((now - self.active.plan.based_on_sim_seconds) / .2)
            return self.active.plan.commands[index], 'predictive'
        self.active = None
        return fast, 'fast'

    def valid(self, context, observation, belief, task_version):
        plan = context.plan
        now = observation.sim_seconds
        if (plan.episode_id != observation.episode_id or context.task_version != task_version
                or plan.gauge_version != belief.gauge_version or plan.hazard_version != belief.hazard_version
                or not plan.based_on_sim_seconds <= now < plan.valid_until_sim_seconds
                or time.monotonic() - plan.created_monotonic_seconds > 1
                or context.maximum_collision_probability > self.risk_ceiling):
            return False
        elapsed = (now - plan.based_on_sim_seconds) / .2
        index = int(elapsed)
        if not 0 <= index < 20:
            return False
        before = context.initial_position if index == 0 else context.expected_positions[index - 1]
        after = context.expected_positions[index]
        fraction = elapsed - index
        expected = tuple(a + fraction * (b - a) for a, b in zip(before, after))
        if math.dist(expected, belief.estimated_state[:3]) > self.state_tolerance_m:
            return False
        if len(context.expected_states) != 20 or len(context.initial_state) != 32:
            return False
        before_state = context.initial_state if index == 0 else context.expected_states[index - 1]
        after_state = context.expected_states[index]
        expected_state = tuple(a + fraction * (b-a) for a,b in zip(before_state, after_state))
        if math.dist(expected_state[3:6], belief.estimated_state[3:6]) > self.velocity_tolerance_mps:
            return False
        # The two stored rotation columns must agree independently; this
        # rejects a nearby position reached with incompatible body heading.
        for start in (6, 9):
            a, b = expected_state[start:start+3], belief.estimated_state[start:start+3]
            denominator = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
            if denominator < 1e-6:
                return False
            cosine = max(-1., min(1., sum(x*y for x,y in zip(a,b))/denominator))
            if math.acos(cosine) > self.rotation_tolerance_radians:
                return False
        return True

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class HardSafetyFilter:
    """Independent 20 Hz brake gate whose penalties are not configurable."""
    def __init__(self, maximum_speed_mps, braking_acceleration_mps2, rgb_stale_seconds=.25,
                 maximum_log_scale_std=.35, collision_probability_ceiling=.1):
        if maximum_speed_mps not in (3.,4.5,6.) or braking_acceleration_mps2 <= 0:
            raise ValueError('Invalid safety calibration')
        self.maximum_speed = maximum_speed_mps
        self.braking_acceleration = braking_acceleration_mps2
        self.rgb_stale_seconds = rgb_stale_seconds
        self.maximum_log_scale_std = maximum_log_scale_std
        self.collision_probability_ceiling = collision_probability_ceiling

    def apply(self, command, observation_age_seconds, tracking_lost, log_scale_std,
              predicted_collision_probability, conservative_clearance_m, current_speed_mps,
              inference_delay_seconds):
        numbers=(observation_age_seconds,log_scale_std,predicted_collision_probability,
                 conservative_clearance_m,current_speed_mps,inference_delay_seconds)
        if any(not math.isfinite(x) for x in numbers) or any(x<0 for x in numbers):
            return Command(0,0,0,0),dict(overridden=True,reason='invalid_safety_estimate',stopping_distance_m=None)
        proposed_speed=math.sqrt(command.forward_mps**2+command.right_mps**2+command.down_mps**2)
        braking_speed=max(current_speed_mps,proposed_speed)
        stopping_distance = braking_speed**2/(2*self.braking_acceleration)+braking_speed*inference_delay_seconds
        unsafe = (observation_age_seconds > self.rgb_stale_seconds or tracking_lost
                  or log_scale_std > self.maximum_log_scale_std
                  or predicted_collision_probability > self.collision_probability_ceiling
                  or conservative_clearance_m <= stopping_distance)
        if unsafe:
            return Command(0,0,0,0), dict(overridden=True,reason='mandatory_brake',stopping_distance_m=stopping_distance)
        horizontal = math.hypot(command.forward_mps,command.right_mps)
        if horizontal > self.maximum_speed:
            factor = self.maximum_speed/horizontal
            command = Command(command.forward_mps*factor,command.right_mps*factor,command.down_mps,command.yaw_dps)
            return command, dict(overridden=True,reason='speed_limit',stopping_distance_m=stopping_distance)
        return command, dict(overridden=False,reason=None,stopping_distance_m=stopping_distance)
