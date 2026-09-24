"""Shared simulator startup/recovery contract, driven only by causal belief."""
import math
from contracts import Command

VERSION = 'bounded-exploration-dispatched/v3'


class StartupState:
    def __init__(self):
        self.state = 'initializing'
        self.started_ns = self.ready_since_ns = self.recovery_ns = None
        self.handover_ns = None
        self.reason = None

    def update(self, now_ns, supported_map, tracking_valid):
        if self.started_ns is None: self.started_ns = now_ns
        if self.state == 'terminated': return self.state
        ready = supported_map and tracking_valid
        if self.state == 'mapped' and not ready:
            self.state = 'recovering'; self.recovery_ns = now_ns; self.ready_since_ns = None
        if self.state in ('initializing', 'recovering'):
            if ready:
                if self.ready_since_ns is None: self.ready_since_ns = now_ns
                if now_ns - self.ready_since_ns >= 1_000_000_000:
                    self.state = 'mapped'; self.handover_ns = self.handover_ns or now_ns
            else: self.ready_since_ns = None
            origin = self.started_ns if self.state == 'initializing' else self.recovery_ns
            limit = 30 if self.state == 'initializing' else 10
            if self.state != 'mapped' and now_ns - origin >= limit * 1_000_000_000:
                self.reason = 'initialization_timeout' if self.state == 'initializing' else 'tracking_recovery_timeout'
                self.state = 'terminated'
        return self.state

    def elapsed(self, now_ns):
        return 0. if self.started_ns is None else (now_ns-self.started_ns)/1e9

    def filter(self, command, age):
        if not math.isfinite(age) or age < 0 or age > .25:
            return Command(0,0,0,0), 'stale_rgb'
        if self.state == 'terminated': return Command(0,0,0,0), self.reason
        if self.state == 'recovering': return Command(0,0,0,15), 'tracking_recovery_scan'
        horizontal = math.hypot(command.forward_mps, command.right_mps)
        factor = min(1., .5/max(horizontal, 1e-9))
        return Command(command.forward_mps*factor, command.right_mps*factor,
                       max(-.2,min(.2,command.down_mps)),max(-15.,min(15.,command.yaw_dps))), 'initialization_limits'


def demonstration(value):
    """No destination coordinates. Targets come from observed memory only."""
    state = value['map_status']; elapsed = value['initialization_elapsed_seconds']
    if state == 'initializing':
        # Translating turns, vertical excitation and short braking examples.
        phase = int(elapsed // 4) % 6
        commands = ((.4,0,0,12),(.35,.1,-.15,-12),(.4,0,.15,10),
                    (0,0,0,0),(.3,-.15,0,-12),(.4,0,0,12))
        return list(commands[phase]), False, 'initialization_brake' if phase == 3 else 'initialization_translation'
    if state == 'recovering': return [0.,0.,0.,15.], False, 'tracking_recovery_scan'
    if state == 'terminated': return [0.,0.,0.,0.], False, 'episode_terminated'
    if value['goal_probability'] >= value.get('goal_match_threshold', .95):
        return [0.,0.,0.,0.], True, 'visually_supported_stop'
    if value['target_available']:
        import torch
        position=value['state'][:3]; rotation=torch.stack((value['state'][6:9],value['state'][9:12],
            torch.linalg.cross(value['state'][6:9],value['state'][9:12])),1)
        delta=rotation.T@(value['target_context'][256:259]-position)
        yaw=math.degrees(math.atan2(float(delta[1]),float(delta[0])))
        return [max(0.,min(.5,float(delta[0]))),max(-.3,min(.3,float(delta[1]))),
                max(-.2,min(.2,float(delta[2]))),max(-15.,min(15.,yaw))],False,'observed_target_motion'
    return [0.,0.,0.,15.],False,'no_observed_target_scan'
