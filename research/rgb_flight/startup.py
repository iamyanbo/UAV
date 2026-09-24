"""Shared simulator startup/recovery contract, driven only by causal belief."""
import math
from contracts import Command

VERSION = 'local-depth-exploration-dispatched/v4'


class StartupState:
    def __init__(self):
        self.state = 'initializing'
        self.started_ns = self.ready_since_ns = self.recovery_ns = None
        self.handover_ns = None
        self.reason = None

    def update(self, now_ns, supported_map, tracking_valid, local_ready=False):
        if self.started_ns is None: self.started_ns = now_ns
        if self.state == 'terminated': return self.state
        mapped = supported_map and tracking_valid
        ready = mapped or local_ready
        if self.state in ('mapped','local_navigation') and not ready:
            self.state = 'recovering'; self.recovery_ns = now_ns; self.ready_since_ns = None
        if self.state in ('mapped','local_navigation') and ready:
            self.state='mapped' if mapped else 'local_navigation'
        if self.state in ('initializing', 'recovering'):
            if ready:
                if self.ready_since_ns is None: self.ready_since_ns = now_ns
                if now_ns - self.ready_since_ns >= 1_000_000_000:
                    self.state = 'mapped' if mapped else 'local_navigation'; self.handover_ns = self.handover_ns or now_ns
            else: self.ready_since_ns = None
            origin = self.started_ns if self.state == 'initializing' else self.recovery_ns
            limit = 30 if self.state == 'initializing' else 10
            if self.state not in ('mapped','local_navigation') and now_ns - origin >= limit * 1_000_000_000:
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


def exploration_motion(elapsed):
    phase=int(elapsed//4)%6
    commands=((.5,0,0,15),(.45,.1,-.15,15),(.5,0,.15,15),
              (0,0,0,0),(.45,-.15,0,15),(.5,0,0,15))
    return list(commands[phase]),False,'exploration_brake' if phase==3 else 'exploration_motion_probe'


def demonstration(value):
    """No destination coordinates. Targets come from observed memory only."""
    state = value['map_status']; elapsed = value['initialization_elapsed_seconds']
    if state in ('mapped','local_navigation') and value['goal_probability'] >= value.get('goal_match_threshold', .95):
        return [0.,0.,0.,0.],True,'visually_supported_stop'
    depth=value.get('depth_tokens')
    if depth is not None and depth[...,1].mean()>.5 and state in ('initializing','local_navigation','mapped'):
        grid=depth.reshape(15,20,2)
        # Visible sectors only. The lower quartile avoids steering through a
        # sector whose average depth conceals a nearby branch or wall.
        clearance=[]
        for first,last in ((0,7),(7,13),(13,20)):
            patch=grid[4:11,first:last];samples=patch[...,0][patch[...,1]>.8]*80
            clearance.append(float(samples.quantile(.25)) if samples.numel() else 0.)
        if clearance[1]<4:
            yaw=-15. if clearance[0]>clearance[2] else 15.
            return [0.,0.,0.,yaw],False,'observed_obstacle_turn'
        if elapsed<24 and state in ('initializing','local_navigation') and not value['target_available']:
            # Local handover can occur after one second. It must not erase
            # the bounded translating-turn, vertical and braking examples.
            # Observed obstacles above and the independent path filter still
            # override this deterministic simulator exploration schedule.
            return exploration_motion(elapsed)
        if state=='initializing' or not value['target_available']:
            # Translating turns bootstrap parallax. Once local vision is
            # ready, an open forward sector permits straight exploration;
            # a permanent yaw bias would make the teacher circle in place.
            yaw=(-8. if clearance[0]>clearance[2] else 8.) if state=='initializing' else 0.
            return [.4,0.,0.,yaw],False,'observed_free_translation'
    if state == 'initializing':
        # A full translating panorama scan exposes nearby structure when the
        # initial view is water/sky. Reversing yaw each phase could keep the
        # entire startup in the same uninformative view. Braking remains an
        # explicit example, and no hidden target chooses the scan direction.
        return exploration_motion(elapsed)
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
