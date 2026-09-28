"""Pure geometry and camera contracts shared by acquisition and inference.

Pitch is positive upward. Transforms map optical coordinates into body NED.
No simulator state or task labels are accessible from this module.
"""
import math
import numpy as np


def camera_rotation(pitch_degrees):
    p = math.radians(float(pitch_degrees))
    c, s = math.cos(p), math.sin(p)
    return np.array([[0., s, c], [1., 0., 0.], [0., c, -s]])


def camera_pitch(calibration):
    rotation = np.asarray(calibration['camera_to_body_rotation']).reshape(3, 3)
    return math.degrees(math.atan2(-rotation[2, 2], rotation[0, 2]))


def direction_pitch(delta):
    d = np.asarray(delta)
    return float(np.clip(math.degrees(math.atan2(-d[2], np.linalg.norm(d[:2]))), -90, 90))


def envelope(value):
    bounds = np.asarray(value['bounds_ned_m'], dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('Declare finite scene flight bounds, including ceiling and lower limit')
    if value.get('camera_profile') != 'pitch-rgb/v1':
        raise ValueError('Qualification requires the pitch-rgb/v1 camera profile')
    return bounds


def path_time(points, cfg, initial_yaw=None, initial_pitch=0.):
    """Travel-time estimate, including look/turn and conservative braking.

    This is a reference estimate, never measured flight time or optimality.
    """
    total = 0.; yaw = initial_yaw; pitch = initial_pitch
    for a, b in zip(points, points[1:]):
        d = np.asarray(b) - a
        if np.linalg.norm(d) < 1e-6:
            continue
        desired = math.atan2(d[1], d[0]) if np.linalg.norm(d[:2]) > .01 else yaw
        if yaw is not None and desired is not None:
            total += abs((desired-yaw+math.pi) % (2*math.pi)-math.pi) / math.radians(cfg['yaw_rate_dps'])
        new_pitch = direction_pitch(d)
        total += abs(new_pitch-pitch)/45. + .2
        total += max(np.linalg.norm(d[:2])/cfg['maximum_speed_mps'], abs(d[2])/cfg['vertical_speed_mps'])
        total += min(cfg['maximum_speed_mps'], np.linalg.norm(d))/cfg['braking_mps2']
        yaw, pitch = desired, new_pitch
    return float(total)


def track(points, position, yaw, cfg, speed_scale=1.):
    """Same bounded waypoint tracker for actual and proposed commands."""
    p = np.asarray(position)
    remaining = [np.asarray(x) for x in points]
    while len(remaining) > 1 and np.linalg.norm(remaining[0]-p) < 2.:
        remaining.pop(0)
    d = remaining[0]-p
    c, s = math.cos(yaw), math.sin(yaw)
    body = np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1], d[2]])
    horizontal = np.linalg.norm(body[:2])
    error = math.atan2(body[1], body[0]) if horizontal > .1 else 0.
    speed = min(cfg['maximum_speed_mps']*speed_scale, math.sqrt(2*cfg['braking_mps2']*horizontal))
    speed *= max(0., math.cos(error))
    xy = body[:2]/max(horizontal, 1e-6)*speed
    command = [float(xy[0]), float(xy[1]), float(np.clip(body[2], -1, 1)),
               float(np.clip(math.degrees(error)*1.5, -45, 45))]
    return command, direction_pitch(body), remaining


def phase_for(command, mode='transit'):
    if command[2]<-.1:return 'climb'
    if command[2]>.1:return 'descent'
    if mode in ('localize', 'search', 'scan', 'inspect', 'verify', 'approach', 'settle'):
        return {'scan': 'scan', 'search': 'search', 'inspect': 'scan', 'verify': 'settle'}.get(mode, mode)
    return 'climb' if command[2] < -.1 else 'descent' if command[2] > .1 else 'transit'
