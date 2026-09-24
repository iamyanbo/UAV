"""Causal scale inputs shared by collection replay and runtime inference.

No simulator state is accepted here. Pose trajectories retain their original
publication times: an offline reconstruction cannot masquerade as live input.
"""
from collections import deque
import math
import numpy as np

STATISTICS = ('camera_dx', 'camera_dy', 'camera_dz', 'rotation_angle',
              'window_seconds', 'path_length', 'net_distance', 'tracking_age',
              'reprojection_pixels', 'correspondence_weight', 'quality_available',
              'gauge_residual', 'gauge_quality_available', 'feature_age',
              'initialized', 'pose_samples')
WINDOW_NS = 5_000_000_000
INTERVAL_NS = 200_000_000


class ScaleInputs:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.poses = deque()
        self.visual = None
        self.latest_decision = -1

    def observe_pose(self, row, observation):
        if observation['episode_id'] != self.episode_id or row['frame_id'] != observation['frame_id']:
            raise ValueError('Cross-episode or mismatched tracking input')
        if row['estimated_pose_source_frame'] != row['frame_id']:
            return  # Old keyframe poses are not current-frame estimates.
        if row['sim_ns'] != observation['sim_ns'] or row['processed_monotonic_seconds'] < observation['received_monotonic']:
            raise ValueError('Invalid tracking publication time')
        pose = np.asarray(row['estimated_c2w_arbitrary_scale'], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Invalid estimated camera pose')
        if self.poses and row['sim_ns'] <= self.poses[-1]['sim_ns']:
            raise ValueError('Tracking input is not strictly causal')
        self.poses.append(dict(row, pose=pose))
        while len(self.poses) > 1 and self.poses[1]['sim_ns'] < row['sim_ns'] - WINDOW_NS:
            self.poses.popleft()

    def observe_visual(self, row, pooled_raw_tokens):
        value = np.asarray(pooled_raw_tokens, dtype=np.float32)
        if row['episode_id'] != self.episode_id or value.shape != (1024,) or not np.isfinite(value).all():
            raise ValueError('Expected same-episode released ViT-L features')
        if row['completed_monotonic'] < row['received_monotonic']:
            raise ValueError('Feature completed before its observation')
        if self.visual and row['latest_observation_ns'] <= self.visual[0]['latest_observation_ns']:
            raise ValueError('Feature stream is not causal')
        self.visual = row, value

    def sample(self, observation):
        """Call after ingesting only outputs completed by this observation."""
        now, wall = observation['sim_ns'], observation['received_monotonic']
        if observation['episode_id'] != self.episode_id or now <= self.latest_decision:
            raise ValueError('Episode reset or nonmonotonic scale decision required')
        self.latest_decision = now
        if not self.poses or self.visual is None:
            return None
        poses = [p for p in self.poses if p['initialized'] and now-WINDOW_NS <= p['sim_ns'] <= now]
        feature, visual = self.visual
        if (not poses or poses[-1]['processed_monotonic_seconds'] > wall
                or feature['completed_monotonic'] > wall or feature['latest_observation_ns'] > now):
            return None
        first, last = poses[0], poses[-1]
        points = np.asarray([p['pose'][:3, 3] for p in poses])
        delta = first['pose'][:3, :3].T @ (points[-1] - points[0])
        relative_rotation = first['pose'][:3, :3].T @ last['pose'][:3, :3]
        angle = math.acos(float(np.clip((np.trace(relative_rotation)-1)/2, -1, 1)))
        quality = last['pose_quality']
        weight, error = quality.get('correspondence_weight'), quality.get('reprojection_error_pixels')
        gauge = last.get('gauge_alignment_residual')
        statistics = np.asarray([*delta, angle, (last['sim_ns']-first['sim_ns'])/1e9,
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum(), np.linalg.norm(delta),
            (now-last['sim_ns'])/1e9, error or 0., weight or 0., float(error is not None and weight is not None),
            gauge or 0., float(gauge is not None), (now-feature['latest_observation_ns'])/1e9,
            float(last['initialized']), len(poses)], dtype=np.float32)
        commands = [x for x in observation['command_history'] if x['sim_ns'] <= now and x['issued_monotonic'] <= wall]
        if any(len(x['values']) != 4 for x in commands):
            raise ValueError('Scale receives only bounded body velocity/yaw commands')
        command = np.asarray(commands[-1]['values'] if commands else [0., 0., 0., 0.], dtype=np.float32)
        if not np.isfinite(statistics).all() or not np.isfinite(command).all():
            raise ValueError('Nonfinite scale inputs')
        return dict(sim_ns=now, received_monotonic=wall, visual_raw=visual.copy(), statistics=statistics,
                    command=command, source_pose_ns=np.asarray([p['sim_ns'] for p in poses], dtype=np.int64),
                    source_pose_positions=points, latest_observation_ns=max(last['sim_ns'], feature['latest_observation_ns']))
