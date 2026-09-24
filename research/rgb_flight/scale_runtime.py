"""Observation-only execution of the calibrated recurrent scale prior.

The model estimates meters per arbitrary map unit. It does not correct drift,
orientation errors, or unsafe geometry. A calibration artifact alone never
enables autonomous movement: full-flight metric acceptance remains separate.
"""
from collections import deque
import math
import torch
from learning_models import ScaleEstimator
from scale_features import ScaleInputs, INTERVAL_NS


class RuntimeScale:
    def __init__(self, artifact, episode_id, device='cpu'):
        saved = torch.load(artifact, map_location='cpu', weights_only=True)
        if saved['schema'] != 1 or not saved['manifest_sha256']:
            raise ValueError('Scale inference requires a trained calibration artifact')
        self.device, self.artifact = device, saved
        self.model = ScaleEstimator().to(device).eval().requires_grad_(False)
        self.model.load_state_dict(saved['model'], strict=True)
        self.projection = {key: saved['projection'][key].to(device) for key in ('mean','components')}
        self.inputs = ScaleInputs(episode_id)
        self.history = deque(maxlen=saved['window_steps'])
        self.next_ns = -1

    def observe_pose(self, row, observation):
        self.inputs.observe_pose(row, observation)

    def observe_visual(self, row, pooled_raw_tokens):
        self.inputs.observe_visual(row, pooled_raw_tokens)

    @torch.inference_mode()
    def estimate(self, observation):
        now = observation['sim_ns']
        if now < self.next_ns:
            return None
        self.next_ns = now+INTERVAL_NS
        sample = self.inputs.sample(observation)
        if sample is None:
            return None
        if self.history and now-self.history[-1]['sim_ns'] > self.artifact['maximum_gap_seconds']*1e9:
            self.history.clear()
        self.history.append(sample)
        if len(self.history) < self.history.maxlen:
            return None
        values = list(self.history)
        visual = torch.stack([torch.as_tensor(x['visual_raw']) for x in values]).to(self.device)
        visual = (visual-self.projection['mean'])@self.projection['components']
        statistics = torch.stack([torch.as_tensor(x['statistics']) for x in values]).to(self.device)
        commands = torch.stack([torch.as_tensor(x['command']) for x in values]).to(self.device)
        prediction = self.model(visual[None], statistics[None], commands[None])
        log_scale = float(prediction['mean'][0,-1])
        sigma = float(prediction['sigma'][0,-1])
        probability = float(prediction['usable_logit'][0,-1].sigmoid())
        radius = sigma*self.artifact['interval_multiplier']
        if not all(math.isfinite(x) for x in (log_scale,sigma,probability,radius)):
            raise RuntimeError('Nonfinite predicted metric scale')
        bounds = self.artifact['acceptance']
        current = sample['statistics']
        fresh = current[7] <= bounds['maximum_tracking_age_seconds'] and current[13] <= bounds['maximum_feature_age_seconds']
        supported = probability >= bounds['minimum_usable_probability'] and radius <= bounds['maximum_log_scale_radius'] and fresh
        # No clipping of an extreme estimate into an apparently safe value.
        supported = bool(supported and abs(log_scale)+radius < 30)
        return dict(episode_id=self.inputs.episode_id,sim_ns=now,latest_observation_ns=sample['latest_observation_ns'],
            log_meters_per_map_unit=log_scale,log_scale_sigma=sigma,empirical_95pct_log_radius=radius,
            usable_probability=probability,calibration_supported=supported,
            flight_usable=bool(supported and self.artifact['metric_navigation_accepted']),
            tracking_age_seconds=float(current[7]),feature_age_seconds=float(current[13]))
