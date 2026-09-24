"""Observation-only adapter to the released Splat-SLAM Gaussian optimizer.

No dataset loader, true camera, simulator depth, final BA or retrospective
training inputs. Upstream Camera calls its initial transform gt_T; we supply
the RGB tracker estimate there, never a label.
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from src.mapper import Mapper
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from thirdparty.monogs.utils.camera_utils import Camera
from thirdparty.gaussian_splatting.utils.general_utils import rotation_matrix_to_quaternion, quaternion_multiply


class BoundedGaussians(GaussianModel):
    cap = 200000

    def cat_tensors_to_optimizer(self,tensors_dict):
        # Degree-zero SH has shape (N,0,3). Some CUDA cat kernels reject
        # these zero-element optimizer moments. Preserve the exact empty shape
        # without launching a kernel; nonempty tensors use native concatenation.
        def concatenate(a,b):
            if a.shape[1:]!=b.shape[1:]:raise ValueError('Gaussian optimizer shape changed')
            if not a.numel() and not b.numel():return a.new_empty((len(a)+len(b),*a.shape[1:]))
            return torch.cat((a,b),dim=0)
        result={}
        for group in self.optimizer.param_groups:
            if len(group['params'])!=1:raise ValueError('Expected one Gaussian tensor per optimizer group')
            old=group['params'][0];extra=tensors_dict[group['name']]
            state=self.optimizer.state.pop(old,None)
            if state is not None:
                for key in ('exp_avg','exp_avg_sq'):state[key]=concatenate(state[key],torch.zeros_like(extra))
            parameter=torch.nn.Parameter(concatenate(old,extra).detach().requires_grad_(True))
            group['params'][0]=parameter
            if state is not None:self.optimizer.state[parameter]=state
            result[group['name']]=parameter
        return result

    def create_pcd_from_image_and_depth(self, camera, rgb, depth, init=False):
        # Sparse, multiview-validated depth deliberately leaves unknown pixels
        # at zero. Upstream's all-pixel median then becomes zero, producing
        # log(0) Gaussian scales. Size points from supported depths only.
        values = np.asarray(depth)
        valid = values[np.isfinite(values) & (values > 0)]
        if not len(valid):
            raise RuntimeError('No supported RGB depth for Gaussian initialization')
        settings = self.config['mapping']
        adaptive, point_size = settings.get('adaptive_pointsize', False), settings['point_size']
        try:
            if adaptive:
                settings['point_size'] = min(.05, point_size * float(np.median(valid)))
                settings['adaptive_pointsize'] = False
            return super().create_pcd_from_image_and_depth(camera, rgb, depth, init)
        finally:
            settings['adaptive_pointsize'], settings['point_size'] = adaptive, point_size

    def budgeted_gradients(self, gradients, threshold, extent, splitting, children=1):
        if threshold <= 0:
            raise ValueError('Positive densification threshold required for capacity filtering')
        score = gradients.norm(dim=-1)
        size = self.get_scaling[:len(score)].max(dim=-1).values
        selected = (score >= threshold) & (size > self.percent_dense*extent if splitting else size <= self.percent_dense*extent)
        remaining = max(0, (self.cap-len(self._xyz))//children)
        candidates = selected.nonzero().flatten()
        keep = candidates[score[candidates].topk(min(remaining, len(candidates))).indices]
        limited = torch.zeros_like(gradients)
        limited[keep] = gradients[keep]
        self.capacity_limited_candidates = getattr(self, 'capacity_limited_candidates', 0) + len(candidates)-len(keep)
        return limited

    def densify_and_clone(self, gradients, threshold, extent):
        gradients = self.budgeted_gradients(gradients, threshold, extent, splitting=False)
        return super().densify_and_clone(gradients, threshold, extent)

    def densify_and_split(self, gradients, threshold, extent, N=2):
        # Budget the temporary children before upstream removes their parents.
        # Keep the strongest eligible gradients and still run normal pruning.
        gradients = self.budgeted_gradients(gradients, threshold, extent, splitting=True, children=N)
        return super().densify_and_split(gradients, threshold, extent, N)


class CausalMapper(Mapper):
    def __init__(self, config, output, episode_id):
        self.config = copy.deepcopy(config)
        self.config['mapping']['Training']['window_size'] = 8
        self.config['mapping']['Training']['gt_camera'] = False
        self.config['mapping']['online_plotting'] = False
        self.config['mapping']['BA'] = False  # RGB tracker owns pose corrections.
        self.config['data'] = {'output': str(output)}
        self.config['scene'] = episode_id
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.device = 'cuda:0'
        self.dtype = torch.float32
        self.printer = SimpleNamespace(print=lambda *args: None)
        self.cameras_extent = 6.
        self.pipeline_params = SimpleNamespace(**self.config['mapping']['pipeline_params'])
        self.opt_params = SimpleNamespace(**self.config['mapping']['opt_params'])
        self.gaussians = BoundedGaussians(0, config=self.config)
        self.gaussians.init_lr(6.)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.zeros(3, device=self.device)
        self.iteration_count = self.last_sent = 0
        self.current_window = []
        self.viewpoints = {}
        self.occ_aware_visibility = {}
        self.depth_dict = {}
        self.sources = {}
        self.history = []
        self.historical_anchors = {}
        self.version = 0
        self.last_observation_ns = -1
        self.initialized = True
        self.set_hyperparams()

    @torch.no_grad()
    def update_mapping_points(self, frame_idx, w2c, w2c_old, depth, depth_old, intrinsics, method=None):
        # Upstream deformation projects behind-camera points and zero depths
        # without a finite visibility mask. Such pixels cannot justify a depth
        # correction: retain rigid anchor motion and record rejected support.
        g = self.gaussians
        selected = (g.unique_kfIDs == frame_idx).to(self.device)
        if not selected.any():
            return
        xyz = g._xyz.detach().clone()
        points = xyz[selected]
        camera = points @ w2c_old[:3, :3].T + w2c_old[:3, 3]
        pixels = camera @ intrinsics.T
        z = camera[:, 2]
        uv = pixels[:, :2] / z[:, None].clamp_min(1e-6)
        finite = torch.isfinite(uv).all(-1) & torch.isfinite(z) & (z > 1e-5)
        inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
        uv = torch.nan_to_num(uv).long()
        x, y = uv[:, 0].clamp(0, depth.shape[1] - 1), uv[:, 1].clamp(0, depth.shape[0] - 1)
        old, new = depth_old[y, x], depth[y, x]
        valid = inside & torch.isfinite(old) & torch.isfinite(new) & (old > 1e-5) & (new > 1e-5)
        factor = 1 + (new - old) / z.clamp_min(1e-5)
        valid &= torch.isfinite(factor) & (factor > 0)
        factor = torch.where(valid, factor, torch.ones_like(factor))
        transformed = camera * factor[:, None]
        c2w = w2c.inverse()
        xyz[selected] = transformed @ c2w[:3, :3].T + c2w[:3, 3]
        scales = g._scaling.detach().clone()
        scales[selected] += factor.log()[:, None]
        rigid = c2w @ w2c_old
        quaternion = rotation_matrix_to_quaternion(rigid[None])
        rotations = g._rotation.detach().clone()
        rotations[selected] = quaternion_multiply(quaternion.expand_as(rotations[selected]), rotations[selected])
        for name, tensor, attribute in [('xyz', xyz, '_xyz'), ('scaling', scales, '_scaling'), ('rotation', rotations, '_rotation')]:
            if not torch.isfinite(tensor).all():
                raise RuntimeError('Nonfinite corrected Gaussian ' + name)
            setattr(g, attribute, g.replace_tensor_to_optimizer(tensor, name)[name])
        self.rejected_depth_corrections = getattr(self, 'rejected_depth_corrections', 0) + int((~valid).sum())

    def tensors(self, mask=None):
        g = self.gaussians
        names = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
                 'unique_kfIDs', 'n_obs')
        result = {}
        for name in names:
            tensor = getattr(g, name).detach().cpu()
            result[name] = tensor.clone() if mask is None else tensor[mask.cpu()].clone()
        count = torch.zeros(len(g._xyz), dtype=torch.int32)
        latest = torch.zeros(len(g._xyz), dtype=torch.int64)
        for key, visibility in self.occ_aware_visibility.items():
            if key in self.sources and len(visibility) == len(count):
                visible = visibility.detach().cpu().bool()
                count += visible.int()
                latest[visible] = torch.maximum(latest[visible], torch.full_like(latest[visible], self.sources[key]['sim_ns']))
        result['observed_active_views'] = count if mask is None else count[mask.cpu()]
        result['last_observed_sim_ns'] = latest if mask is None else latest[mask.cpu()]
        covariance = g.get_covariance().detach().cpu()
        result['spatial_covariance_packed'] = covariance if mask is None else covariance[mask.cpu()]
        return result

    def archive_keyframe(self, keyframe, observed_ns):
        mask = self.gaussians.unique_kfIDs == keyframe
        path = self.output / f'history-{keyframe:06d}-v{self.version:06d}.pt'
        camera = self.viewpoints[keyframe]
        torch.save(dict(episode_id=self.episode_id, latest_observation_ns=observed_ns,
                        keyframe_source=self.sources[keyframe], gaussians=self.tensors(mask),
                        w2c=camera.world_view_transform.T.detach().cpu(),
                        depth=torch.from_numpy(camera.depth).half()), path)
        self.history.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 keyframe=keyframe, latest_observation_ns=observed_ns))
        self.historical_anchors[keyframe] = dict(w2c=camera.world_view_transform.T.detach().cpu().clone(),
                                                depth=torch.from_numpy(camera.depth.copy()).half())
        self.gaussians.prune_points(mask.to(self.device))
        del self.viewpoints[keyframe]
        del self.depth_dict[keyframe]

    def update(self, keyframe, source, image, depth, valid, w2c, calibration, observed_ns, corrections=()):
        if observed_ns <= self.last_observation_ns or source['sim_ns'] > observed_ns:
            raise ValueError('Noncausal or repeated mapping update')
        if source['episode_id'] != self.episode_id:
            raise ValueError('Cross-episode map update')
        # Apply only corrections computed from this received prefix.
        intrinsic = torch.tensor([[calibration['fx'], 0, calibration['cx']],
                                  [0, calibration['fy'], calibration['cy']], [0, 0, 1]], device=self.device)
        for old_key, corrected_w2c, corrected_depth in corrections:
            if old_key not in self.viewpoints:
                if old_key in self.historical_anchors:
                    self.historical_anchors[old_key] = dict(w2c=corrected_w2c.detach().cpu().clone(),
                                                           depth=corrected_depth.detach().cpu().half())
                continue
            camera = self.viewpoints[old_key]
            old_pose = camera.world_view_transform.T.detach().clone()
            self.update_mapping_points(old_key, corrected_w2c, old_pose, corrected_depth,
                                       self.depth_dict[old_key], intrinsic)
            camera.update_RT(corrected_w2c[:3, :3], corrected_w2c[:3, 3])
            camera.depth = corrected_depth.detach().cpu().numpy()
            self.depth_dict[old_key] = corrected_depth.detach().clone()
        if keyframe in self.viewpoints:
            return None
        if valid.sum().item() < 100:
            return None
        if len(self.current_window) == 8:
            old = self.current_window.pop()
            self.archive_keyframe(old, observed_ns)
        # Reserve worst-case point insertion without exceeding the fixed cap.
        expected = math.ceil(depth.numel() / self.config['mapping']['pcd_downsample_init'])
        while len(self.gaussians._xyz) + expected > self.gaussians.cap and self.current_window:
            old = self.current_window.pop()
            self.archive_keyframe(old, observed_ns)
        fx, fy, cx, cy = [calibration[k] for k in ('fx', 'fy', 'cx', 'cy')]
        projection = getProjectionMatrix2(znear=.01, zfar=100., fx=fx, fy=fy, cx=cx, cy=cy, W=640, H=480).T
        filtered = depth.detach().clone()
        filtered[~valid] = 0
        camera = Camera(keyframe, image.to(self.device), filtered.cpu().numpy(), w2c,
                        projection, fx, fy, cx, cy, 2 * math.atan(640 / (2 * fx)),
                        2 * math.atan(480 / (2 * fy)), 480, 640, device=self.device)
        camera.update_RT(w2c[:3, :3], w2c[:3, 3])
        camera.cam_rot_delta.requires_grad_(False)
        camera.cam_trans_delta.requires_grad_(False)
        camera.compute_grad_mask(self.config)
        first = not self.viewpoints
        self.sources[keyframe] = {k: source[k] for k in ('episode_id', 'frame_id', 'sim_ns', 'rgb_sha256')}
        self.current_window.insert(0, keyframe)
        self.viewpoints[keyframe] = camera
        self.depth_dict[keyframe] = filtered
        self.add_next_kf(keyframe, camera, init=first, depth_map=filtered.cpu().numpy())
        if first:
            self.initialize_map(keyframe, camera)
        params = [{'params': [view.exposure_a, view.exposure_b], 'lr': .01} for view in self.viewpoints.values()]
        self.keyframe_optimizers = torch.optim.Adam(params)
        self.map(self.current_window, iters=self.mapping_itr_num)
        bad = {name: int((~torch.isfinite(getattr(self.gaussians, name))).sum()) for name in ('_xyz', '_scaling', '_rotation', '_opacity')}
        if any(bad.values()):
            raise RuntimeError('Nonfinite Gaussian reconstruction: ' + json.dumps(bad))
        if len(self.gaussians._xyz) > self.gaussians.cap:
            raise RuntimeError('Gaussian capacity exceeded')
        self.version += 1
        self.last_observation_ns = observed_ns
        return self.publish(observed_ns)

    def publish(self, observed_ns):
        cameras = {}
        for key, view in self.viewpoints.items():
            cameras[key] = dict(source=self.sources[key], w2c=view.world_view_transform.T.detach().cpu().clone(),
                                depth=torch.from_numpy(view.depth.copy()).half())
        state = dict(episode_id=self.episode_id, version=self.version, latest_observation_ns=observed_ns,
                     coordinate_frame='RGB reconstruction; arbitrary scale', metric_scale_established=False,
                     gaussians=self.tensors(), cameras=cameras, history=list(self.history),
                     historical_anchors=self.historical_anchors,
                     unsupported_depth_corrections=getattr(self, 'rejected_depth_corrections', 0),
                     capacity_limited_densification_candidates=getattr(self.gaussians, 'capacity_limited_candidates', 0),
                     optimizer_updates=self.iteration_count, gaussian_opacity_is_collision_probability=False)
        path = self.output / f'memory-v{self.version:06d}.pt'
        torch.save(state, path)
        row = dict(version=self.version, latest_observation_ns=observed_ns, path=path.name,
                   published_monotonic_seconds=time.monotonic(),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(), active_keyframes=len(cameras),
                   active_gaussians=len(self.gaussians._xyz), historical_submaps=len(self.history),
                   optimizer_updates=self.iteration_count)
        with (self.output / 'versions.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return row
