"""Past-keyframe-only adaptation of Splat-SLAM's pose trajectory filler.

The released offline filler can use the next keyframe. Online estimation must
use only already observed anchors. Temporary image features and a pose-only
factor graph estimate the current pose; they never become historical input.
"""
import torch


class CausalPoseEstimator:
    def __init__(self, net, video):
        self.net, self.video = net, video
        self.latest_features = None
        # Reuse the feature computation already performed by MotionFilter.
        self.hook = net.fnet.register_forward_hook(self._features)

    def _features(self, module, inputs, output):
        self.latest_features = output.detach()

    @torch.no_grad()
    def estimate(self, index, image, intrinsic):
        from thirdparty.glorie_slam.factor_graph import FactorGraph
        video = self.video
        count = video.counter.value
        if count < 2 or self.latest_features is None:
            raise RuntimeError('Current-frame pose requires initialized past anchors')
        if int(video.timestamp[count - 1]) == index:
            return video.get_pose(count - 1, 'cpu'), dict(method='current_keyframe', correspondence_weight=None,
                                                        reprojection_error_pixels=None)
        anchors = torch.arange(max(0, count - 2), count, device='cuda')
        if bool((video.timestamp[anchors] > index).any()):
            raise ValueError('Future anchor is forbidden in online pose estimation')
        # The frontend seeds the next unused slot. Restore it after temporary
        # pose-only optimization so the next keyframe sees unchanged storage.
        fields = ('timestamp', 'images', 'poses', 'disps', 'intrinsics', 'fmaps')
        saved = {name: getattr(video, name)[count].clone() for name in fields}
        # FactorGraph.update upsamples its source anchors even with
        # motion_only=True. The temporary filler must not replace the map's
        # historical depth images with those temporary correlation weights.
        anchor_depth = video.disps_up[anchors].clone()
        graph = None
        try:
            video[count] = (index, image[0], video.poses[count - 1].clone(),
                            video.disps[count - 1].mean(), None,
                            intrinsic / video.down_scale, self.latest_features[0])
            graph = FactorGraph(video, self.net.update, device='cuda:0', max_factors=2)
            graph.add_factors(anchors, torch.full_like(anchors, count))
            for _ in range(12):
                graph.update(count, count + 1, motion_only=True)
            projected, mask = video.reproject(graph.ii, graph.jj)
            weights = graph.weight.mean(-1) * mask.squeeze(-1)
            residual = (projected - graph.target).norm(dim=-1) * video.down_scale
            valid = weights > 0
            error = float((residual * weights).sum() / weights.sum().clamp_min(1e-6))
            pose = video.get_pose(count, 'cpu')
            if not torch.isfinite(pose).all() or not bool(valid.any()):
                raise RuntimeError('Current-frame visual pose has no finite supported solution')
            return pose, dict(method='causal_past_anchor_pose_only', correspondence_weight=float(weights.mean()),
                              reprojection_error_pixels=error, anchor_count=len(anchors))
        finally:
            if graph is not None:
                graph.clear_edges()
            with video.get_lock():
                for name, value in saved.items():
                    getattr(video, name)[count].copy_(value)
                video.disps_up[anchors] = anchor_depth
                video.counter.value = count


def gauge_tracked_video(video_type):
    class GaugeTrackedVideo(video_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gauge_scale = 1.
            self.gauge_changes = 0

        def normalize(self):
            factor = float(self.disps[:self.counter.value].mean())
            if not 0 < factor < float('inf'):
                raise RuntimeError('Invalid monocular gauge normalization')
            super().normalize()
            self.gauge_scale *= factor
            self.gauge_changes += 1

    return GaugeTrackedVideo


def fixed_gauge_pose(pose, gauge_scale):
    result = pose.clone()
    result[:3, 3] /= gauge_scale
    return result


class CausalMapGauge:
    """Align successive estimates using common, previously observed cameras.

    Monocular BA can change its global similarity gauge beyond an explicit
    normalize() call. This fixes coordinate convention using *past estimates*,
    never true poses. It does not estimate metric scale or eliminate drift.
    """
    def __init__(self):
        self.previous = {}
        self.scale = 1.
        self.rotation = torch.eye(3)
        self.translation = torch.zeros(3)
        self.alignment_residual = None

    @torch.no_grad()
    def update(self, video):
        import lietorch
        count = video.counter.value
        poses = lietorch.SE3(video.poses[:count].clone()).inv().matrix().cpu()
        ids = [int(x) for x in video.timestamp[:count].cpu()]
        common = [index for index, key in enumerate(ids) if key in self.previous]
        if len(common) >= 3:
            # Distributed anchors avoid tying the convention to one newly
            # inserted frame or an almost identical pair of hover frames.
            if len(common) > 32:
                common = [common[int(i)] for i in torch.linspace(0, len(common)-1, 32)]
            current = poses[common]
            past = torch.stack([self.previous[ids[i]] for i in common])
            relatives = past[:, :3, :3] @ current[:, :3, :3].transpose(-1, -2)
            u, _, vh = torch.linalg.svd(relatives.mean(0))
            sign = torch.ones(3)
            sign[-1] = torch.det(u @ vh).sign()
            rotation = u @ torch.diag(sign) @ vh
            x, y = current[:, :3, 3], past[:, :3, 3]
            xc, yc = x-x.mean(0), y-y.mean(0)
            denominator = xc.square().sum()
            if float(denominator) > 1e-8:
                scale = float(((xc @ rotation.T)*yc).sum()/denominator)
                if not 0 < scale < float('inf'):
                    raise RuntimeError('Causal camera anchors cannot establish a consistent similarity gauge')
                self.scale = scale
                self.rotation = rotation
                self.translation = y.mean(0) - scale*rotation@x.mean(0)
                residual = self.scale*x@rotation.T + self.translation - y
                self.alignment_residual = float(residual.square().sum(-1).mean().sqrt())
        # Current prefix corrections may improve old estimated positions. The
        # next alignment uses this prefix, not a final globally optimized map.
        self.previous = {key: self.pose(value) for key, value in zip(ids, poses)}

    def pose(self, pose):
        rotation = self.rotation.to(pose.device)
        result = pose.clone()
        result[:3, :3] = rotation @ pose[:3, :3]
        result[:3, 3] = self.scale*rotation@pose[:3, 3] + self.translation.to(pose.device)
        return result
