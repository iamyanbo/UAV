"""Observation-built SplaTAM-derived RGB-D mapping, with gsplat backend.

Uses the pinned upstream backprojection function verbatim at runtime. Mapping
uses observed-depth insertion and RGB/SSIM+depth optimization; expected-depth
normalization, pixel-spacing initialization, backend and known poses are explicit
adaptations. This is NOT a native SplaTAM tracking/SLAM reproduction.
"""
import ast
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .contracts import Observation
from .scene import rasterize
from .upstream import verify_revision


def upstream_backprojection(root):
    repo = Path(root) / "upstream/splatam"
    verify_revision(repo, "da6bbcd24c248dc884ac7f49d62e91b841b26ccc")
    path = repo / "scripts/splatam.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_pointcloud")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_pointcloud"]


class ObservedGaussianMap:
    def __init__(self, root, maximum_gaussians=750000, pixel_stride=2):
        self.backproject = upstream_backprojection(root)
        self.maximum = maximum_gaussians
        self.pixel_stride = pixel_stride
        self.params = None
        self.support_frame = np.empty(0, dtype=np.int64)
        self.support_patch = np.empty(0, dtype=np.int64)
        self.timestamps = np.empty(0, dtype=np.float32)
        self.frames = []
        self.episode = None

    def fingerprint(self):
        digest = hashlib.sha256()
        if self.params is not None:
            for name in sorted(self.params):
                digest.update(self.params[name].numpy().tobytes())
        digest.update(self.support_frame.tobytes())
        digest.update(self.support_patch.tobytes())
        digest.update(self.timestamps.tobytes())
        digest.update(str((self.episode, tuple(self.frames))).encode())
        return digest.hexdigest()

    @staticmethod
    def ssim(x, y):
        # SplaTAM's SSIM form: 11x11 Gaussian window sigma1.5, C1=.01^2 C2=.03^2.
        coords = torch.arange(11, device=x.device, dtype=x.dtype) - 5
        gaussian = torch.exp(-coords.square() / (2 * 1.5**2))
        gaussian = gaussian / gaussian.sum()
        window = (gaussian[:, None] * gaussian[None, :])[None, None].expand(3, 1, -1, -1)
        conv = lambda t: torch.nn.functional.conv2d(t, window, padding=5, groups=3)
        mx, my = conv(x), conv(y)
        vx, vy, cov = conv(x*x) - mx*mx, conv(y*y) - my*my, conv(x*y) - mx*my
        return (((2*mx*my + .01**2) * (2*cov + .03**2)) / ((mx*mx + my*my + .01**2) * (vx + vy + .03**2))).mean()

    def update(self, observation: Observation, iterations=10):
        if type(observation) is not Observation:
            raise TypeError("Mapper accepts only the observation contract, never simulator scene/state objects")
        observation.validate()
        if self.episode is not None and self.episode != observation.episode:
            raise ValueError("Cross-episode map contamination")
        if self.frames and observation.frame <= self.frames[-1]:
            raise ValueError("Observation prefix must be strictly increasing")
        self.episode = observation.episode
        height, width = observation.depth.shape
        tensor = lambda a: torch.as_tensor(np.array(a, copy=True), dtype=torch.float32, device="cuda")
        color = tensor(observation.rgb).permute(2, 0, 1) / 255
        depth = tensor(observation.depth)
        camera, intrinsics = tensor(observation.c2w), tensor(observation.intrinsics)
        valid = depth > 0
        if not valid.any():
            raise ValueError("Cannot map an observation without valid depth")
        params = {} if self.params is None else {key: value.cuda() for key, value in self.params.items()}
        mask = valid.clone()
        if params:
            with torch.no_grad():
                _, rendered_depth, alpha, _ = rasterize(params["means"], params["quats"], params["log_scales"].exp(),
                    params["opacity_logits"].sigmoid(), params["colors"], camera, intrinsics, width, height)
                error = (rendered_depth - depth).abs()[valid]
                threshold = 50 * error.median().clamp_min(.001)
                mask &= (alpha < .5) | ((rendered_depth > depth) & ((rendered_depth-depth).abs() > threshold))
        sample = torch.zeros_like(mask)
        sample[::self.pixel_stride, ::self.pixel_stride] = True
        mask &= sample
        points, distance = self.backproject(color, depth[None], intrinsics, torch.linalg.inv(camera),
                                           mask=mask.flatten(), compute_mean_sq_dist=True)
        new_count = len(points)
        if len(self.support_frame) + new_count > self.maximum:
            raise RuntimeError("Mapping admission limit reached; map is not truncated or replaced with oracle geometry")
        if new_count:
            quats = points.new_zeros((new_count, 4)); quats[:, 0] = 1
            new = dict(means=points[:, :3], colors=points[:, 3:], quats=quats,
                       log_scales=(distance.sqrt() * self.pixel_stride).clamp_min(.001).log()[:, None].expand(-1, 3).clone(),
                       opacity_logits=points.new_zeros(new_count))
            params = new if not params else {key: torch.cat([params[key], value]) for key, value in new.items()}
            yy, xx = torch.where(mask)
            patches = (yy * 32 // height) * 32 + (xx * 32 // width)
            self.support_frame = np.concatenate([self.support_frame, np.full(new_count, observation.frame, dtype=np.int64)])
            self.support_patch = np.concatenate([self.support_patch, patches.cpu().numpy()])
            self.timestamps = np.concatenate([self.timestamps, np.full(new_count, observation.timestamp, dtype=np.float32)])
        params = {key: nn.Parameter(value.detach()) for key, value in params.items()}
        rates = dict(means=.0001, colors=.0025, quats=.001, log_scales=.001, opacity_logits=.01)
        optimizer = torch.optim.Adam([{"params": [value], "lr": rates[key]} for key, value in params.items()])
        history = []
        for _ in range(iterations):
            rgb, predicted_depth, alpha, _ = rasterize(params["means"], params["quats"], params["log_scales"].exp(),
                params["opacity_logits"].sigmoid(), params["colors"], camera, intrinsics, width, height)
            image = rgb.permute(2, 0, 1)
            rgb_loss = .8 * (image-color).abs().mean() + .2 * (1-self.ssim(image[None], color[None]))
            # Relative-depth loss is an aerial scale adaptation; record it, do not
            # claim equivalence to SplaTAM's indoor absolute-metre objective.
            depth_loss = ((predicted_depth-depth).abs()[valid] / depth[valid].clamp_min(1.)).mean()
            loss = rgb_loss + depth_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite mapping loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        self.params = {key: value.detach().cpu() for key, value in params.items()}
        self.frames.append(observation.frame)
        return dict(frame=observation.frame, gaussians=len(self.support_frame), inserted=new_count,
                    loss_before=history[0], loss_after=history[-1], iterations=iterations)

    def save(self, path):
        torch.save(dict(params=self.params, support_frame=self.support_frame, support_patch=self.support_patch,
                        timestamps=self.timestamps, frames=self.frames, episode=self.episode,
                        fingerprint=self.fingerprint(), provenance="observation-only SplaTAM-derived gsplat RGB-D adaptation"), path)

    @classmethod
    def restore(cls, root, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        mapper = cls(root, maximum_gaussians=2000000)
        for name in ("params", "support_frame", "support_patch", "timestamps", "frames", "episode"):
            setattr(mapper, name, saved[name])
        if mapper.fingerprint() != saved["fingerprint"]:
            raise RuntimeError("Observation map checkpoint fingerprint mismatch")
        return mapper


def spatial_hierarchy(saved_map, feature_directory, maximum_candidates=2048):
    """CPU spatial groups from the entire observed map, not a task-selected oracle.

    Feature associations are observed frame/patch IDs. Coarsening for the reference
    is independent of mission; compact arms see the exact same candidates.
    """
    data = torch.load(saved_map, map_location="cpu", weights_only=False)
    xyz = data["params"]["means"].numpy()
    if not len(xyz):
        raise ValueError("Empty observation-built memory")
    sizes = [1., 4., 16.]
    while True:
        levels = [np.unique(np.floor(xyz/size).astype(np.int64), axis=0, return_inverse=True) for size in sizes]
        if sum(len(keys) for keys, _ in levels) <= maximum_candidates:
            break
        sizes = [size*1.5 for size in sizes]
    cached = {int(frame): torch.load(Path(feature_directory) / f"{int(frame):04d}.pt", weights_only=True).numpy()
              for frame in np.unique(data["support_frame"])}
    output_features, output_geometry = [], []
    for level, ((keys, inverse), size) in enumerate(zip(levels, sizes)):
        count = np.bincount(inverse)
        positions = np.column_stack([np.bincount(inverse, weights=xyz[:, axis])/count for axis in range(3)])
        feats = np.zeros((len(keys), 768), dtype=np.float32)
        # Stream associations; no N_gaussians x 768 allocation.
        for start in range(0, len(xyz), 8192):
            end = min(len(xyz), start+8192)
            selected = np.empty((end-start, 768), dtype=np.float32)
            frame_ids = data["support_frame"][start:end]
            patch_ids = data["support_patch"][start:end]
            for frame in np.unique(frame_ids):
                mask = frame_ids == frame
                selected[mask] = cached[int(frame)][patch_ids[mask]]
            np.add.at(feats, inverse[start:end], selected)
        feats /= count[:, None]
        age = np.bincount(inverse, weights=data["timestamps"])/count
        geometry = np.column_stack([positions, np.full(len(keys), size), np.full(len(keys), level), age,
                                    np.log1p(count), np.ones(len(keys)), np.zeros(len(keys))]).astype(np.float32)
        output_features.append(feats); output_geometry.append(geometry)
    return torch.from_numpy(np.concatenate(output_features)), torch.from_numpy(np.concatenate(output_geometry)), data["fingerprint"]
