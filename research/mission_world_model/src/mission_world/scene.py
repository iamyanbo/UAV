"""Private CityGaussian scene backend. Never pass this object to a policy/map.

Released MatrixCity coordinates are 100-metre units (MatrixCity README, pose
splits; CityGaussian transform_json2txt_mc_all.py). We scale both means/scales
and cameras into metres. Renderer backend is gsplat, not native CityGS LoD.
"""
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
import torch


def rasterize(means, quaternions, scales, opacities, colors, c2w, intrinsics, width, height, sh_degree=None):
    from gsplat import rasterization
    rendered, alpha, info = rasterization(means=means, quats=quaternions, scales=scales,
        opacities=opacities, colors=colors, viewmats=torch.linalg.inv(c2w)[None],
        Ks=intrinsics[None], width=width, height=height, packed=True,
        near_plane=.1, far_plane=3000., render_mode="RGB+ED", sh_degree=sh_degree)
    return rendered[0, ..., :3].clamp(0, 1), rendered[0, ..., 3], alpha[0, ..., 0], info


class CityScene:
    def __init__(self, root, width=960, height=540, max_visible=2500000, lod=75):
        self.path = Path(root) / "scenes/MatrixCityAerial_1/mc_aerial_c36"
        if lod not in (0, 50, 66, 75):
            raise ValueError("Only the published complete-scene asset levels are supported")
        gaussian_path = self.path / "point_cloud/iteration_30000/point_cloud.ply" if lod == 0 else Path(root) / f"scenes/MatrixCityAeiral_2/mc_aerial_c36_light_{lod}_vq/point_cloud.ply"
        self.vertices = PlyData.read(gaussian_path, mmap="r")["vertex"].data
        print(json.dumps({"loading_scene": str(gaussian_path), "gaussians": len(self.vertices)}), flush=True)
        self.positions = np.column_stack([self.vertices[n] for n in ("x", "y", "z")]) * 100
        self.radii = np.exp(np.maximum.reduce([self.vertices[f"scale_{i}"] for i in range(3)])) * 300
        self.cameras = json.loads((self.path / "cameras.json").read_text())
        self.width, self.height, self.max_visible = width, height, max_visible
        if width < 640 or height < 480:
            raise ValueError("Sensor resolution below the declared experiment floor")
        self.rest = sorted((name for name in self.vertices.dtype.names if name.startswith("f_rest_")), key=lambda n: int(n.rsplit("_", 1)[1]))
        self.sh_degree = round(np.sqrt(1 + len(self.rest) // 3) - 1)
        self.receipt = dict(scene="MatrixCityAerial", gaussian_count=len(self.vertices),
                            renderer="gsplat 1.4.0; released CityGS asset, not native dynamic LoD selection",
                            released_pruning_level_percent=lod, gaussian_asset=str(gaussian_path),
                            meters_per_source_unit=100., resolution=[width, height],
                            depth_track="privileged reconstructed expected depth, NOT exact mesh depth",
                            scale_evidence=["https://github.com/city-super/MatrixCity#data-structure",
                                            "upstream/citygaussian/tools/transform_json2txt_mc_all.py"])

    def camera(self, index):
        record = self.cameras[index]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = record["rotation"]
        c2w[:3, 3] = np.asarray(record["position"]) * 100
        k = np.array([[record["fx"] * self.width / record["width"], 0, self.width / 2],
                      [0, record["fy"] * self.height / record["height"], self.height / 2], [0, 0, 1]], dtype=np.float32)
        if not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Invalid released camera rotation")
        return c2w, k

    def visible_indices(self, c2w, intrinsics):
        """Conservative CPU frustum cull of the entire mmap, including splat extent.

        No world crop / top-N geometry replacement. Oversized views fail admission.
        """
        w2c = np.linalg.inv(c2w)
        selected = []
        count = 0
        for start in range(0, len(self.vertices), 250000):
            xyz = self.positions[start:start + 250000]
            camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
            radius = self.radii[start:start + 250000]
            tx, ty = self.width / (2 * intrinsics[0, 0]), self.height / (2 * intrinsics[1, 1])
            mask = (camera[:, 2] + radius > .1) & (camera[:, 2] - radius < 3000)
            mask &= np.abs(camera[:, 0]) <= camera[:, 2] * tx + radius * np.sqrt(1 + tx*tx)
            mask &= np.abs(camera[:, 1]) <= camera[:, 2] * ty + radius * np.sqrt(1 + ty*ty)
            ids = np.flatnonzero(mask) + start
            count += len(ids)
            if count > self.max_visible:
                raise RuntimeError(f"Visible Gaussian count exceeds admitted GPU staging budget ({self.max_visible}); no truncation performed")
            selected.append(ids)
        return np.concatenate(selected)

    @torch.inference_mode()
    def render(self, c2w, intrinsics):
        ids = self.visible_indices(c2w, intrinsics)
        if len(ids) == 0:
            raise RuntimeError("Camera has no reconstructed scene support")
        v = self.vertices[ids]
        tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")
        means = tensor(np.column_stack([v[n] for n in ("x", "y", "z")]) * 100)
        scales = tensor(np.exp(np.column_stack([v[f"scale_{i}"] for i in range(3)])) * 100)
        quats = torch.nn.functional.normalize(tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)])), dim=-1)
        opacity = torch.sigmoid(tensor(v["opacity"]))
        dc = np.column_stack([v[f"f_dc_{i}"] for i in range(3)])[:, None, :]
        rest = np.column_stack([v[n] for n in self.rest]).reshape(len(v), 3, -1).transpose(0, 2, 1)
        colors = tensor(np.concatenate([dc, rest], axis=1))
        rgb, depth, alpha, _ = rasterize(means, quats, scales, opacity, colors, tensor(c2w), tensor(intrinsics), self.width, self.height, self.sh_degree)
        valid = (alpha > .9) & torch.isfinite(depth) & (depth > .1) & (depth < 3000)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        result = dict(rgb=(rgb * 255).round().byte().cpu().numpy(), depth=depth.cpu().numpy(),
                      alpha=alpha.cpu().numpy(), c2w=np.asarray(c2w, dtype=np.float32), intrinsics=np.asarray(intrinsics, dtype=np.float32),
                      render_gaussians=len(ids), valid_fraction=float(valid.float().mean()))
        return result
