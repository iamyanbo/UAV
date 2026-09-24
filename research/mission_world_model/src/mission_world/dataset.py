"""Deterministic real-city trajectories and isolated counterfactual label branches.

Collection labels are not observations. Policy/map consumers are explicitly given
prefix paths; branch images are consumed only by offline feature/label stages.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .upstream import sha256


def cache_signature(root, stage):
    from importlib.metadata import version, PackageNotFoundError
    dependencies = {
        "encode_dataset": ("vision.py", "dataset.py", "upstream.py"),
        "map_dataset": ("mapping.py", "scene.py", "contracts.py", "dataset.py", "upstream.py"),
        "ground_dataset": ("grounding.py", "contracts.py", "dataset.py", "upstream.py")}
    packages = {}
    for name in ("torch", "numpy", "gsplat", "transformers", "autoawq", "pillow"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    result = dict(dataset_sha256=sha256(Path(root) / "experiment/dataset.json"),
                  source={name: sha256(Path(__file__).parent / name) for name in dependencies[stage]}, packages=packages)
    if stage == "map_dataset":
        result["encoded_features_manifest_sha256"] = sha256(Path(root) / "experiment/cache-manifests/encode_dataset.json")
    return result


def reuse_cache(root, stage):
    path = Path(root) / "experiment/cache-manifests" / (stage + ".json")
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if manifest["signature"] != cache_signature(root, stage):
        return False
    for relative, expected in manifest["artifacts"].items():
        artifact = Path(root) / "experiment" / relative
        if not artifact.exists() or sha256(artifact) != expected:
            raise RuntimeError("Derived artifact changed; explicit regeneration is required: " + relative)
    from .pipeline import receipt
    receipt(Path(root), stage, {"passed": True, "resumed_verified_cache": True,
                              "manifest": str(path), "artifacts": len(manifest["artifacts"])})
    return True


def commit_cache(root, stage, paths):
    base = Path(root) / "experiment"
    destination = base / "cache-manifests" / (stage + ".json")
    destination.parent.mkdir(exist_ok=True)
    payload = dict(signature=cache_signature(root, stage), artifacts={str(path.relative_to(base)): sha256(path) for path in paths})
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2)); temporary.replace(destination)


def processed_identity(root):
    base = Path(root) / "experiment/cache-manifests"
    return {stage: sha256(base / (stage + ".json")) for stage in ("encode_dataset", "map_dataset", "ground_dataset")}


def observation_artifact(stem, extension):
    """Match the committed v1 writer, including its decimal-horizon stems.

    The manifest's h0.5 stem was written as h0.png by Path.with_suffix.
    Preserve and resolve those hash-committed files, never rename raw evidence.
    """
    return Path(stem).with_suffix(extension)


def save_observation(path, observed, state, command, timestamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observed["rgb"]).save(path.with_suffix(".png"))
    np.savez_compressed(path.with_suffix(".npz"), depth=observed["depth"], c2w=observed["c2w"],
                        intrinsics=observed["intrinsics"], state=state, command=command, timestamp=timestamp)


def collect(root, episodes=16, frames=40):
    from .scene import CityScene
    from .flight import FigsDynamics
    from .pipeline import receipt
    scene, dynamics = CityScene(root), FigsDynamics(root)
    output = Path(root) / "experiment"
    if (output / "dataset.json").exists():
        saved = json.loads((output / "dataset.json").read_text())
        if len(saved["episodes"]) != episodes or any(len(e["frames"]) != frames for e in saved["episodes"]):
            raise RuntimeError("Existing dataset does not match the requested collection recipe")
        for episode in saved["episodes"]:
            for relative, expected in episode["artifacts"].items():
                if sha256(output / relative) != expected:
                    raise RuntimeError("Raw dataset artifact changed: " + relative)
        receipt(Path(root), "collect", {"passed": True, "resumed_verified_dataset": True, "episodes": episodes,
                "manifest": str(output / "dataset.json"), "manifest_sha256": sha256(output / "dataset.json")})
        return
    output.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(3407)
    candidates = np.arange(0, len(scene.cameras), max(1, len(scene.cameras)//episodes))[:episodes]
    manifest = dict(version=1, scene=scene.receipt, seed=3407, episodes=[],
        split_definition="episode-disjoint and camera-start X-half held out; same city, image footprints may overlap; not unseen-site generalization",
        horizons=[.5, 1., 2.], branches_per_prefix=4, sensor_track="privileged pose and reconstructed expected depth",
        collision_ground_truth=False)
    camera_positions = np.asarray([scene.camera(int(i))[0][:3, 3] for i in candidates])
    ordering = np.argsort(camera_positions[:, 0])
    test_ids = set(int(candidates[i]) for i in ordering[-max(2, episodes//4):])
    for episode_index, camera_index in enumerate(candidates):
        name = f"episode_{episode_index:03d}"
        base = output / name
        random = np.random.default_rng(3407 + int(camera_index))
        if (base / "episode.json").exists():
            completed = json.loads((base / "episode.json").read_text())
            if completed["initial_camera"] != int(camera_index) or len(completed["frames"]) != frames:
                raise RuntimeError("Completed episode recipe mismatch")
            for relative, expected in completed["artifacts"].items():
                if sha256(output / relative) != expected:
                    raise RuntimeError("Completed episode changed: " + relative)
            manifest["episodes"].append(completed)
            continue
        c2w, intrinsics = scene.camera(int(camera_index))
        start = np.array([0., 0., -c2w[2, 3], 5., 0., 0., 0., 0., 0., 1.])
        state = start.copy()
        entry = dict(id=name, initial_camera=int(camera_index), split="test" if int(camera_index) in test_ids else "train", frames=[], prefixes=[])
        for frame in range(frames):
            command = np.array([dynamics.hover, 0., .004*np.sin(frame*.2), .015*np.cos(frame*.1)])
            camera = dynamics.camera_pose(state, c2w, start, 1.)
            observed = scene.render(camera, intrinsics)
            if observed["valid_fraction"] < .6:
                raise RuntimeError(f"Insufficient view support: {name}/{frame}; no replacement toy observation")
            location = base / "observations" / f"{frame:04d}"
            save_observation(location, observed, state, command, frame*.5)
            entry["frames"].append(frame)
            if frame in (7, 15, 23, 31):
                prefix = dict(frame=frame, branches=[])
                for branch in range(4):
                    controls = np.zeros((10, 4)); controls[:, 0] = dynamics.hover
                    controls[:, 0] += random.uniform(-.02, .02)
                    controls[:, 1:3] = random.uniform(-.06, .06, size=(1, 2))
                    controls[:, 3] = random.uniform(-.2, .2)
                    for horizon in manifest["horizons"]:
                        future = state.copy()
                        for control in controls:
                            future = dynamics.step(future, control, horizon/10)
                        future_camera = dynamics.camera_pose(future, c2w, start, 1.)
                        target = scene.render(future_camera, intrinsics)
                        target_path = base / "labels" / f"prefix_{frame:04d}_branch_{branch}_h{horizon:g}"
                        save_observation(target_path, target, future, controls, frame*.5+horizon)
                        prefix["branches"].append(dict(id=branch, horizon=horizon, path=str(target_path.relative_to(output)),
                                                        valid_fraction=target["valid_fraction"]))
                entry["prefixes"].append(prefix)
            print(json.dumps({"episode": name, "frame": frame, "valid_fraction": observed["valid_fraction"],
                              "label_branches": sum(len(p["branches"]) for p in entry["prefixes"])}), flush=True)
            state = dynamics.step(state, command, .5)
        entry["artifacts"] = {str(path.relative_to(output)): sha256(path)
                              for folder in (base / "observations", base / "labels")
                              for path in sorted(folder.iterdir()) if path.suffix in (".png", ".npz")}
        manifest["episodes"].append(entry)
        (base / "episode.json").write_text(json.dumps(entry, indent=2))
        # Partial progress is separate from the committed dataset manifest.
        (output / "dataset.partial.json").write_text(json.dumps(manifest, indent=2))
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2))
    receipt(Path(root), "collect", {"passed": True, "episodes": episodes, "frames_per_episode": frames,
            "prefixes": sum(len(e["prefixes"]) for e in manifest["episodes"]), "manifest": str(output / "dataset.json"),
            "claims": "same-city preliminary data, not a benchmark reproduction or unseen-site result"})


def encode_dataset(root):
    from .vision import DinoVision
    from .pipeline import receipt
    base = Path(root) / "experiment"
    if reuse_cache(root, "encode_dataset"):
        return
    manifest = json.loads((base / "dataset.json").read_text())
    encoder = DinoVision(root)
    count = 0
    artifacts = []
    for episode in manifest["episodes"]:
        image_paths = [base / episode["id"] / "observations" / f"{frame:04d}.png" for frame in episode["frames"]]
        image_paths += [observation_artifact(base / b["path"], ".png") for p in episode["prefixes"] for b in p["branches"]]
        for path in image_paths:
            features = encoder.encode(Image.open(path))
            torch.save(features, path.with_suffix(".pt"))
            artifacts.append(path.with_suffix(".pt"))
            count += 1
        print(json.dumps({"encoded_episode": episode["id"], "frames": count}), flush=True)
    commit_cache(root, "encode_dataset", artifacts)
    receipt(Path(root), "encode_dataset", {"passed": True, "encoded_frames": count, **encoder.receipt})


def map_dataset(root):
    from .mapping import ObservedGaussianMap, spatial_hierarchy
    from .contracts import Observation
    from .pipeline import receipt
    base = Path(root) / "experiment"
    if reuse_cache(root, "map_dataset"):
        return
    manifest = json.loads((base / "dataset.json").read_text())
    count = 0
    artifacts = []
    for episode in manifest["episodes"]:
        mapper = ObservedGaussianMap(root, maximum_gaussians=2000000)
        directory = base / episode["id"] / "observations"
        output = base / episode["id"] / "memory"
        output.mkdir(exist_ok=True)
        prefix_ids = {p["frame"] for p in episode["prefixes"]}
        for frame in episode["frames"]:
            if frame > max(prefix_ids):
                break
            values = np.load(directory / f"{frame:04d}.npz")
            observed = Observation(episode["id"], frame, float(values["timestamp"]),
                np.asarray(Image.open(directory / f"{frame:04d}.png").convert("RGB")),
                values["depth"].astype(np.float32), values["c2w"], values["intrinsics"])
            metrics = mapper.update(observed, iterations=10)
            print(json.dumps({"episode": episode["id"], **metrics}), flush=True)
            if frame in prefix_ids:
                map_path = output / f"prefix_{frame:04d}.pt"
                mapper.save(map_path)
                features, geometry, fingerprint = spatial_hierarchy(map_path, directory)
                if len(features) < 512:
                    raise RuntimeError("Insufficient observed hierarchy for the declared 512-token comparison")
                torch.save(dict(features=features, geometry=geometry, fingerprint=fingerprint, last_observed_frame=frame),
                           output / f"hierarchy_{frame:04d}.pt")
                artifacts.extend([map_path, output / f"hierarchy_{frame:04d}.pt"])
                count += 1
    commit_cache(root, "map_dataset", artifacts)
    receipt(Path(root), "map_dataset", {"passed": True, "prefix_maps": count, "map_source": "executed observations only"})


def ground_dataset(root):
    from .grounding import QwenGrounder
    from .pipeline import receipt
    base = Path(root) / "experiment"
    if reuse_cache(root, "ground_dataset"):
        return
    manifest = json.loads((base / "dataset.json").read_text())
    model = QwenGrounder(root)
    count = 0
    artifacts = []
    for episode in manifest["episodes"]:
        for prefix in episode["prefixes"]:
            frame = prefix["frame"]
            image = base / episode["id"] / "observations" / f"{frame:04d}.png"
            output = base / episode["id"] / "missions"
            output.mkdir(exist_ok=True)
            for kind, instruction in (("navigate", "Move toward the prominent building closest to the center of the image."),
                                      ("inspect", "Inspect the prominent building closest to the center of the image.")):
                result = model.ground(instruction, frame, image)
                if result["mission"]["kind"] != kind:
                    raise RuntimeError("Grounder changed the requested mission class")
                torch.save(result, output / f"{frame:04d}_{kind}.pt")
                artifacts.append(output / f"{frame:04d}_{kind}.pt")
                print(json.dumps({"episode": episode["id"], "frame": frame, "mission": result["mission"]}), flush=True)
                count += 1
    commit_cache(root, "ground_dataset", artifacts)
    receipt(Path(root), "ground_dataset", {"passed": True, "grounded_missions": count, **model.receipt})


def observed_target_points(values, bbox):
    depth, intrinsics, camera = values["depth"], values["intrinsics"], values["c2w"]
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:4, 0:w:4]
    z = depth[::4, ::4]
    mask = (z > 0) & (xx >= bbox[0]*w) & (xx < bbox[2]*w) & (yy >= bbox[1]*h) & (yy < bbox[3]*h)
    x, y, z = xx[mask], yy[mask], z[mask]
    camera_points = np.column_stack([(x-intrinsics[0, 2])/intrinsics[0, 0]*z, (y-intrinsics[1, 2])/intrinsics[1, 1]*z, z])
    return camera_points @ camera[:3, :3].T + camera[:3, 3]


def observed_surface_visibility(points, values):
    transform = np.linalg.inv(values["c2w"])
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    k = values["intrinsics"]
    z = camera[:, 2]
    uv = camera[:, :2] / np.maximum(z[:, None], .01)
    px = (uv[:, 0]*k[0, 0]+k[0, 2]).round().astype(int)
    py = (uv[:, 1]*k[1, 1]+k[1, 2]).round().astype(int)
    depth = values["depth"]; h, w = depth.shape
    valid = (z > .1) & (px >= 0) & (py >= 0) & (px < w) & (py < h)
    visible = np.zeros(len(points), dtype=bool)
    idx = np.flatnonzero(valid)
    measured = depth[py[idx], px[idx]]
    visible[idx] = (measured > 0) & (np.abs(measured-z[idx]) < np.maximum(1., .01*z[idx]))
    return float(visible.mean()) if len(points) else 0.


class BranchDataset:
    def __init__(self, root, split):
        self.base = Path(root) / "experiment"
        manifest = json.loads((self.base / "dataset.json").read_text())
        self.items = []
        self.excluded_unresolved = 0
        self._map_key = None
        self._map_tree = None
        self._map_radius = None
        for episode in manifest["episodes"]:
            if episode["split"] != split:
                continue
            for prefix in episode["prefixes"]:
                for kind in ("navigate", "inspect"):
                    mission = torch.load(self.base / episode["id"] / "missions" / f"{prefix['frame']:04d}_{kind}.pt", weights_only=False)
                    if mission["mission"]["unresolved"]:
                        self.excluded_unresolved += 1
                        continue
                    for branch in prefix["branches"]:
                        self.items.append((episode["id"], prefix["frame"], kind, branch))
        if not self.items:
            raise RuntimeError("No grounded branch samples; unresolved references cannot be replaced with oracle targets")

    def __len__(self):
        return len(self.items)

    def future_observed_support(self, episode, frame, future):
        """Label proximity to already observed Gaussians, NOT sensor validity.

        A valid future depth can expose previously unseen space. Labels may use
        that future measurement; the earlier model input never receives it.
        This support test is not calibrated uncertainty or proof of free space.
        """
        from scipy.spatial import cKDTree
        key = (episode, frame)
        if self._map_key != key:
            mapped = torch.load(self.base / episode / "memory" / f"prefix_{frame:04d}.pt", weights_only=False)
            self._map_tree = cKDTree(mapped["params"]["means"].numpy())
            self._map_radius = mapped["params"]["log_scales"].exp().max(1).values.numpy() * 3
            self._map_key = key
        depth, camera, k = future["depth"], future["c2w"], future["intrinsics"]
        height, width = depth.shape
        yy, xx = np.mgrid[0:128, 0:128]
        px = np.minimum(((xx+.5)*width/128).astype(int), width-1)
        py = np.minimum(((yy+.5)*height/128).astype(int), height-1)
        z = depth[py, px]
        valid = z > 0
        points = np.column_stack([((px-k[0, 2])/k[0, 0]*z)[valid], ((py-k[1, 2])/k[1, 1]*z)[valid], z[valid]])
        points = points @ camera[:3, :3].T + camera[:3, 3]
        distance, nearest = self._map_tree.query(points, workers=4)
        support = np.zeros((128, 128), dtype=np.float32)
        support[valid] = distance <= np.maximum(.1, self._map_radius[nearest])
        return torch.from_numpy(support).reshape(32, 4, 32, 4).mean((1, 3)).flatten()

    def __getitem__(self, index):
        episode, frame, kind, branch = self.items[index]
        directory = self.base / episode
        prefix = torch.load(directory / "memory" / f"hierarchy_{frame:04d}.pt", weights_only=False)
        if prefix["last_observed_frame"] != frame:
            raise RuntimeError("Future-contaminated map prefix")
        mission = torch.load(directory / "missions" / f"{frame:04d}_{kind}.pt", weights_only=False)
        present = np.load(directory / "observations" / f"{frame:04d}.npz")
        future = np.load(observation_artifact(self.base / branch["path"], ".npz"))
        geometry = prefix["geometry"].clone()
        geometry[:, :3] = (geometry[:, :3] - torch.from_numpy(present["c2w"][:3, 3]).float()) / 100
        geometry[:, 3] /= 100
        geometry[:, 5] = float(present["timestamp"]) - geometry[:, 5]
        ego = torch.zeros(16); ego[:10] = torch.from_numpy(present["state"]).float(); ego[:3] /= 100
        ego[10:16] = torch.from_numpy(present["c2w"][:3, :2].copy()).float().flatten()
        camera = np.linalg.inv(present["c2w"]) @ future["c2w"]
        camera[:3, 3] /= 100
        # Future camera comes from known applied-action dynamics, not a visual
        # target. This explicit egomotion adapter is common to every model arm.
        inputs = dict(memory=prefix["features"], geometry=geometry, mission=mission["embedding"], ego=ego,
                      actions=torch.from_numpy(future["command"]).float(), future_camera=torch.from_numpy(camera).float().flatten(),
                      horizon=torch.tensor(branch["horizon"]))
        h, w = present["depth"].shape
        k = present["intrinsics"]
        inputs["camera_intrinsics"] = torch.tensor([k[0, 0]/w, k[1, 1]/h, k[0, 2]/w, k[1, 2]/h], dtype=torch.float32)
        valid = torch.nn.functional.interpolate(torch.from_numpy((future["depth"]>0).astype(np.float32))[None, None], size=(32, 32), mode="area")[0, 0].flatten() > .9
        points = observed_target_points(present, mission["mission"]["bbox_xyxy"])
        if len(points) < 16:
            raise RuntimeError("Grounded bbox has insufficient observed depth evidence")
        center = np.median(points, axis=0)
        # The VLM reference is tied back to observed DINO patches and measured
        # geometry. No hidden simulator target coordinate enters the model.
        bbox = mission["mission"]["bbox_xyxy"]
        grid_y, grid_x = np.mgrid[0:32, 0:32]
        region = ((grid_x+.5)/32 >= bbox[0]) & ((grid_x+.5)/32 < bbox[2]) & ((grid_y+.5)/32 >= bbox[1]) & ((grid_y+.5)/32 < bbox[3])
        present_features = torch.load(directory / "observations" / f"{frame:04d}.pt", weights_only=True)
        if not region.any():
            raise RuntimeError("Grounded reference is smaller than the declared visual patch grid")
        inputs["mission_visual"] = present_features[torch.from_numpy(region.flatten())].mean(0)
        anchor = np.concatenate([(center-present["c2w"][:3, 3])/100, np.std(points, axis=0)/100, [kind == "navigate", kind == "inspect"]])
        inputs["mission_anchor"] = torch.from_numpy(anchor).float()
        before_distance = np.linalg.norm(center-present["c2w"][:3, 3])
        after_distance = np.linalg.norm(center-future["c2w"][:3, 3])
        before_visibility = observed_surface_visibility(points, present)
        after_visibility = observed_surface_visibility(points, future)
        progress = (before_distance-after_distance)/50 if kind == "navigate" else after_visibility-before_visibility
        coverage = self.future_observed_support(episode, frame, future)
        targets = dict(features=torch.load(observation_artifact(self.base / branch["path"], ".pt"), weights_only=True), valid=valid, coverage=coverage,
                       outcomes=torch.tensor([progress, after_visibility, branch["valid_fraction"]], dtype=torch.float32))
        metadata = dict(episode=episode, frame=frame, mission=kind, branch=branch["id"], horizon=branch["horizon"],
                        memory_fingerprint=prefix["fingerprint"], label_scope="VLM-bbox referenced surface only; hidden surfaces unknown",
                        future_unobserved_fraction=float((1-coverage)[valid].mean()) if valid.any() else None,
                        coverage_definition="measured future surfaces near prefix Gaussians; not calibrated uncertainty")
        return inputs, targets, metadata
