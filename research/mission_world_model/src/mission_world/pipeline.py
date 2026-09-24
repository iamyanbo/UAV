"""Inspectable stage implementations. No generated code or background agents."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch


def receipt(root, stage, value):
    from .upstream import implementation_fingerprint
    target = root / "gates" / (stage + ".json")
    target.parent.mkdir(exist_ok=True)
    record = {"stage": stage, "completed_at": time.time(), "implementation_sha256": implementation_fingerprint(), **value}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(target)
    print(json.dumps(record), flush=True)


def selftest(root):
    from .flight import FigsDynamics
    from .predictor import official_transformer
    dynamics = FigsDynamics(root)
    state = np.array([0., 0., -150., 0., 0., 0., 0., 0., 0., 1.])
    hovering = dynamics.step(state, [dynamics.hover, 0, 0, 0], 1.)
    np.testing.assert_allclose(hovering, state, atol=1e-9)
    turning = dynamics.step(state, [dynamics.hover, 0, .1, 0], 1.)
    if not np.linalg.norm(turning[:3] - state[:3]) > .05:
        raise RuntimeError("Applied pitch failed to change physical trajectory")
    model = official_transformer(root, 16, frames=2, dropout=0).eval()
    x = torch.randn(1, 32, 768)
    maximum_error = 0.
    for attention, _ in model.transformer.layers:
        with torch.no_grad():
            error = float((attention(x) - attention.native_forward(x)).abs().max())
        maximum_error = max(maximum_error, error)
    if maximum_error > 1e-5:
        raise RuntimeError("SDPA adapter is not numerically equivalent to native attention")
    receipt(root, "selftest", {"passed": True, "hover_drift_m": float(np.linalg.norm(hovering[:3] - state[:3])),
             "pitch_displacement_m": float(np.linalg.norm(turning[:3] - state[:3])), "attention_max_error": maximum_error,
             "dynamics_reference": "actual FiGS equations; RK4 adaptation, native IRK/controller gate still separate"})


def build(root):
    import gsplat.cuda._backend as backend
    if backend._C is None:
        raise RuntimeError("Gaussian CUDA extension is unavailable")
    receipt(root, "build", {"passed": True, "extension": str(backend._C.__file__),
                           "installation": "official prebuilt pt21cu118 wheel; source fallback is limited to one compiler"})


def render(root):
    from .flight import FigsDynamics
    from .scene import CityScene
    import imageio.v2 as imageio
    scene, dynamics = CityScene(root), FigsDynamics(root)
    output = root / "integration/flight"
    output.mkdir(parents=True, exist_ok=True)
    c2w, intrinsics = scene.camera(2810)
    start = np.array([0., 0., -350., 5., 0., 0., 0., 0., 0., 1.])
    state = start.copy()
    records = []
    video = imageio.get_writer(str(output / "action_rollout.mp4"), fps=10, codec="libx264", macro_block_size=2)
    try:
        for frame in range(24):
            command = np.array([dynamics.hover, 0., .035 if frame < 12 else -.035, .08])
            camera = dynamics.camera_pose(state, c2w, start, 1.)
            observed = scene.render(camera, intrinsics)
            if observed["valid_fraction"] < .5:
                raise RuntimeError("Insufficient reconstructed depth support for mapping")
            Image.fromarray(observed["rgb"]).save(output / f"{frame:04d}.png")
            np.savez_compressed(output / f"{frame:04d}.npz", depth=observed["depth"], c2w=camera,
                                intrinsics=intrinsics, state=state, command=command, timestamp=frame * .1)
            video.append_data(observed["rgb"])
            records.append({"frame": frame, "timestamp": frame * .1, "state": state.tolist(), "command": command.tolist(),
                            "render_gaussians": observed["render_gaussians"], "valid_fraction": observed["valid_fraction"]})
            print(json.dumps(records[-1]), flush=True)
            state = dynamics.step(state, command, .1)
    finally:
        video.close()
    (output / "trajectory.json").write_text(json.dumps(records, indent=2))
    receipt(root, "render", {"passed": True, **scene.receipt, "frames": len(records), "video": str(output / "action_rollout.mp4"),
                            "flight_displacement_m": float(np.linalg.norm(state[:3] - start[:3])),
                            "collision_claim": "none: released splats are not collision mesh ground truth"})


def vision(root):
    from .vision import DinoVision
    source = root / "integration/flight"
    frames = sorted(source.glob("*.png"))
    if not frames:
        raise RuntimeError("Actual scene rollout is required before the visual-model gate")
    encoder = DinoVision(root)
    output = root / "integration/features"
    output.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        features = encoder.encode(Image.open(frame))
        torch.save(features, output / (frame.stem + ".pt"))
        print(json.dumps({"frame": frame.name, "shape": list(features.shape)}), flush=True)
    reference = encoder.encode(Image.open(frames[0]), side=224)
    receipt(root, "vision", {"passed": True, **encoder.receipt, "frames": len(frames), "patch_shape": list(features.shape),
                            "reference_interface_shape": list(reference.shape), "native_benchmark_reproduction": False})


def mapping(root):
    from .mapping import ObservedGaussianMap, spatial_hierarchy
    from .contracts import Observation
    source = root / "integration/flight"
    mapper = ObservedGaussianMap(root)
    history = []
    output = root / "integration/memory"
    output.mkdir(parents=True, exist_ok=True)
    # Only the prefix is mapped. Later frames remain held-out training targets.
    for frame in range(8):
        values = np.load(source / f"{frame:04d}.npz")
        observed = Observation("integration_prefix", frame, float(values["timestamp"]),
            np.asarray(Image.open(source / f"{frame:04d}.png").convert("RGB")),
            values["depth"].astype(np.float32), values["c2w"], values["intrinsics"])
        result = mapper.update(observed, iterations=10)
        history.append(result)
        print(json.dumps(result), flush=True)
        mapper.save(output / f"prefix_{frame:04d}.pt")
    features, geometry, fingerprint = spatial_hierarchy(output / "prefix_0007.pt", root / "integration/features")
    torch.save({"features": features, "geometry": geometry, "fingerprint": fingerprint, "last_observed_frame": 7}, output / "hierarchy.pt")
    receipt(root, "mapping", {"passed": True, "observed_frames": list(range(8)), "gaussians": len(mapper.support_frame),
                             "hierarchy_candidates": len(features), "fingerprint": fingerprint, "optimization": history,
                             "adaptation": "SplaTAM backprojection/insertion/objective structure; gsplat expected-depth, relative aerial depth loss, supplied poses; no native tracking claim"})


def modelcheck(root):
    from .predictor import MissionWorldModel, prediction_loss
    prefix = torch.load(root / "integration/memory/hierarchy.pt", weights_only=False)
    if prefix["last_observed_frame"] != 7:
        raise RuntimeError("Training label prefix mismatch")
    target_frame = 17
    target_features = torch.load(root / f"integration/features/{target_frame:04d}.pt", weights_only=True)
    future = np.load(root / f"integration/flight/{target_frame:04d}.npz")
    present = np.load(root / "integration/flight/0007.npz")
    actions = np.stack([np.load(root / f"integration/flight/{i:04d}.npz")["command"] for i in range(7, 17)])
    geometry = prefix["geometry"].clone()
    geometry[:, :3] = (geometry[:, :3] - torch.from_numpy(present["c2w"][:3, 3]).float()) / 100
    geometry[:, 3] /= 100
    ego = torch.zeros(1, 16)
    ego[0, :10] = torch.from_numpy(present["state"]).float()
    ego[0, :3] /= 100
    ego[0, 10:16] = torch.from_numpy(present["c2w"][:3, :2].copy()).float().flatten()
    camera = np.linalg.inv(present["c2w"]) @ future["c2w"]
    camera[:3, 3] /= 100
    valid = torch.nn.functional.interpolate(torch.from_numpy((future["depth"] > 0).astype(np.float32))[None, None], size=(32, 32), mode="area")[0, 0].flatten() > .9
    # Diagnostic zero mission is NOT a grounded-language experiment. This gate
    # establishes real-data optimization and is deliberately reported as such.
    batch = dict(memory=prefix["features"][None].cuda(), geometry=geometry[None].cuda(), mission=torch.zeros(1, 2048, device="cuda"),
                 ego=ego.cuda(), actions=torch.from_numpy(actions).float()[None].cuda(),
                 future_camera=torch.from_numpy(camera).float().flatten()[None].cuda(), horizon=torch.ones(1, device="cuda"))
    h, w = present["depth"].shape
    k = present["intrinsics"]
    batch["camera_intrinsics"] = torch.tensor([[k[0, 0]/w, k[1, 1]/h, k[0, 2]/w, k[1, 2]/h]], dtype=torch.float32, device="cuda")
    target = dict(features=target_features[None].cuda(), valid=valid[None].cuda(), coverage=valid[None].cuda(),
                  outcomes=torch.zeros(1, 3, device="cuda"))
    model = MissionWorldModel(root).cuda().train()
    # Mission outcomes/coverage are not trained using fabricated targets in this
    # integration check. Only measured future visual features drive this update.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    scaler = torch.cuda.amp.GradScaler()
    history = []
    tracked = model.dynamics.transformer.layers[0][0].to_qkv.weight
    before = tracked.detach().cpu().clone()
    selector_before = model.configurator.score.weight.detach().cpu().clone()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(**batch)
            error = (result["features"] - target["features"]).square().mean(-1)
            loss = (error * target["valid"]).sum() / target["valid"].sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite measured-future feature loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite model gradients")
        scaler.step(optimizer); scaler.update()
        history.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        print(json.dumps(history[-1]), flush=True)
    change = float((tracked.detach().cpu() - before).abs().max())
    selector_change = float((model.configurator.score.weight.detach().cpu() - selector_before).abs().max())
    if change == 0 or selector_change == 0:
        raise RuntimeError("Dynamics and hard-selection scoring must both receive a real update")
    output = root / "integration/model"
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        expected = model(**batch)["features"].float().cpu()
    checkpoint = dict(model={key: value.cpu() for key, value in model.state_dict().items()}, optimizer=optimizer.state_dict(),
                      scaler=scaler.state_dict(), step=len(history), torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), memory_fingerprint=prefix["fingerprint"],
                      description="real future DINO feature integration check; zero mission; not an efficacy experiment")
    torch.save(checkpoint, output / "checkpoint.pt")
    model.load_state_dict(torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)["model"], strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        restored = model(**batch)["features"].float().cpu()
    if not torch.equal(expected, restored):
        raise RuntimeError("Checkpoint round-trip changed deterministic predictions")
    receipt(root, "modelcheck", {"passed": True, "parameters": sum(p.numel() for p in model.parameters()),
            "history": history, "dynamics_max_update": change, "selector_max_update": selector_change, "checkpoint_roundtrip_equal": True,
            "prediction_shape": list(expected.shape), "target_frame": target_frame, "last_observed_frame": 7,
            "claim": "real-data trainability only; grounded mission, counterfactual outcome supervision and closed-loop efficacy NOT established"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--root", required=True)
    parser.add_argument("--strategy", choices=["full", "fixed", "configured"], default="configured")
    parser.add_argument("--budget", type=int, choices=[256, 512], default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--decisions", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    np.random.seed(7)
    torch.manual_seed(7)
    root = Path(args.root).resolve()
    # Capture inherited site packages as well as explicitly pinned dependencies.
    # The full package set matters because these venvs share a pre-existing base.
    from importlib.metadata import distributions
    import hashlib
    import platform
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       torch=torch.__version__, torch_cuda=torch.version.cuda,
                       packages=sorted({(d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"]}))
    serialized = json.dumps(environment, indent=2)
    environment_path = root / "environment-receipts" / (hashlib.sha256(serialized.encode()).hexdigest() + ".json")
    environment_path.parent.mkdir(exist_ok=True)
    if not environment_path.exists():
        environment_path.write_text(serialized)
    print(json.dumps({"environment_receipt": str(environment_path)}), flush=True)
    stages = {"build": build, "selftest": selftest, "render": render, "vision": vision, "mapping": mapping, "modelcheck": modelcheck}
    if args.stage == "grounding":
        from .grounding import grounding_gate
        stages["grounding"] = grounding_gate
    if args.stage in ("collect", "encode_dataset", "map_dataset", "ground_dataset"):
        from . import dataset
        stages[args.stage] = getattr(dataset, args.stage)
    if args.stage in ("train", "cache_reference", "evaluate_offline"):
        from . import training
        if args.stage == "train":
            stages["train"] = lambda root: training.train(root, args.strategy, args.budget, args.seed, args.steps)
        elif args.stage == "evaluate_offline":
            stages["evaluate_offline"] = lambda root: training.evaluate_offline(root, args.strategy, args.budget, args.seed)
        else:
            stages["cache_reference"] = lambda root: training.cache_reference(root, args.seed)
    if args.stage == "evaluate_closed_loop":
        from .closed_loop import evaluate_closed_loop
        stages[args.stage] = lambda root: evaluate_closed_loop(root, args.strategy, args.budget, args.seed, args.decisions)
    if args.stage not in stages:
        raise RuntimeError(f"Stage {args.stage} is not implemented; no surrogate fallback")
    stages[args.stage](root)


if __name__ == "__main__":
    main()
