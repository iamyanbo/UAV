"""Simulated action-driven evaluation with a mid-flight mission intervention.

No collision mesh is available: log proximity/support aborts, never certify
collision-free navigation. Language targets are VLM boxes, not human annotations.
"""
import json
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import torch
from .contracts import Observation
from .dataset import observed_target_points, observed_surface_visibility, save_observation
from .flight import FigsDynamics
from .mapping import ObservedGaussianMap, spatial_hierarchy
from .planning import CEMPlanner, SearchBudget
from .predictor import MissionWorldModel
from .scene import CityScene
from .training import checkpoint_path, model_identity
from .vision import DinoVision
from .upstream import sha256


def evaluate_closed_loop(root, strategy, budget=256, seed=0, decisions=12):
    if not 2 <= decisions <= 24:
        raise ValueError("Expected 2–24 closed-loop decisions")
    root = Path(root)
    base = root / "experiment"
    manifest = json.loads((base / "dataset.json").read_text())
    checkpoint = checkpoint_path(root, strategy, budget, seed)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["identity"] != model_identity(root, strategy, budget, seed):
        raise RuntimeError("Closed-loop checkpoint identity mismatch")
    checkpoint_hash = sha256(checkpoint)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    model = MissionWorldModel(root, strategy=strategy, budget=budget).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    encoder, dynamics, scene = DinoVision(root), FigsDynamics(root), CityScene(root)
    episodes = [e for e in manifest["episodes"] if e["split"] == "test"][:2]
    summaries = []
    for entry in episodes:
        initial_frame = entry["prefixes"][0]["frame"]
        source = base / entry["id"]
        output = checkpoint.parent / "closed_loop" / attempt / entry["id"]
        output.mkdir(parents=True, exist_ok=False)
        observations = output / "observations"; observations.mkdir()
        for frame in range(initial_frame+1):
            shutil.copy2(source / "observations" / f"{frame:04d}.pt", observations / f"{frame:04d}.pt")
        mapper = ObservedGaussianMap.restore(root, source / "memory" / f"prefix_{initial_frame:04d}.pt")
        missions = {kind: torch.load(source / "missions" / f"{initial_frame:04d}_{kind}.pt", weights_only=False)
                    for kind in ("navigate", "inspect")}
        if any(m["mission"]["unresolved"] for m in missions.values()):
            summaries.append(dict(episode=entry["id"], status="unresolved_reference", decisions=0))
            continue
        present = np.load(source / "observations" / f"{initial_frame:04d}.npz")
        state = present["state"].copy()
        camera, intrinsics = present["c2w"].copy(), present["intrinsics"].copy()
        timestamp = float(present["timestamp"])
        initial_features = torch.load(source / "observations" / f"{initial_frame:04d}.pt", weights_only=True)
        goal = {}
        for kind, mission in missions.items():
            bbox = mission["mission"]["bbox_xyxy"]
            points = observed_target_points(present, bbox)
            if len(points) < 16:
                raise RuntimeError("Unusable observed reference geometry")
            yy, xx = np.mgrid[0:32, 0:32]
            mask = ((xx+.5)/32 >= bbox[0]) & ((xx+.5)/32 < bbox[2]) & ((yy+.5)/32 >= bbox[1]) & ((yy+.5)/32 < bbox[3])
            if not mask.any():
                raise RuntimeError("Grounding box is smaller than a visual patch")
            goal[kind] = dict(points=points, center=np.median(points, axis=0), extent=np.std(points, axis=0),
                              visual=initial_features[torch.from_numpy(mask.flatten())].mean(0).cuda()[None],
                              embedding=mission["embedding"].cuda()[None])
        planner = CEMPlanner(dynamics, model, SearchBudget(), seed=seed)
        trajectory = []
        map_path = output / "current_map.pt"
        with (output / "trajectory.jsonl").open("w", buffering=1) as log:
            for decision in range(decisions):
                before_fingerprint = mapper.fingerprint()
                kind = "navigate" if decision < decisions//2 else "inspect"
                mapper.save(map_path)
                memory, geometry, fingerprint = spatial_hierarchy(map_path, observations)
                geometry[:, :3] = (geometry[:, :3]-torch.from_numpy(camera[:3, 3]).float())/100
                geometry[:, 3] /= 100
                geometry[:, 5] = timestamp-geometry[:, 5]
                ego = torch.zeros(1, 16, device="cuda")
                ego[0, :10] = torch.from_numpy(state).float().cuda(); ego[0, :3] /= 100
                ego[0, 10:16] = torch.from_numpy(camera[:3, :2].copy()).float().flatten().cuda()
                target = goal[kind]
                anchor = np.concatenate([(target["center"]-camera[:3, 3])/100, target["extent"]/100, [kind == "navigate", kind == "inspect"]])
                torch.cuda.synchronize(); started = time.perf_counter()
                commands, search = planner.choose(memory.cuda()[None], geometry.cuda()[None], target["embedding"], ego, state, camera,
                    target["visual"], torch.from_numpy(anchor).float().cuda()[None],
                    torch.tensor([[intrinsics[0, 0]/scene.width, intrinsics[1, 1]/scene.height, intrinsics[0, 2]/scene.width, intrinsics[1, 2]/scene.height]], device="cuda", dtype=torch.float32))
                torch.cuda.synchronize(); planning_seconds = time.perf_counter()-started
                if mapper.fingerprint() != before_fingerprint:
                    raise RuntimeError("Planning/mission configuration modified persistent memory")
                previous_state, previous_camera = state.copy(), camera.copy()
                # Recede by one second, tracking five physical .2s segments.
                for command in commands[:5]:
                    state = dynamics.step(state, command, .2)
                camera = dynamics.camera_pose(state, previous_camera, previous_state, 1.)
                timestamp += 1.
                observed = scene.render(camera, intrinsics)
                valid_depth = observed["depth"][observed["depth"] > 0]
                minimum_depth = float(valid_depth.min()) if len(valid_depth) else None
                stop = "insufficient_reconstruction_support" if observed["valid_fraction"] < .6 else "observed_proximity_abort" if minimum_depth is not None and minimum_depth < 1. else None
                frame = initial_frame+decision+1
                save_observation(observations / f"{frame:04d}", observed, state, commands[:5], timestamp)
                torch.save(encoder.encode(observed["rgb"]), observations / f"{frame:04d}.pt")
                record = dict(episode=entry["id"], decision=decision, mission=kind, mission_switched=decision == decisions//2,
                              memory_before=before_fingerprint, configuration_preserved_memory=True,
                              state=state.tolist(), applied_commands=commands[:5].tolist(), planning_seconds=planning_seconds,
                              distance_to_observed_reference_m=float(np.linalg.norm(target["center"]-camera[:3, 3])),
                              observed_reference_visibility=observed_surface_visibility(target["points"], observed),
                              minimum_rendered_depth_m=minimum_depth, valid_fraction=observed["valid_fraction"], stop_reason=stop, search=search)
                trajectory.append(record); log.write(json.dumps(record)+"\n"); print(json.dumps(record), flush=True)
                if stop:
                    break
                mapper.update(Observation(entry["id"], frame, timestamp, observed["rgb"], observed["depth"].astype(np.float32), camera, intrinsics), iterations=10)
        summaries.append(dict(episode=entry["id"], decisions=len(trajectory), stop_reason=trajectory[-1]["stop_reason"] if trajectory else "no_decisions",
                              mean_planning_seconds=float(np.mean([t["planning_seconds"] for t in trajectory])),
                              collision_ground_truth=False, semantic_target_ground_truth="VLM bbox only"))
    report = dict(strategy=strategy, budget=budget, seed=seed, episodes=summaries, decisions_requested=decisions,
                  checkpoint_sha256=checkpoint_hash, attempt=attempt,
                  claim="action-driven simulated mission-switch evaluation; no collision certification, human goal annotations, or real-time claim")
    serialized = json.dumps(report, indent=2)
    (checkpoint.parent / "closed_loop" / attempt / "summary.json").write_text(serialized)
    (checkpoint.parent / "closed_loop_summary.json").write_text(serialized)
    from .pipeline import receipt
    receipt(root, f"closed_loop_{strategy}_{budget}_{seed}", {"passed": True, **report})
