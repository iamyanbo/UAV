"""Fixed, paired-action supervised comparisons with resumable optimizer/RNG state."""
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .dataset import BranchDataset, processed_identity
from .predictor import MissionWorldModel, prediction_loss, projected_query_bias
from .upstream import sha256, implementation_fingerprint


def device_batch(samples):
    inputs = {key: torch.stack([item[0][key] for item in samples]).cuda() for key in samples[0][0]}
    targets = {key: torch.stack([item[1][key] for item in samples]).cuda() for key in samples[0][1]}
    return inputs, targets


def geometric_feature_prediction(inputs):
    """Training-free reprojection of observed fine-level DINO features.

    Uses the same known future camera and observed hierarchy as the learned
    reference. It neither reads a future image nor fills memory with simulator
    geometry. Unsupported rays receive a diffuse observed-feature mixture, not
    a claim of calibrated uncertainty or newly observed content.
    """
    geometry = inputs["geometry"]
    logits = projected_query_bias(geometry, inputs["ego"], inputs["future_camera"], inputs["camera_intrinsics"])[..., :-1]
    fine = geometry[:, :, 4] == geometry[:, :, 4].amin(1, keepdim=True)
    logits = logits.masked_fill(~fine[:, None], -torch.inf)
    return logits.softmax(-1) @ inputs["memory"]


def checkpoint_path(root, strategy, budget, seed):
    version = implementation_fingerprint()[:12]
    return Path(root) / "experiment/models" / version / f"{strategy}_k{budget}_s{seed}" / "checkpoint.pt"


def model_identity(root, strategy, budget, seed):
    identity = dict(strategy=strategy, budget=budget, seed=seed,
                dataset_sha256=sha256(Path(root) / "experiment/dataset.json"),
                processed_artifacts=processed_identity(root),
                backbone_sha256="0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
                training_objective="measured future features + measured mission outcome + paired outcome differences; student reference consistency",
                architecture_version=1, implementation_sha256=implementation_fingerprint())
    if strategy != "full":
        identity["reference_checkpoint_sha256"] = sha256(checkpoint_path(root, "full", 256, seed))
    return identity


def save_checkpoint(path, model, optimizer, scaler, step, identity):
    temporary = path.with_suffix(".partial.pt")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step,
                    identity=identity, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate()), temporary)
    temporary.replace(path)


def outcome_statistics(dataset):
    """Training-only unit normalization; never fit scalers on held-out episodes."""
    path = dataset.base / "outcome_statistics.json"
    identity = dict(dataset_sha256=sha256(dataset.base / "dataset.json"), processed_artifacts=processed_identity(dataset.base.parent),
                    normalizer_source_sha256=sha256(__file__))
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] == identity:
            return saved["missions"]
    values = {"navigate": [], "inspect": []}
    unknown = []
    for index in range(len(dataset)):
        _, target, metadata = dataset[index]
        values[metadata["mission"]].append(target["outcomes"].numpy())
        unknown.append(metadata["future_unobserved_fraction"])
        if (index+1) % 96 == 0:
            print(json.dumps({"normalizer_training_samples": index+1, "total": len(dataset)}), flush=True)
    result = {}
    for kind, records in values.items():
        if not records:
            raise RuntimeError("No grounded training examples for mission " + kind)
        array = np.stack(records)
        deviation = array.std(0)
        # Do not amplify nearly constant sensor/support noise into a major task.
        scale = np.where(deviation >= .001, deviation, 1.)
        result[kind] = dict(mean=array.mean(0).tolist(), scale=scale.tolist(), observed_std=deviation.tolist(), samples=len(array))
    path.write_text(json.dumps(dict(identity=identity, missions=result,
        mean_future_unobserved_fraction=float(np.mean([x for x in unknown if x is not None])),
        scope="training labels only; low-variance mission outcomes weaken the experiment, not evidence of a solved task"), indent=2))
    return result


def normalized_targets(targets, metadata, statistics):
    mean = targets["outcomes"].new_tensor([statistics[item["mission"]]["mean"] for item in metadata])
    scale = targets["outcomes"].new_tensor([statistics[item["mission"]]["scale"] for item in metadata])
    return {**targets, "outcomes": (targets["outcomes"]-mean)/scale}, mean, scale


def scaled_update(model, optimizer, scaler, objective, maximum_attempts=8):
    """One successful update; bounded AMP retries reuse the exact dropout RNG.

    A finite loss with scaled-gradient overflow is normal AMP scale adaptation,
    not an optimizer update. Nonfinite objectives or persistent overflow stop.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
    for attempt in range(maximum_attempts):
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite supervised training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if torch.isfinite(norm):
            scaler.step(optimizer); scaler.update()
            return loss.detach(), parts, norm.detach(), attempt
        previous_scale = scaler.get_scale()
        scaler.update()  # unscale_ recorded the overflow; no optimizer step.
        next_scale = scaler.get_scale()
        print(json.dumps({"precision_scale_backoff": attempt+1, "previous_scale": previous_scale,
                          "next_scale": next_scale, "optimizer_updated": False}), flush=True)
        if next_scale >= previous_scale or next_scale < 1:
            raise RuntimeError("Nonfinite gradients are not recoverable by bounded AMP scaling")
    raise RuntimeError("Persistent nonfinite gradients after bounded AMP backoff")


def train(root, strategy, budget=256, seed=0, steps=600):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dataset = BranchDataset(root, "train")
    statistics = outcome_statistics(dataset)
    identity = model_identity(root, strategy, budget, seed)
    path = checkpoint_path(root, strategy, budget, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    for index, (episode, frame, kind, branch) in enumerate(dataset.items):
        groups.setdefault((episode, frame, kind, branch["horizon"]), []).append(index)
    groups = [items for items in groups.values() if len(items) >= 2]
    if len(groups) < 16:
        raise RuntimeError("Too few real counterfactual groups for the training gate")
    model = MissionWorldModel(root, budget=budget, strategy=strategy).cuda().train()
    print(json.dumps({"training_arm": strategy, "budget": budget, "seed": seed, "requested_steps": steps,
                      "samples": len(dataset), "parameters": sum(p.numel() for p in model.parameters()),
                      "checkpoint": str(path)}), flush=True)
    if strategy != "full" and not path.exists():
        reference_path = checkpoint_path(root, "full", 256, seed)
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        if reference["identity"] != model_identity(root, "full", 256, seed):
            raise RuntimeError("Student initialization/reference identity mismatch")
        # Both compact arms inherit the exact same trained shared components.
        shared = {key: value for key, value in reference["model"].items()
                  if (not key.startswith("configurator.") or key.startswith("configurator.geometry.")) and key != "dynamics.pos_embedding"}
        model.load_state_dict(shared, strict=False)
        with torch.no_grad():
            model.dynamics.pos_embedding.copy_(reference["model"]["dynamics.pos_embedding"][:, :budget+1])
        del reference, shared
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, foreach=False)
    scaler = torch.cuda.amp.GradScaler()
    start = 0
    if path.exists():
        restored = torch.load(path, map_location="cpu", weights_only=False)
        if restored["identity"] != identity:
            raise RuntimeError("Resume identity mismatch; refusing incompatible checkpoint reuse")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scaler.load_state_dict(restored["scaler"])
        start = restored["step"]
        if start > steps:
            raise RuntimeError("Checkpoint already exceeds requested training steps; refusing a misleading shorter-run receipt")
        torch.set_rng_state(restored["torch_rng"])
        torch.cuda.set_rng_state_all(restored["cuda_rng"])
        np.random.set_state(restored["numpy_rng"]); random.setstate(restored["python_rng"])
        del restored
    reference_cache = None
    if strategy != "full":
        reference_path = checkpoint_path(root, "full", 256, seed)
        cached = torch.load(reference_path.parent / "reference_predictions.pt", weights_only=False)
        if cached["dataset_sha256"] != identity["dataset_sha256"]:
            raise RuntimeError("Teacher cache belongs to another dataset")
        if cached["reference_checkpoint_sha256"] != sha256(reference_path):
            raise RuntimeError("Teacher cache does not match the trained reference checkpoint")
        reference_cache = cached["outcomes"]
    begun = time.monotonic()
    last_checkpoint = begun
    with (path.parent / "metrics.jsonl").open("a", buffering=1) as log:
        for step in range(start, steps):
            indices = random.sample(random.choice(groups), 2)
            samples = [dataset[index] for index in indices]
            inputs, targets = device_batch(samples)
            targets, _, _ = normalized_targets(targets, [sample[2] for sample in samples], statistics)
            teacher = None if reference_cache is None else {"outcomes": reference_cache[indices].cuda()}
            def objective():
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(**inputs)
                    loss, parts = prediction_loss(prediction, targets, teacher)
                    predicted_difference = prediction["outcomes"][0]-prediction["outcomes"][1]
                    measured_difference = targets["outcomes"][0]-targets["outcomes"][1]
                    action_consequences = F.smooth_l1_loss(predicted_difference, measured_difference)
                    loss = loss + action_consequences
                    if teacher is not None:
                        loss = loss + .25 * F.smooth_l1_loss(predicted_difference, (teacher["outcomes"][0]-teacher["outcomes"][1]).detach())
                return loss, {**parts, "action_difference_loss": action_consequences.detach()}
            loss, parts, norm, backoffs = scaled_update(model, optimizer, scaler, objective)
            record = dict(step=step+1, total_loss=float(loss), precision_backoffs=backoffs, precision_scale=scaler.get_scale(),
                          gradient_norm=float(norm), elapsed_seconds=time.monotonic()-begun,
                          samples=[sample[2] for sample in samples], **{k: float(v) for k, v in parts.items()})
            log.write(json.dumps(record)+"\n")
            if (step+1) % 10 == 0:
                print(json.dumps({k: v for k, v in record.items() if k != "samples"}), flush=True)
            if step+1 == 5 or step+1 == steps or time.monotonic()-last_checkpoint >= 180:
                save_checkpoint(path, model, optimizer, scaler, step+1, identity)
                last_checkpoint = time.monotonic()
    from .pipeline import receipt
    receipt(Path(root), f"train_{strategy}_{budget}_{seed}", {"passed": True, **identity, "steps": steps,
             "checkpoint": str(path), "train_samples": len(dataset), "excluded_unresolved_missions": dataset.excluded_unresolved,
             "claim": "training completed; held-out effectiveness requires separate evaluation"})


@torch.inference_mode()
def cache_reference(root, seed=0):
    dataset = BranchDataset(root, "train")
    saved = torch.load(checkpoint_path(root, "full", 256, seed), map_location="cpu", weights_only=False)
    if saved["identity"] != model_identity(root, "full", 256, seed):
        raise RuntimeError("Reference cache/checkpoint identity mismatch")
    model = MissionWorldModel(root, strategy="full").cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    outcomes = []
    for index in range(len(dataset)):
        inputs, _ = device_batch([dataset[index]])
        with torch.autocast("cuda", dtype=torch.float16):
            outcomes.append(model(**inputs)["outcomes"][0].float().cpu())
        if (index+1) % 96 == 0:
            print(json.dumps({"cached_reference_samples": index+1, "total": len(dataset)}), flush=True)
    reference = checkpoint_path(root, "full", 256, seed)
    torch.save(dict(outcomes=torch.stack(outcomes), dataset_sha256=sha256(Path(root) / "experiment/dataset.json"),
                    reference_checkpoint_sha256=sha256(reference)), reference.parent / "reference_predictions.pt")
    from .pipeline import receipt
    receipt(Path(root), f"cache_reference_{seed}", {"passed": True, "samples": len(outcomes), "seed": seed})


@torch.inference_mode()
def evaluate_offline(root, strategy, budget=256, seed=0):
    dataset = BranchDataset(root, "test")
    statistics = json.loads((Path(root) / "experiment/outcome_statistics.json").read_text())["missions"]
    path = checkpoint_path(root, strategy, budget, seed)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = model_identity(root, strategy, budget, seed)
    if saved["identity"] != expected:
        raise RuntimeError("Evaluation/checkpoint identity mismatch")
    evaluated_steps = saved["step"]
    evaluated_checkpoint_sha256 = sha256(path)
    model = MissionWorldModel(root, strategy=strategy, budget=budget).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    records = []
    groups = {}
    with (path.parent / "heldout_predictions.jsonl").open("w", buffering=1) as output:
        for index in range(len(dataset)):
            sample = dataset[index]
            inputs, target = device_batch([sample])
            normalized, mean, scale = normalized_targets(target, [sample[2]], statistics)
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(**inputs)
                loss, parts = prediction_loss(prediction, normalized)
            torch.cuda.synchronize()
            record = dict(**sample[2], prediction=(prediction["outcomes"]*scale+mean)[0].float().cpu().tolist(),
                          target=target["outcomes"][0].cpu().tolist(), latent_mse=float(parts["latent"]),
                          model_seconds=time.perf_counter()-started,
                          selected_indices=prediction["selected_indices"][0].cpu().tolist())
            # Strong interpretability check: known camera motion plus the
            # observed target centroid exactly solves our navigation-distance
            # label. No learned world model is needed for that particular label.
            anchor = sample[0]["mission_anchor"][:3].numpy()
            columns = sample[0]["ego"][10:16].reshape(3, 2).numpy()
            rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
            motion = rotation @ sample[0]["future_camera"].reshape(4, 4)[:3, 3].numpy()
            record["analytic_navigation_progress"] = float((np.linalg.norm(anchor)-np.linalg.norm(anchor-motion))*2) if record["mission"] == "navigate" else None
            previous = torch.load(dataset.base / record["episode"] / "observations" / f"{record['frame']:04d}.pt", weights_only=True).cuda()
            persistence_error = (previous-target["features"][0]).square().mean(-1)
            record["persistence_latent_mse"] = float(persistence_error[target["valid"][0]].mean())
            geometric = geometric_feature_prediction(inputs)
            geometric_error = (geometric-target["features"]).square().mean(-1)
            record["geometric_reprojection_latent_mse"] = float(geometric_error[target["valid"]].mean())
            records.append(record); output.write(json.dumps(record)+"\n")
            group = (record["episode"], record["frame"], record["mission"], record["horizon"])
            groups.setdefault(group, []).append(record)
            if (index+1) % 96 == 0:
                print(json.dumps({"evaluated_heldout_samples": index+1, "total": len(dataset)}), flush=True)
    regrets = []
    mission_regrets = {"navigate": [], "inspect": []}
    analytic_navigation_regrets = []
    for items in groups.values():
        chosen = max(items, key=lambda x: x["prediction"][0])
        regret = max(x["target"][0] for x in items)-chosen["target"][0]
        regrets.append(regret); mission_regrets[chosen["mission"]].append(regret)
        if chosen["mission"] == "navigate":
            analytic = max(items, key=lambda x: x["analytic_navigation_progress"])
            analytic_navigation_regrets.append(max(x["target"][0] for x in items)-analytic["target"][0])
    paired_missions = {}
    for record in records:
        paired_missions.setdefault((record["episode"], record["frame"], record["branch"], record["horizon"]), {})[record["mission"]] = record
    overlaps = []
    for pair in paired_missions.values():
        if set(pair) != {"navigate", "inspect"}:
            continue
        if pair["navigate"]["memory_fingerprint"] != pair["inspect"]["memory_fingerprint"]:
            raise RuntimeError("Mission intervention changed the underlying memory")
        first, second = (set(pair[kind]["selected_indices"]) for kind in ("navigate", "inspect"))
        overlaps.append(len(first & second)/len(first | second))
    summary = dict(**expected, samples=len(records), counterfactual_groups=len(groups),
                   trained_steps=evaluated_steps, evaluated_checkpoint_sha256=evaluated_checkpoint_sha256,
                   mean_measured_action_regret=float(np.mean(regrets)), mean_latent_mse=float(np.mean([r["latent_mse"] for r in records])),
                   mean_model_seconds=float(np.mean([r["model_seconds"] for r in records])),
                   mean_persistence_latent_mse=float(np.mean([r["persistence_latent_mse"] for r in records])),
                   mean_geometric_reprojection_latent_mse=float(np.mean([r["geometric_reprojection_latent_mse"] for r in records])),
                   mission_action_regret={kind: float(np.mean(values)) if values else None for kind, values in mission_regrets.items()},
                   analytic_navigation_action_regret=float(np.mean(analytic_navigation_regrets)) if analytic_navigation_regrets else None,
                   matched_memory_mission_interventions=len(overlaps),
                   mean_selection_jaccard_after_mission_change=float(np.mean(overlaps)) if overlaps else None,
                   excluded_unresolved_missions=dataset.excluded_unresolved,
                   claim="offline, episode-held-out counterfactual evaluation only; not closed-loop navigation or paper evidence")
    (path.parent / "heldout_summary.json").write_text(json.dumps(summary, indent=2))
    from .pipeline import receipt
    receipt(Path(root), f"offline_{strategy}_{budget}_{seed}", {"passed": True, **summary})
