"""Deterministic Windows supervisor. No LLM agent, shell command generation or retries.

Run from Windows: py -3.10 research/mission_world_model/overnight.py --stage render
Each stage is an owned WSL process group, with logs, deadlines and stop markers.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("idea1_resource_backend", REPO / "scripts/idea1/job_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def linux_path(path):
    path = str(Path(path).resolve()).replace("\\", "/")
    return "/mnt/" + path[0].lower() + path[2:]


def code_fingerprint():
    digest = hashlib.sha256()
    files = sorted((REPO / "research/mission_world_model").rglob("*.py"))
    files += [REPO / "scripts/idea1/job_backend.py", REPO / "scripts/cuda-memory-guard/sitecustomize.py"]
    for path in files:
        digest.update(str(path.relative_to(REPO)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def stage_plan(mode):
    checks = ["build", "selftest", "render", "vision", "mapping", "modelcheck", "grounding"]
    if mode == "all":
        return [(name, name, []) for name in checks]
    if mode == "prepare":
        return [(name, name, []) for name in ("encode_dataset", "map_dataset", "ground_dataset")]
    if mode == "validate_training":
        common = ["--seed", "997"]  # Never mixed into the three study seeds.
        plan = [("train_full", "train", ["--strategy", "full", "--steps", "3", *common]),
                ("cache_reference", "cache_reference", common)]
        for budget in (256, 512):
            for strategy in ("fixed", "configured"):
                parameters = ["--strategy", strategy, "--budget", str(budget), *common]
                plan.append((f"train_{strategy}_k{budget}", "train", [*parameters, "--steps", "3"]))
        plan.append(("offline_configured", "evaluate_offline", ["--strategy", "configured", *common]))
        plan.append(("closed_loop_configured", "evaluate_closed_loop", ["--strategy", "configured", "--decisions", "2", *common]))
        return plan
    if mode != "overnight":
        return [(mode, mode, [])]
    plan = [(name, name, []) for name in checks + ["collect", "encode_dataset", "map_dataset", "ground_dataset"]]
    for seed in (0, 1, 2):
        common = ["--seed", str(seed)]
        plan.append((f"train_full_s{seed}", "train", ["--strategy", "full", "--steps", "2000", *common]))
        plan.append((f"cache_reference_s{seed}", "cache_reference", common))
        plan.append((f"offline_full_s{seed}", "evaluate_offline", ["--strategy", "full", *common]))
        for budget in (256, 512):
            for strategy in ("fixed", "configured"):
                parameters = ["--strategy", strategy, "--budget", str(budget), *common]
                plan.append((f"train_{strategy}_k{budget}_s{seed}", "train", [*parameters, "--steps", "2000"]))
                plan.append((f"offline_{strategy}_k{budget}_s{seed}", "evaluate_offline", parameters))
    # Establish all matched offline comparisons before expensive online search;
    # an early closed-loop failure must not consume the entire comparison night.
    for strategy, budget in (("configured", 256), ("fixed", 256), ("configured", 512), ("fixed", 512), ("full", 256)):
        parameters = ["--strategy", strategy, "--budget", str(budget), "--seed", "0"]
        plan.append((f"closed_loop_{strategy}_k{budget}_s0", "evaluate_closed_loop", parameters))
    return plan


def write_report(run, root, results, planned, status):
    digest = hashlib.sha256()
    for path in sorted((run / "source/mission_world").glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    version = digest.hexdigest()[:12]
    passed = sum(r.get("return_code") == 0 and not r.get("stop_reason") for r in results)
    peak = max((r.get("peak_total_gpu_mib", 0) for r in results), default=0)
    lines = ["# Idea 1 — run report", "", f"Status: **{status}**. Completed stages: {passed}/{planned}.", "",
             f"Peak sampled total GPU memory: {peak:.0f} MiB. Ceiling: 6553.6 MiB. Code snapshot: `{version}`.", "",
             "The experiment uses released models and a complete CityGaussian asset. Its observations/maps are isolated from counterfactual labels.", "",
             "## Evidence", "", "- [Action-driven city video](../../integration/flight/action_rollout.mp4)",
             "- [Stage/resource receipts](summary.json)", "- [Exact execution plan and code identity](plan.json)",
             "- [Executed source snapshot](source/)", ""]
    failures = [r for r in results if r.get("return_code") != 0 or r.get("stop_reason")]
    if failures:
        failure = failures[-1]
        lines.extend([f"Stopped at `{failure['stage']}`: {failure.get('stop_reason') or failure.get('error') or 'process failure'}.",
                      f"See [the exact log]({failure['stage']}/stdout.log).", ""])
    completed = {r["stage"] for r in results if r.get("return_code") == 0 and not r.get("stop_reason")}
    allowed = set()
    for name, stage, parameters in json.loads((run / "plan.json").read_text())["stages"]:
        if stage == "evaluate_offline" and name in completed:
            options = dict(zip(parameters[::2], parameters[1::2]))
            allowed.add((options.get("--strategy", "configured"), int(options.get("--budget", 256)), int(options.get("--seed", 0))))
    summaries = []
    for path in (root / "experiment/models" / version).glob("*/heldout_summary.json"):
        value = json.loads(path.read_text())
        if (value["strategy"], value["budget"], value["seed"]) in allowed:
            summaries.append(path)
    if summaries:
        lines.extend(["## Held-out evaluations completed in this run", "",
            "Seed 997 is a three-update execution diagnostic, excluded from the study. Navigation regret is metres; visibility regret is percentage points of previously observed target surface.", "",
            "| Arm | Tokens | Seed | Steps | Feature MSE | Geometry MSE | Unchanged-image MSE | Nav regret (m) | Visibility regret (pp) | Model seconds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for path in sorted(summaries):
            value = json.loads(path.read_text())
            tokens = "all hierarchy" if value["strategy"] == "full" else value["budget"]
            regrets = value["mission_action_regret"]
            nav = "n/a" if regrets["navigate"] is None else f"{50*regrets['navigate']:.3f}"
            visibility = "n/a" if regrets["inspect"] is None else f"{100*regrets['inspect']:.3f}"
            arm = f"[{value['strategy']}](../../experiment/models/{version}/{path.parent.name}/heldout_summary.json)"
            lines.append(f"| {arm} | {tokens} | {value['seed']} | {value['trained_steps']} | {value['mean_latent_mse']:.4f} | {value['mean_geometric_reprojection_latent_mse']:.4f} | {value['mean_persistence_latent_mse']:.4f} | {nav} | {visibility} | {value['mean_model_seconds']:.3f} |")
        lines.append("")
    lines.extend(["## What these results do not establish", "",
        "No novelty, paper-level result, collision safety, real-world transfer or native-paper benchmark reproduction is established. "
        "MatrixCity is a synthetic Unreal Engine 5 benchmark reconstruction, not a real-world city capture (https://city-super.github.io/matrixcity/). "
        "Pose and reconstructed depth are privileged. Language targets are VLM bounding boxes. The data are from one city; held-out episodes can share image content. "
        "Navigation distance progress also admits an analytic geometry baseline: outperforming a learned fixed-token control alone is not evidence that a world model is necessary.", "",
        "OpenFly native evaluation, native SplaTAM tracking, native FiGS controller validation and independent collision geometry remain promotion gates, not completed baselines.", ""])
    temporary = run / "REPORT.partial.md"
    temporary.write_text("\n".join(lines), encoding="utf-8"); temporary.replace(run / "REPORT.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-diagnostic", action="store_true", help="Explicitly reproduce the historical privileged DINO proxy; not the current RGB flight study")
    parser.add_argument("--root", default="D:/uav-research/idea1")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--stage", choices=["build", "selftest", "render", "vision", "mapping", "modelcheck", "grounding",
                       "collect", "encode_dataset", "map_dataset", "ground_dataset", "prepare", "validate_training", "train", "cache_reference", "evaluate_offline", "evaluate_closed_loop", "all", "overnight"], default="all")
    parser.add_argument("--run-id")
    parser.add_argument("--strategy", choices=["full", "fixed", "configured"], default="configured")
    parser.add_argument("--budget", type=int, choices=[256, 512], default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--decisions", type=int, default=12)
    args = parser.parse_args()
    if not args.legacy_diagnostic:
        parser.error("This is the superseded privileged DINO diagnostic. Use research/rgb_flight/run.py for the current study; historical reproduction requires --legacy-diagnostic.")
    if os.name != "nt":
        raise RuntimeError("Launch supervisor on Windows so host RAM/desktop GPU usage are monitored")
    if not 0 < args.hours <= 8:
        raise ValueError("Deadline must be within the approved eight-hour envelope")
    root = Path(args.root).resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError("Expected an existing, dedicated research data directory")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not all(c.isalnum() or c in "_-" for c in run_id):
        raise ValueError("Invalid run id")
    run = root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    source = run / "source"
    (source / "mission_world").mkdir(parents=True)
    (source / "guard").mkdir()
    for path in (REPO / "research/mission_world_model/src/mission_world").glob("*.py"):
        shutil.copy2(path, source / "mission_world" / path.name)
    shutil.copy2(REPO / "scripts/idea1/job_backend.py", source / "job_backend.py")
    shutil.copy2(REPO / "scripts/cuda-memory-guard/sitecustomize.py", source / "guard/sitecustomize.py")
    shutil.copy2(__file__, source / "supervisor.py")
    policy = dict(workspace=str(root), deadline_ms=int((time.time() + args.hours * 3600) * 1000),
                  cancel_file=str(run / "STOP"), guard_directory=str(source / "guard"),
                  vram_fraction=.8, burst_margin_mib=512, minimum_available_ram_gib=12)
    backend.save(run / "policy.json", policy)
    stages = stage_plan(args.stage)
    if args.stage in ("train", "cache_reference", "evaluate_offline", "evaluate_closed_loop"):
        parameters = ["--strategy", args.strategy, "--budget", str(args.budget), "--seed", str(args.seed)]
        if args.stage == "train":
            if not 1 <= args.steps <= 10000:
                raise ValueError("Training steps must be in [1, 10000]")
            parameters += ["--steps", str(args.steps)]
        if args.stage == "evaluate_closed_loop":
            parameters += ["--decisions", str(args.decisions)]
        stages = [(args.stage, args.stage, parameters)]
    fingerprint = code_fingerprint()
    backend.save(run / "plan.json", {"stages": stages, "source_sha256": fingerprint, "hours": args.hours})
    results = []
    run_status = "running"
    write_report(run, root, results, len(stages), run_status)
    print(json.dumps({"run": str(run), "stages": stages, "stop_file": str(run / "STOP")}), flush=True)
    try:
        for name, stage, parameters in stages:
            # Jobs execute this immutable run snapshot, not a live editable repo.
            job = run / name
            job.mkdir()
            request = dict(policy_path=str(run / "policy.json"), deadline_ms=policy["deadline_ms"],
                           kind="cpu" if stage in ("selftest", "build") else "gpu", executable=linux_path(root / ("envs/vlm/bin/python" if stage in ("grounding", "ground_dataset") else "envs/research/bin/python")),
                           args=["-m", "mission_world.pipeline", stage, "--root", linux_path(root), *parameters],
                           source_directory=linux_path(source), cwd=".", timeout_seconds=2400)
            backend.save(job / "request.json", request)
            child = subprocess.Popen(["wsl.exe", "-d", "Ubuntu", "--exec", "python3",
                                      linux_path(source / "job_backend.py"), "--request", linux_path(job / "request.json")],
                                      creationflags=0x08000000)
            while child.poll() is None:
                reason = None
                if backend.available_ram() < 12 * 1024**3:
                    reason = "Windows available RAM below 12 GiB reserve"
                if time.time() * 1000 > policy["deadline_ms"]:
                    reason = "Windows supervisor deadline"
                if reason:
                    (run / "STOP").write_text(reason)
                time.sleep(2)
            result_path = job / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "failed", "error": "backend exited without receipt"}
            results.append({"stage": name, **result})
            print(json.dumps(results[-1]), flush=True)
            backend.save(run / "summary.json", {"stages": results, "research_claim": "preliminary integration/offline comparisons only; no paper or closed-loop efficacy claim"})
            write_report(run, root, results, len(stages), run_status)
            if child.returncode or result.get("return_code") != 0:
                run_status = "stopped on a failed gate"
                print(f"STOPPED: inspect {job / 'stdout.log'}", flush=True)
                return 1
        run_status = "finished planned stages"
        return 0
    except KeyboardInterrupt:
        run_status = "cancelled by operator"
        (run / "STOP").write_text("Operator interrupted supervisor")
        return 130
    finally:
        # Backends enforce their own deadline even if this supervisor disappears.
        if "child" in locals() and child.poll() is None:
            (run / "STOP").write_text("Supervisor exited")
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                registry = job / "process.json"
                if registry.exists():
                    subprocess.run(["wsl.exe", "-d", "Ubuntu", "--exec", "python3", linux_path(source / "job_backend.py"),
                                    "--stop", linux_path(registry)], timeout=20, creationflags=0x08000000)
        if run_status == "running":
            run_status = "stopped before completion"
        write_report(run, root, results, len(stages), run_status)


if __name__ == "__main__":
    sys.exit(main())
