"""Deterministic entry point for the RGB-only continuous-flight adaptation.

Implemented jobs cover guarded acquisition, continuous reference-flight probes,
runtime isolation, native pretrained-model checks and lossless video archives.
Learning stages fail closed until their actual adapters and foundation evidence
exist. Legacy proxy results cannot satisfy this campaign's gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import preflight

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ("preflight", "build-obstacle-field", "generate-manifests", "capture-goals", "collect-expert",
          "collect-expert-campaign", "prepare-visual-dataset",
          "reconstruct", "encode", "train-odometry", "train-goal", "train-world", "train-policy",
          "train-configurator", "dagger", "ppo", "validate", "evaluate", "report")


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_backend():
    spec = importlib.util.spec_from_file_location("rgb_resource_backend", REPO / "scripts/idea1/job_backend.py")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def resource_guard(backend, root, run, deadline, history):
    if (run / "STOP").exists():
        raise RuntimeError("Operator STOP marker")
    if time.monotonic() >= deadline:
        raise RuntimeError("Job window expired")
    total, used = backend.gpu_usage()
    ram = backend.available_ram()
    sample = dict(time_utc=datetime.now(timezone.utc).isoformat(), total_gpu_mib=total,
                  used_gpu_mib=used, available_host_ram_bytes=ram)
    history.append(sample)
    if not (math.isfinite(total) and math.isfinite(used) and total > 0 and 0 <= used < .8 * total):
        raise RuntimeError("Total-device GPU usage reached 80% ceiling or accounting is invalid")
    if ram < 12 * 1024**3:
        raise RuntimeError("Available host RAM below 12 GiB reserve")
    for target, reserve in [(Path("C:/"), 30), (Path("D:/"), 80)]:
        if target.exists() and shutil.disk_usage(target).free < reserve * 1024**3:
            raise RuntimeError(f"Free-space reserve breached on {target}")


def write_report(run, state):
    save(run / "state.json", state)
    rows = ["# RGB-only OpenFly-scene continuous-flight adaptation", "",
            "Status: **" + state["status"] + "**. " + state["reason"], "",
            "This stage receipt alone does not establish a trained visual-goal navigation result.", "",
            "| Stage | Status |", "|---|---|"]
    rows += [f"| {name} | {value} |" for name, value in state["stages"].items()]
    rows += ["", "Next action: " + state["next_action"], "",
             "Resume command (starts a new immutable attempt):", "", "```powershell",
             state["next_command"], "```", "",
             "The access receipt, resource samples, source snapshot and configuration are adjacent to this report.",
             "Later stages remain unimplemented; dependency receipts alone do not implement or pass them.", ""]
    (run / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="preflight")
    parser.add_argument("--root", default="D:/uav-research/idea1")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--run-id")
    parser.add_argument("--backend",choices=("spark","windows"),default="spark")
    parser.add_argument('--preflight-step',choices=('diagnostics','models','reference','reference-campaign','archive','scale-prepare','scale'),default='diagnostics')
    parser.add_argument('--remote-observations',help='Absolute Spark observation-only directory for model/video checks')
    parser.add_argument('--remote-route-file',help='Absolute Spark engineering-only route proposal file')
    parser.add_argument('--route-id')
    parser.add_argument('--remote-campaign',help='Prior interrupted Stage-A survey directory, with matching source/configuration')
    parser.add_argument('--remote-import-campaign',help='Prior terminated visual-goal campaign to migrate with verified evidence')
    parser.add_argument('--remote-dataset',help='Training-only Spark dataset bundle with foundation and split receipts')
    parser.add_argument('--remote-obstacle-field',help='Privileged Spark obstacle-field NPZ')
    parser.add_argument('--remote-captures',help='Existing complete Spark depth/semantic survey to fuse offline')
    parser.add_argument('--remote-evaluator-labels',help='Privileged Spark split label JSON')
    parser.add_argument('--remote-goal',help='Coordinate-free Spark goal-observation directory')
    parser.add_argument('--episode-id',help='Episode identity from a deterministic manifest')
    parser.add_argument('--maximum-speed-mps',type=float,choices=(3.,4.5,6.),default=3.)
    parser.add_argument('--episode-limit',type=int,help='Bound a collection campaign to the first N manifest episodes')
    parser.add_argument('--goal-checkpoint',help='Relative frozen goal checkpoint path inside the training dataset')
    parser.add_argument('--resume',help='Relative checkpoint in the training bundle')
    parser.add_argument('--frames',type=int,default=0,help='Optional reconstruction prefix; zero processes the full recording')
    parser.add_argument('--asynchronous-map',action='store_true',help='Run mapping optimization in a bounded separate process')
    parser.add_argument('--tracking-optimizer',choices=['DSPO','DBA'],default='DSPO',help='Released tracking optimizer for reconstruction diagnosis')
    parser.add_argument('--clock-speed',type=float,default=1.,help='Continuous simulation clock scale in (0,1]; always recorded')
    parser.add_argument('--live-perception',action='store_true',help='Run isolated Splat mapping, V-JEPA and Qwen throughout reference flight')
    parser.add_argument("--local-archive", type=Path)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("Run on Windows to measure host RAM and total desktop GPU usage")
    if not math.isfinite(args.hours) or not 0 < args.hours <= 8:
        parser.error("Hours must be in (0, 8]")
    if bool(args.local_archive) != bool(args.archive_sha256):
        parser.error("Local archive and trusted SHA-256 must be supplied together")
    root = Path(args.root).resolve()
    if not root.is_dir() or root == Path(root.anchor):
        parser.error("Expected existing dedicated research data directory")
    if args.backend=="spark":
        if args.local_archive:
            parser.error("Spark uses its transferred checksum-verified scene; local archive import requires --backend windows")
        from spark_remote import execute
        return execute(args)
    run_id = args.run_id or ("rgb-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    if not run_id or not all(c.isalnum() or c in "_-" for c in run_id):
        parser.error("Invalid run ID")
    run = root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    source = run / "source"
    source.mkdir()
    files = list(HERE.glob("*.py")) + [HERE / "campaign.json", REPO / "scripts/idea1/job_backend.py"]
    hashes = {}
    for path in files:
        data = path.read_bytes()
        (source / path.name).write_bytes(data)
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    save(run / "source_hashes.json", hashes)
    state = dict(status="running", requested_stage=args.stage, reason="Checking scene prerequisites",
                 stages={s: "not_run" for s in STAGES},
                 next_action="Resolve preflight before collecting or training",
                 next_command='py -3.10 research/rgb_flight/run.py --stage preflight --root "' + str(root) + '" --hours 8')
    write_report(run, state)
    print(json.dumps({"run": str(run), "report": str(run / "REPORT.md"), "stop_marker": str(run / "STOP")}), flush=True)
    if args.stage == "report":
        prior = sorted((p for p in (root/"runs").glob("rgb-*/state.json") if p.parent != run), key=lambda p:p.stat().st_mtime)
        if prior:
            state = json.loads(prior[-1].read_text(encoding="utf-8"))
            state["source_report"] = str(prior[-1])
            state["requested_stage"] = "report"
            state["stages"]["report"] = "completed_existing_evidence_report"
        else:
            state.update(status="blocked", reason="No previous RGB flight gate evidence exists")
        write_report(run,state)
        print(json.dumps({"status":state["status"],"report":str(run/"REPORT.md")}),flush=True)
        return 0
    history = []
    deadline = time.monotonic() + args.hours * 3600
    backend = load_backend()
    guard = lambda: resource_guard(backend, root, run, deadline, history)
    try:
        guard()
        # Only read host/WSL capability; never launch an unvalidated executable.
        wsl = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--exec", "python3", "-c",
                              "import json,pathlib; print(json.dumps({'python':True,'mem_available_kib':next(int(x.split()[1]) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))}))"],
                             capture_output=True, text=True, timeout=30, creationflags=0x08000000)
        if wsl.returncode:
            raise RuntimeError("Ubuntu Python capability check failed")
        capability = json.loads(wsl.stdout)
        save(run / "wsl.json", capability)
        if capability["mem_available_kib"] < 12 * 1024**2:
            raise RuntimeError("Available Ubuntu RAM below 12 GiB reserve")
        if args.local_archive:
            archive = preflight.verify_local_archive(args.local_archive, args.archive_sha256, guard)
            receipt = dict(status="local_archive_verified", archive=archive, downloaded=False)
        else:
            receipt = preflight.probe(root, guard)
        save(run / "scene_access.json", receipt)
        guard()
        state["status"] = "blocked"
        state["stages"]["preflight"] = "blocked"
        if receipt["status"] == "blocked":
            state["reason"] = receipt["reason"]
            state["next_action"] = ("Enable download access for IPEC-COMMUNITY/OpenFly_DataGen and refresh the local HF login; "
                                    "or supply an authorized env_airsim_16.zip with a trusted SHA-256. Do not paste tokens into reports.")
        else:
            from acquire import acquire, extract_verified
            if args.local_archive:
                archive_path = args.local_archive
                revision = "local-" + args.archive_sha256[:12]
            else:
                receipt = acquire(root, run, guard)
                archive_path = Path(receipt["archive"]["path"])
                revision = receipt["revision"]
            save(run / "asset.json", receipt)
            if args.local_archive:
                state["reason"] = "Local archive checksum verified; match its provenance to the selected published scene before execution"
                state["next_action"] = "Verify archive identity against the published env_airsim_16 checksum and record its scene revision"
            else:
                scene_root = root / "scenes/openfly" / revision
                records = extract_verified(archive_path, scene_root, guard)
                save(run / "extracted.json", records)
                state["reason"] = "Scene verified and extracted; checking actual RGB and physics RPC"
                write_report(run,state)
                result = subprocess.run([sys.executable,str(HERE / "simulator_gate.py"),"--root",str(root),
                                         "--renderer","vulkan","--rpc-check","--parent-stop",str(run/"STOP"),
                                         "--deadline-ms",str(int((time.time()+max(0,deadline-time.monotonic()))*1000))],
                                         capture_output=True,text=True,timeout=min(270,max(1,deadline-time.monotonic())+30))
                (run / "simulator_gate.log").write_text(result.stdout+result.stderr,encoding="utf-8")
                first = next((line for line in result.stdout.splitlines() if line.startswith('{"run":')),None)
                if first:
                    sim_run=Path(json.loads(first)["run"])
                    state["simulator_run"]=str(sim_run)
                    rpc_path=sim_run/"engineering_only/rpc.json"
                    rpc=json.loads(rpc_path.read_text()) if rpc_path.exists() else {}
                    state["rpc_status"]=rpc.get("status","not_completed")
                    launch_path=sim_run/"launch.json"
                    launch=json.loads(launch_path.read_text()) if launch_path.exists() else {}
                    state["graphics_evidence"]=launch.get("details",{})
                else:
                    rpc={}
                if result.returncode or rpc.get("status")!="primitive_checks_only":
                    state["reason"]="Actual simulator gate failed: " + rpc.get("status","inspect simulator_gate.log")
                    if rpc.get("error_type"):
                        state["reason"] += "; " + rpc["error_type"] + ": " + rpc.get("error","")
                    if state.get("graphics_evidence",{}).get("status")=="blocked_graphics_capabilities":
                        missing=sorted({item for device in state["graphics_evidence"]["devices"] for item in device["missing"]})
                        state["reason"]="Selected scene requires unavailable Vulkan features: " + ", ".join(missing)
                    state["rpc_evidence"] = rpc
                    state["next_action"]="Resolve the measured executable/RGB/physics failure using the saved simulator evidence before collection or training"
                else:
                    state["reason"]="Primitive RPC checks completed; 20 full reference flights and RGB-only perception/resource feasibility remain unimplemented and unverified"
                    state["next_action"]="Implement route annotation and 20 varied physical reference flights, then validate causal Splat-SLAM tracking/scale and the selected perception resource profile"
        if args.stage not in ("all", "preflight", "report"):
            state["stages"][args.stage] = "blocked_by_preflight"
        state["stages"]["report"] = "completed_blocker_report"
        return 2
    except KeyboardInterrupt:
        state.update(status="cancelled", reason="Interrupted by operator")
        return 130
    except Exception as error:
        # Do not persist arbitrary network exception text or credential values.
        reason = str(error) if type(error) in (RuntimeError, ValueError) else type(error).__name__
        state.update(status="blocked", reason=reason)
        state["stages"]["preflight"] = "blocked"
        return 2
    finally:
        save(run / "resources.json", history)
        write_report(run, state)
        print(json.dumps({"status": state["status"], "reason": state["reason"], "report": str(run / "REPORT.md")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
