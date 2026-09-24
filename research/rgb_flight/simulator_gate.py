"""Windows resource supervisor for a single real AirSim launch attempt."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run import HERE, REPO, load_backend, resource_guard, save


def linux(path):
    path=str(Path(path).resolve()).replace("\\","/")
    return "/mnt/"+path[0].lower()+path[2:]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("D:/uav-research/idea1"))
    parser.add_argument("--renderer",choices=("vulkan","opengl"),default="vulkan")
    parser.add_argument("--rpc-check",action="store_true")
    parser.add_argument("--display-mode",choices=("offscreen","xvfb"),default="offscreen")
    parser.add_argument("--graphics-driver",choices=("system","dozen","lavapipe","swiftshader"),default="system")
    parser.add_argument("--debug-start",action="store_true")
    parser.add_argument("--disable-rhi-thread",action="store_true")
    parser.add_argument("--validate-graphics",action="store_true")
    parser.add_argument("--parent-stop",type=Path)
    parser.add_argument("--deadline-ms",type=int)
    args=parser.parse_args()
    if os.name != "nt":
        parser.error("Launch from Windows for host resource monitoring")
    root=args.root.resolve()
    scene=root/"scenes/openfly/b051daff7afe74bcf695f8b72922b13386bc75ea/env_airsim_16/LinuxNoEditor"
    run=root/"runs"/("rgb-launch-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+args.renderer)
    run.mkdir(parents=True,exist_ok=False)
    source=run/"source"
    source.mkdir()
    shutil.copy2(HERE/"linux_probe.py",source/"linux_probe.py")
    shutil.copy2(HERE/"rpc_probe.py",source/"rpc_probe.py")
    shutil.copy2(HERE/"graphics_probe.py",source/"graphics_probe.py")
    shutil.copy2(REPO/"scripts/idea1/job_backend.py",source/"job_backend.py")
    shutil.copytree(REPO/"scripts/cuda-memory-guard",source/"guard",ignore=shutil.ignore_patterns("__pycache__"))
    deadline_ms=min(int((time.time()+240)*1000),args.deadline_ms or int((time.time()+240)*1000))
    policy=dict(workspace=str(root),deadline_ms=deadline_ms,cancel_file=str(run/"STOP"),
                guard_directory=str(source/"guard"),vram_fraction=.8,burst_margin_mib=512,minimum_available_ram_gib=12)
    save(run/"policy.json",policy)
    request=dict(policy_path=str(run/"policy.json"),deadline_ms=policy["deadline_ms"],kind="gpu",executable="/usr/bin/python3",
                 args=[linux(source/"linux_probe.py"),"--scene",linux(scene),"--output",linux(run),"--renderer",args.renderer,"--display-mode",args.display_mode,"--graphics-driver",args.graphics_driver],
                 source_directory=linux(source),cwd=".",timeout_seconds=220)
    if args.rpc_check:
        request["args"] += ["--rpc-python",linux(root/"envs/airsim/bin/python")]
    if args.debug_start:
        request["args"] += ["--debug-start"]
    if args.disable_rhi_thread:
        request["args"] += ["--disable-rhi-thread"]
    if args.validate_graphics:
        request["args"] += ["--validate-graphics"]
    save(run/"request.json",request)
    print(json.dumps({"run":str(run)}),flush=True)
    backend=load_backend()
    history=[]
    deadline=time.monotonic()+max(0,deadline_ms/1000-time.time())
    child=None
    try:
        resource_guard(backend,root,run,deadline,history)
        if args.parent_stop and args.parent_stop.exists():
            raise RuntimeError("Parent preflight cancelled before simulator launch")
        child=subprocess.Popen(["wsl.exe","-d","Ubuntu","--exec","python3",linux(source/"job_backend.py"),"--request",linux(run/"request.json")],creationflags=0x08000000)
        while child.poll() is None:
            try:
                if args.parent_stop and args.parent_stop.exists():
                    raise RuntimeError("Parent preflight cancelled")
                resource_guard(backend,root,run,deadline,history)
            except RuntimeError as error:
                (run/"STOP").write_text(str(error))
            time.sleep(1)
        result=json.loads((run/"result.json").read_text()) if (run/"result.json").exists() else {"status":"missing_backend_receipt"}
        print(json.dumps(result),flush=True)
        return child.returncode
    finally:
        if child is not None and child.poll() is None:
            (run/"STOP").write_text("Supervisor stopped")
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                subprocess.run(["wsl.exe","-d","Ubuntu","--exec","python3",linux(source/"job_backend.py"),"--stop",linux(run/"process.json")],timeout=20,creationflags=0x08000000)
        save(run/"host_resources.json",history)


if __name__ == "__main__":
    raise SystemExit(main())
