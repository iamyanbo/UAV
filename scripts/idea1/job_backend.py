"""Owned, foreground command groups for the bounded Idea 1 campaign.

This is resource/process containment, not an adversarial OS sandbox.
The policy lives outside the agent's writable workspace. No background jobs.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

CAMPAIGN = "idea1-mission-world-model-2026-09-21"


def native_path(value):
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        return "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_identity(pid):
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        try:
            created, ended, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(ended), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        # comm may contain spaces/parentheses; starttime is field 22.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + tail[19]
    except (OSError, IndexError):
        return None


def stop_group(record):
    pid = record.get("child_pid")
    if not pid or process_identity(pid) != record.get("identity"):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=15, creationflags=0x08000000)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
            time.sleep(0.3)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def gpu_usage():
    executable = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.name != "nt" else "nvidia-smi")
    result = subprocess.run([executable, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                            **({"creationflags": 0x08000000} if os.name == "nt" else {}))
    if result.returncode:
        raise RuntimeError("GPU accounting unavailable: " + result.stderr[:200])
    total, used = [float(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
    return total, used


def available_ram():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]
        info = Memory()
        info.length = ctypes.sizeof(info)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
            raise RuntimeError("RAM accounting unavailable")
        return info.available
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


def allocation_fraction(total, used, ceiling, margin):
    if not 0 < ceiling <= .8 or margin < 512 or total <= 0 or used < 0:
        raise ValueError("Invalid approved GPU resource policy")
    headroom = total * ceiling - used - margin
    if used / total >= .75 or headroom < 256:
        raise RuntimeError(f"GPU admission denied: used={used} MiB, remaining allocation={headroom} MiB")
    return min(.8, headroom / total)


def run(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8-sig"))
    policy = json.loads(Path(native_path(request["policy_path"])).read_text(encoding="utf-8-sig"))
    if request["kind"] not in ("cpu", "gpu", "download"):
        raise ValueError("Unrecognized job resource kind")
    if not 0 < policy["vram_fraction"] <= .8 or policy["minimum_available_ram_gib"] < 12:
        raise ValueError("Policy exceeds approval or removes the RAM reserve")
    root = Path(native_path(policy["workspace"])).resolve()
    cwd = (root / request.get("cwd", ".")).resolve()
    if not cwd.is_relative_to(root) or not cwd.is_dir():
        raise RuntimeError("Command working directory escapes workspace or does not exist")
    executable = request["executable"]
    if not isinstance(request["args"], list) or not all(isinstance(a, str) for a in request["args"]):
        raise RuntimeError("Arguments must be literal strings")
    if executable.lower().split("/")[-1] in ("ssh", "ssh.exe", "shutdown", "reboot", "wsl", "wsl.exe"):
        raise RuntimeError("Remote/system lifecycle commands are not campaign jobs")
    job = Path(request_path).parent
    registry = job / "process.json"
    deadline = min(policy["deadline_ms"] / 1000, request["deadline_ms"] / 1000,
                   time.time() + request["timeout_seconds"])
    cancel_paths = [job / "CANCEL", Path(native_path(policy["cancel_file"]))]
    if time.time() >= deadline or any(p.exists() for p in cancel_paths):
        raise RuntimeError("Cancelled or expired before launch")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AR_", "SPARK_", "ALPACA_", "APCA_", "OPENAI_", "ANTHROPIC_", "CURI_"))}
    env.update({"CURI_APPROVED_CAMPAIGN": CAMPAIGN, "CURI_GPU_MEMORY_GUARD": "1",
                "PYTHONPATH": native_path(policy["guard_directory"]),
                "HF_HOME": str(root / "cache/huggingface"), "TORCH_HOME": str(root / "cache/torch"),
                "PIP_CACHE_DIR": str(root / "cache/pip"), "XDG_CACHE_HOME": str(root / "cache/xdg"),
                "TMPDIR": str(root / "tmp"), "TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
                "CONDA_PKGS_DIRS": str(root / "cache/conda"), "MAMBA_ROOT_PREFIX": str(root / "envs/mamba"),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "MAX_JOBS": "1", "CMAKE_BUILD_PARALLEL_LEVEL": "1", "WANDB_MODE": "disabled",
                "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    env["PYTHONFAULTHANDLER"] = "1"
    if request.get("source_directory"):
        source_directory = Path(native_path(request["source_directory"])).resolve()
        if not source_directory.is_relative_to(root) or not source_directory.is_dir():
            raise RuntimeError("Code snapshot escapes the dedicated research workspace")
        env["PYTHONPATH"] += os.pathsep + str(source_directory)
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_EXTENSIONS_DIR"] = str(root / "cache/torch_extensions")
    env["TORCH_CUDA_ARCH_LIST"] = "8.6"
    env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    if os.name != "nt":
        env["CC"] = "/usr/bin/gcc-10"
        env["CXX"] = "/usr/bin/g++-10"
    for folder in ["cache", "tmp", "envs"]:
        (root / folder).mkdir(exist_ok=True)
    if request["kind"] == "gpu":
        total, used = gpu_usage()
        env["CURI_MAX_VRAM_FRACTION"] = str(allocation_fraction(total, used, policy["vram_fraction"], policy["burst_margin_mib"]))
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CURI_MAX_VRAM_FRACTION"] = "0.6"
    minimum_ram = policy["minimum_available_ram_gib"] * 1024**3
    if available_ram() < minimum_ram:
        raise RuntimeError("RAM admission denied")
    output_path = job / "stdout.log"
    started = time.time()
    child = None
    gpu_lock = None
    if request["kind"] == "gpu" and os.name != "nt":
        import fcntl
        gpu_lock = (root / ".gpu-job.lock").open("a+")
        try:
            fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            gpu_lock.close()
            raise RuntimeError("Another owned GPU stage is active; concurrent GPU jobs are forbidden")
    record = {"backend_pid": os.getpid(), "kind": request["kind"], "started_at": started,
              "allocation_fraction": float(env["CURI_MAX_VRAM_FRACTION"]), "status": "starting"}
    try:
        with output_path.open("w", encoding="utf-8") as output:
            child = subprocess.Popen([executable, *request["args"]], cwd=cwd, env=env,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     **({"creationflags": 0x08000000 | 0x00000200} if os.name == "nt"
                                        else {"start_new_session": True, "pass_fds": (gpu_lock.fileno(),) if gpu_lock else ()}))
            record.update(child_pid=child.pid, identity=process_identity(child.pid), status="running")
            save(registry, record)
            reason = None
            peak_gpu = 0
            while child.poll() is None:
                if any(p.exists() for p in cancel_paths):
                    reason = "cancelled"
                elif time.time() >= deadline:
                    reason = "deadline"
                elif available_ram() < minimum_ram:
                    reason = "RAM reserve breached"
                if request["kind"] == "gpu":
                    total, used = gpu_usage()
                    peak_gpu = max(peak_gpu, used)
                    if used / total >= policy["vram_fraction"]:
                        reason = "total VRAM ceiling reached"
                for drive, reserve in [("C:/", 30), ("D:/", 80)]:
                    target = native_path(drive)
                    if Path(target).exists() and shutil.disk_usage(target).free < reserve * 1024**3:
                        reason = f"{drive} free-space reserve breached"
                if output_path.stat().st_size > 64 * 1024**2:
                    reason = "job log exceeded 64 MiB"
                if reason:
                    stop_group(record)
                    break
                time.sleep(1)
            return_code = child.wait(timeout=20)
            # Reap detached grandchildren even if the original shell exits first.
            if os.name != "nt":
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            record.update(status="finished", return_code=return_code, stop_reason=reason,
                          peak_total_gpu_mib=peak_gpu, duration_seconds=time.time() - started)
            save(registry, record)
            save(job / "result.json", record)
            return 0 if return_code == 0 and reason is None else 1
    finally:
        if child and child.poll() is None:
            stop_group(record)
        if gpu_lock:
            gpu_lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--stop")
    args = parser.parse_args()
    try:
        if args.stop:
            record = json.loads(Path(args.stop).read_text())
            print(json.dumps({"stopped": stop_group(record)}))
        else:
            sys.exit(run(args.request))
    except Exception as error:
        if args.request:
            save(Path(args.request).parent / "result.json", {"status": "failed", "error": str(error), "return_code": 1})
        print(str(error), file=sys.stderr)
        sys.exit(1)
