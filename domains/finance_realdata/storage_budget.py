"""Shared storage accounting and admission control for the local pipeline.

The budget measures disk bytes (NTFS compressed size on Windows), counting hard
links once. Transparent compression preserves paths, hashes and reproducibility.
It never deletes evidence or mutates SQLite content. Reservations serialize
admission across cooperating pipeline processes; this is not an OS disk quota.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

HARD_LIMIT = 40_000_000_000
HEADROOM = 1_000_000_000
COMPRESSION_RETRY_SECONDS = 30 * 60

# Reuse the Windows API binding across the scan instead of loading a DLL for
# every file. This monitor runs alongside live research and acquisition.
if os.name == "nt":
    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _compressed_size = _kernel.GetCompressedFileSizeW
    _compressed_size.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    _compressed_size.restype = ctypes.c_ulong


def allocated_bytes(path: Path, stat=None) -> int:
    stat = stat or path.stat()
    if os.name == "nt":
        high = ctypes.c_ulong()
        ctypes.set_last_error(0)
        low = _compressed_size(str(path.absolute()), ctypes.byref(high))
        if low == 0xFFFFFFFF and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        return (high.value << 32) | low
    return stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size


def files_under(root: Path):
    # Do not follow junctions/symlinks into other projects or volumes.
    def scan_error(error):
        if not isinstance(error, FileNotFoundError):
            raise error  # unreadable storage is unknown, not zero bytes

    for parent, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
        children = []
        for name in dirs:
            child = Path(parent, name)
            try:
                stat = child.lstat()
                if not child.is_symlink() and not (getattr(stat, "st_file_attributes", 0) & 0x400):
                    children.append(name)
            except FileNotFoundError:
                pass  # a concurrent task or acquisition finished its temporary directory
        dirs[:] = children
        for name in files:
            path = Path(parent, name)
            if not path.is_symlink():
                yield path


def roots_for(project: Path, config: dict) -> list[Path]:
    roots = [Path(x).resolve() for x in config.get("managed_roots", [])]
    # All profiles count, including archived profiles and their workspaces.
    roots.extend(p.resolve() for p in project.glob(".curi*") if p.is_dir())
    for value in config.get("additional_projects", []):
        roots.extend(p.resolve() for p in Path(value).glob(".curi*") if p.is_dir())
    unique = []
    for root in sorted(set(roots), key=lambda p: len(p.parts)):
        if root == Path(root.anchor) or root == project or root == Path.home():
            raise ValueError(f"storage root is too broad: {root}")
        if not any(root == p or p in root.parents for p in unique):
            unique.append(root)
    return unique


def measure(roots: list[Path]) -> dict:
    seen = set()
    rows = []
    for root in roots:
        logical = allocated = count = 0
        for path in files_under(root):
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                allocated += allocated_bytes(path, stat)
                logical += stat.st_size
                count += 1
            except FileNotFoundError:  # a concurrently finished temporary file
                continue
        rows.append(dict(path=str(root), allocated_bytes=allocated, logical_bytes=logical, files=count))
    return dict(allocated_bytes=sum(r["allocated_bytes"] for r in rows),
                logical_bytes=sum(r["logical_bytes"] for r in rows), roots=rows)


def load_policy(project: Path) -> tuple[Path, dict]:
    path = project / "storage-policy.json"
    if not path.exists():
        raise RuntimeError("storage-policy.json is required for shared budget enforcement")
    config = json.loads(path.read_text("utf-8"))
    limit = int(config.get("max_bytes", HARD_LIMIT))
    if not 0 < limit <= HARD_LIMIT:
        raise ValueError("storage budget must be positive and cannot exceed 40 GB")
    return path, config


@contextlib.contextmanager
def locked(project: Path):
    directory = project / ".curi-storage"
    directory.mkdir(exist_ok=True)
    lock_path = directory / "budget.lock"
    try:
        with lock_path.open("xb") as initial:
            initial.write(b"0")
    except FileExistsError:
        pass
    # Windows byte-range locks prohibit reading the locked byte as well as
    # writing it. Open without reading and let the retry loop acquire it.
    handle = lock_path.open("r+b")
    deadline = time.monotonic() + 300
    while True:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                raise RuntimeError("storage budget lock is busy")
            time.sleep(0.1)
    try:
        yield directory
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def compress(roots: list[Path]) -> list[dict]:
    if os.name != "nt":
        return [{"supported": False, "reason": "Transparent NTFS compression requires Windows"}]
    results = []
    for root in roots:
        if not root.is_dir():
            continue
        # /C also marks directories so newly created descendants inherit
        # compression. /I continues past locked active files; the subsequent
        # measurement remains authoritative. No /F: already compressed files
        # need not be recompressed on every maintenance pass.
        result = subprocess.run(["compact.exe", "/C", f"/S:{root}", "/I", "/Q"],
                                cwd=root, capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        results.append(dict(path=str(root), exit_code=result.returncode,
                            detail=(result.stdout + result.stderr)[-1000:]))
    return results


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: protect it
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return not ok or code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def check(project: Path, reserve_bytes: int = 0, compact: bool = False,
          owner_pid: int | None = None, monitor: bool = False) -> dict:
    if reserve_bytes < 0:
        raise ValueError("reservation must not be negative")
    _, config = load_policy(project)
    limit = int(config.get("max_bytes", HARD_LIMIT))
    with locked(project) as state:
        lease_file = state / "reservations.json"
        leases = json.loads(lease_file.read_text("utf-8")) if lease_file.exists() else {}
        leases = {k: v for k, v in leases.items() if process_alive(int(v["pid"]))}
        reserved = sum(v["bytes"] for v in leases.values())
        roots = roots_for(project, config)
        result = measure(roots)
        prior_status = state / "status.json"
        last_compression = 0
        if prior_status.exists():
            last_compression = json.loads(prior_status.read_text("utf-8")).get("compression_checked_at", 0)
        pressure = result["allocated_bytes"] + reserved + reserve_bytes + HEADROOM > limit
        if compact or (not monitor and pressure and time.time() - last_compression >= COMPRESSION_RETRY_SECONDS):
            result["compression"] = compress(roots)
            result.update(measure(roots))
            last_compression = time.time()
        result["compression_checked_at"] = last_compression
        result.update(max_bytes=limit, reserve_bytes=reserve_bytes, reserved_bytes=reserved,
                      headroom_bytes=HEADROOM, checked_at=time.time())
        result["free_bytes"] = max(0, limit - result["allocated_bytes"])
        result["available_bytes"] = max(0, limit - result["allocated_bytes"] - reserved - HEADROOM)
        # Admission accounts for promised future writes and operating headroom.
        # An already admitted study is stopped only by measured exhaustion.
        result["exhausted"] = result["allocated_bytes"] >= limit
        result["admitted"] = result["allocated_bytes"] + reserved + reserve_bytes + HEADROOM <= limit
        result["state"] = "ready" if result["admitted"] else "storage_pressure"
        if owner_pid is not None and result["admitted"]:
            token = uuid.uuid4().hex
            leases[token] = dict(pid=owner_pid, bytes=reserve_bytes, created_at=time.time())
            result["reservation"] = token
        temporary_leases = state / "reservations.tmp"
        temporary_leases.write_text(json.dumps(leases), "utf-8")
        temporary_leases.replace(lease_file)
        temporary = state / "status.tmp"
        temporary.write_text(json.dumps(result, indent=2), "utf-8")
        temporary.replace(state / "status.json")
        return result


def require_capacity(project: Path, reserve_bytes: int = 0) -> dict:
    result = check(project, reserve_bytes)
    if not result["admitted"]:
        raise RuntimeError("Shared 40 GB storage budget exhausted; acquisition is paused. "
                           "Narrow the data request or compact storage; do not raise the cap.")
    return result


def release(project: Path, token: str):
    with locked(project) as state:
        path = state / "reservations.json"
        leases = json.loads(path.read_text("utf-8")) if path.exists() else {}
        leases.pop(token, None)
        temporary = state / "reservations.tmp"
        temporary.write_text(json.dumps(leases), "utf-8")
        temporary.replace(path)


@contextlib.contextmanager
def reservation(project: Path, size: int):
    result = check(project, size, owner_pid=os.getpid())
    if not result["admitted"]:
        raise RuntimeError("Shared 40 GB storage budget cannot accommodate this operation")
    try:
        yield result
    finally:
        release(project, result["reservation"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "observe", "maintain", "check", "reserve", "release"])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--token")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    if args.action == "release":
        release(project, args.token or "")
        print('{}')
        return
    if args.action == "reserve" and not args.owner_pid:
        parser.error("reserve requires --owner-pid")
    try:
        result = check(project, args.reserve_bytes, compact=args.action == "maintain",
                       owner_pid=args.owner_pid if args.action == "reserve" else None,
                       monitor=args.action == "observe")
    except Exception as error:
        # Retain the real diagnostic; callers must not label every process,
        # permission, lock or filesystem failure as exhausted capacity.
        print(json.dumps(dict(state="unknown", checked_at=time.time(),
                              error=f"{type(error).__name__}: {error}")))
        raise SystemExit(1)
    print(json.dumps(result))
    if args.action not in {"status", "observe"} and not result["admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
