"""Download only the selected pinned scene; resume bytes and verify upstream hash."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import preflight
from run import load_backend, resource_guard, save


def extract_verified(archive, destination, guard):
    """Extract only contained regular files; record the exact extracted bytes."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe scene ZIP member")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            import hashlib
            digest = hashlib.sha256()
            with source.open(entry) as src, temporary.open("wb") as out:
                checked = 0
                while True:
                    if time.monotonic() - checked > 1:
                        guard()
                        checked = time.monotonic()
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
            temporary.replace(target)
            records.append({"path": entry.filename, "bytes": entry.file_size, "sha256": digest.hexdigest()})
    save(destination / "EXTRACTED.json", records)
    return records


def acquire(root, run, guard):
    receipt = preflight.probe(root, guard)
    save(run / "scene_access.json", receipt)
    if receipt["status"] != "accessible" or not receipt["published_sha256"]:
        raise RuntimeError("Authorized scene access and published SHA-256 are required")
    folder = root / "assets/openfly" / receipt["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "env_airsim_16.zip"
    if destination.exists():
        verified = preflight.verify_local_archive(destination, receipt["published_sha256"], guard)
        return {**receipt, "archive": verified, "status": "verified"}
    partial = destination.with_suffix(".zip.partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > receipt["archive_bytes"]:
        raise RuntimeError("Partial download larger than pinned archive; inspect manually")
    if offset < receipt["archive_bytes"]:
        opener = urllib.request.build_opener(preflight.SafeRedirect())
        response = None
        for _, token in preflight.credentials(root):
            guard()
            req = urllib.request.Request(receipt["source_url"], headers={"Authorization": "Bearer " + token,
                                           "Range": f"bytes={offset}-"})
            try:
                response = opener.open(req, timeout=20)
                break
            except urllib.error.HTTPError as error:
                code = error.code
                error.close()
                if code not in (401, 403):
                    raise RuntimeError(f"Download HTTP status {code}") from None
        if response is None:
            raise RuntimeError("Download rejected available credentials")
        with response:
            expected_range = f"bytes {offset}-"
            if response.status == 206 and not response.headers.get("Content-Range", "").startswith(expected_range):
                raise RuntimeError("Download range mismatch")
            if response.status not in (200, 206) or (offset and response.status != 206):
                raise RuntimeError("Server did not honor resumable range; partial preserved")
            last_check = 0
            with partial.open("ab") as out:
                while True:
                    if time.monotonic() - last_check > 1:
                        guard()
                        last_check = time.monotonic()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if out.tell() + len(chunk) > receipt["archive_bytes"]:
                        raise RuntimeError("Download exceeds pinned size")
                    out.write(chunk)
    if partial.stat().st_size != receipt["archive_bytes"]:
        raise RuntimeError("Incomplete download; rerun to resume verified byte range")
    verified = preflight.verify_local_archive(partial, receipt["published_sha256"], guard)
    partial.replace(destination)
    verified["path"] = str(destination.resolve())
    return {**receipt, "archive": verified, "status": "verified", "downloaded": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("D:/uav-research/idea1"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("Research root must exist")
    run = root / "runs" / ("rgb-acquire-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True, exist_ok=False)
    history = []
    backend = load_backend()
    deadline = time.monotonic() + 8 * 3600
    guard = lambda: resource_guard(backend, root, run, deadline, history)
    print(json.dumps({"run": str(run)}), flush=True)
    try:
        guard()
        receipt = acquire(root, run, guard)
        save(run / "asset.json", receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        save(run / "resources.json", history)


if __name__ == "__main__":
    main()
