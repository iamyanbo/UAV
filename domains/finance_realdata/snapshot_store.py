"""Content-addressed snapshot storage and conservative retention.

Snapshot directories keep their paths, bytes and hashes. Identical table files
are stored once under objects/ and hard-linked into every snapshot containing
them, so readers see ordinary files and the storage budget counts bytes once.

Retention removes only snapshots nothing depends on. A snapshot is kept when a
task, canonical evaluation, trial, shadow prediction, evidence bundle, paper plan
or current pointer in any local profile names it, when it is the newest snapshot
of a UTC day inside the retention window, or when it is recent. A removed
snapshot leaves its manifest in retired/ and a line in retired/retention-log.jsonl.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage_budget import load_policy, locked

SNAPSHOT_ID = re.compile(r"DATA-\d{8}T\d{6}\d*Z-[0-9a-f]{10}")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def configured_data_root(project: Path, config: dict) -> Path:
    """Mirror data_pipeline.configured_data_root without importing its providers."""
    configured = os.environ.get("AR_FINANCE_DATA_ROOT", "").strip() or str(config.get("data_root", ""))
    if configured:
        base = Path(configured)
    else:
        state = os.environ.get("CURI_STATE_DIR", "").strip()
        base = (Path(state) if Path(state).is_absolute() else project / state) if state else project / ".curi"
        base = base / "data"
    return base.resolve() / config["id"]


def object_path(data_root: Path, digest: str, suffix: str) -> Path:
    return data_root / "objects" / digest[:2] / f"{digest}{suffix}"


def same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def link_into_store(data_root: Path, path: Path, digest: str, size: int, *, verify: bool = True) -> str:
    """Make path a hard link to its content object. The path and bytes do not change."""
    target = object_path(data_root, digest, path.suffix)
    if target.exists() and same_file(target, path):
        return "already"
    if path.stat().st_size != size or (verify and sha256_file(path) != digest):
        return "mismatch"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(path, target)
            return "stored"
        except OSError:
            return "unsupported"
    if target.stat().st_size != size:
        return "mismatch"
    temporary = path.with_name(f".{path.name}.link-{os.getpid()}")
    try:
        os.link(target, temporary)
        os.replace(temporary, path)
        return "linked"
    except OSError:  # a per-file link limit, or a reader holding the file open
        temporary.unlink(missing_ok=True)
        return "busy"


def snapshot_dirs(data_root: Path) -> list[Path]:
    root = data_root / "snapshots"
    return sorted(path for path in root.glob("DATA-*") if path.is_dir()) if root.exists() else []


def read_manifest(snapshot: Path) -> dict | None:
    try:
        return json.loads((snapshot / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


def dedupe(data_root: Path, *, verify: bool = True) -> dict:
    counts: Counter = Counter()
    for snapshot in snapshot_dirs(data_root):
        manifest = read_manifest(snapshot)
        if not manifest:
            counts["unreadable_manifest"] += 1
            continue
        for item in manifest.get("files", []):
            path = snapshot / item["path"]
            if not path.is_file():
                counts["missing"] += 1
                continue
            counts[link_into_store(data_root, path, item["sha256"], int(item["bytes"]), verify=verify)] += 1
    return dict(counts)


def state_dirs(project: Path, additional_projects=()) -> list[Path]:
    dirs = [path for pattern in (".curi*", ".autoresearch*") for path in project.glob(pattern) if path.is_dir()]
    for value in additional_projects:
        dirs.extend(path for path in Path(value).glob(".curi*") if path.is_dir())
    return sorted(set(dirs))


def read_only(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=30)


def referenced_snapshots(project: Path, additional_projects=()) -> dict[str, set[str]]:
    references: dict[str, set[str]] = {}

    def note(text: object, why: str) -> None:
        for snapshot_id in SNAPSHOT_ID.findall(str(text or "")):
            references.setdefault(snapshot_id, set()).add(why)

    for state in state_dirs(project, additional_projects):
        database = state / "research.sqlite"
        if database.is_file():
            connection = read_only(database)
            try:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table, column in (("task_data_snapshots", "snapshot_id"), ("quant_evaluations", "snapshot_id"),
                                      ("shadow_predictions", "snapshot_id"), ("quant_trials", "details_json")):
                    if table in tables:
                        for (value,) in connection.execute(f"SELECT {column} FROM {table}"):
                            note(value, f"{state.name}:{table}")
            finally:
                connection.close()
        for pointer in (state / "data").glob("*/current.json"):
            note(pointer.read_text("utf-8"), f"{state.name}:current")
        paper = state / "trading" / "paper.sqlite"
        if paper.is_file():
            connection = read_only(paper)
            try:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='plans'").fetchone():
                    for (payload,) in connection.execute("SELECT payload FROM plans"):
                        note(payload, f"{state.name}:paper_plan")
            finally:
                connection.close()
        evidence = state / "evidence"
        if evidence.is_dir():
            for manifest in evidence.rglob("manifest.json"):
                note(manifest.read_text("utf-8", errors="replace"), f"{state.name}:evidence")
    return references


def snapshot_time(snapshot_id: str) -> datetime | None:
    match = re.match(r"DATA-(\d{8})T(\d{6})", snapshot_id)
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) if match else None


def plan_retention(data_root: Path, references: dict[str, set[str]], *, now: datetime,
                   keep_days: int, grace_hours: int) -> dict:
    times = {path.name: snapshot_time(path.name) for path in snapshot_dirs(data_root)}
    times = {key: value for key, value in times.items() if value is not None}
    keep: dict[str, list[str]] = {key: sorted(references[key]) for key in times if key in references}
    if times:
        keep.setdefault(max(times, key=lambda key: times[key]), []).append("newest")
    daily: dict[str, str] = {}
    for key, at in sorted(times.items(), key=lambda item: item[1]):
        if now - at <= timedelta(hours=grace_hours):
            keep.setdefault(key, []).append("recent")
        if now - at <= timedelta(days=keep_days):
            daily[at.date().isoformat()] = key
    for key in daily.values():
        keep.setdefault(key, []).append("daily")
    return {"keep": keep, "delete": sorted(key for key in times if key not in keep)}


def apply_retention(data_root: Path, plan: dict, *, now: datetime) -> dict:
    retired = data_root / "retired"
    retired.mkdir(parents=True, exist_ok=True)
    removed, logical = 0, 0
    with (retired / "retention-log.jsonl").open("a", encoding="utf-8") as log:
        for snapshot_id in plan["delete"]:
            snapshot = data_root / "snapshots" / snapshot_id
            if not snapshot.is_dir():
                continue
            manifest = snapshot / "manifest.json"
            text = manifest.read_text("utf-8") if manifest.exists() else "{}"
            # Manifests carry long provenance lists; gzip keeps them whole at a fraction of the size.
            (retired / f"{snapshot_id}.manifest.json.gz").write_bytes(gzip.compress(text.encode("utf-8")))
            size = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
            shutil.rmtree(snapshot)
            try:
                content_hash = json.loads(text).get("content_hash")
            except ValueError:
                content_hash = None
            log.write(json.dumps({"snapshot_id": snapshot_id, "content_hash": content_hash,
                                  "retired_at": now.isoformat().replace("+00:00", "Z"), "logical_bytes": size,
                                  "reason": "unreferenced and outside the daily retention window"}) + "\n")
            removed += 1
            logical += size
    return {"removed": removed, "logical_bytes_removed": logical}


def collect_orphan_objects(data_root: Path) -> int:
    root = data_root / "objects"
    removed = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_nlink <= 1:
                path.unlink()
                removed += 1
    return removed


def maintain(project: Path, config: dict, *, apply: bool, keep_days: int | None = None,
             grace_hours: int | None = None, verify: bool = True, now: datetime | None = None) -> dict:
    now = now or utc_now()
    settings = config.get("retention", {})
    keep_days = int(keep_days if keep_days is not None else settings.get("keep_daily_days", 30))
    grace_hours = int(grace_hours if grace_hours is not None else settings.get("grace_hours", 48))
    data_root = configured_data_root(project, config)
    try:
        additional = load_policy(project)[1].get("additional_projects", [])
    except RuntimeError:
        additional = []
    references = referenced_snapshots(project, additional)
    with locked(project):
        plan = plan_retention(data_root, references, now=now, keep_days=keep_days, grace_hours=grace_hours)
        result = {"data_root": str(data_root), "snapshots": len(plan["keep"]) + len(plan["delete"]),
                  "keep": len(plan["keep"]), "delete": len(plan["delete"]), "applied": apply,
                  "keep_daily_days": keep_days, "grace_hours": grace_hours,
                  "kept_reasons": dict(Counter(reason.split(":")[-1] for reasons in plan["keep"].values() for reason in reasons))}
        if apply:
            result["retention"] = apply_retention(data_root, plan, now=now)
    if apply:
        # Links replace files atomically, so hashing need not hold the budget
        # lock; orphan collection afterwards does.
        result["dedupe"] = dedupe(data_root, verify=verify)
        with locked(project):
            result["orphan_objects_removed"] = collect_orphan_objects(data_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "maintain", "dedupe"])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-days", type=int)
    parser.add_argument("--grace-hours", type=int)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    config = json.loads(Path(args.manifest).read_text("utf-8"))
    if args.action == "dedupe":
        result = dedupe(configured_data_root(project, config), verify=not args.no_verify)
    else:
        result = maintain(project, config, apply=args.action == "maintain" and args.apply,
                          keep_days=args.keep_days, grace_hours=args.grace_hours, verify=not args.no_verify)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
