"""Mechanical task check for a runtime-bound finance snapshot.

This deliberately does not impose one finance evaluator or score. It verifies
only the frozen part of the contract: one staged snapshot exists, its files are
unchanged, and point-in-time tables expose non-null availability timestamps.
Each research question remains free to implement the evaluation it actually
needs in ordinary task files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root")
    args = parser.parse_args()
    roots = ([Path(args.snapshot_root) / "manifest.json"] if args.snapshot_root
             else list(Path(".research-data/finance-realdata").glob("*/manifest.json")))
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one bound data snapshot, found {len(roots)}")
    manifest_path = roots[0]
    manifest = json.loads(manifest_path.read_text("utf-8"))
    failures: list[str] = []
    for item in manifest["files"]:
        path = manifest_path.parent / item["path"]
        if not path.is_file():
            failures.append(f"missing {item['path']}")
            continue
        if path.stat().st_size != item["bytes"]:
            failures.append(f"size mismatch {item['path']}")
        if sha256_file(path) != item["sha256"]:
            failures.append(f"hash mismatch {item['path']}")
        available = item.get("availableAtField")
        if path.suffix == ".parquet" and available:
            frame = pd.read_parquet(path, columns=[available])
            if frame[available].isna().any():
                failures.append(f"null {available} values in {item['path']}")
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({
        "snapshot_id": manifest["snapshot_id"],
        "validation_state": manifest["validation_state"],
        "files": len(manifest["files"]),
        "status": "bound snapshot intact",
    }))


if __name__ == "__main__":
    main()
