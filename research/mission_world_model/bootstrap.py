"""Pinned code and data acquisition. Never installs or trains on import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

REPOSITORIES = {
    "dinov2": ("facebookresearch/dinov2", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"),
    "dino_wm": ("gaoyuezhou/dino_wm", "0a9492fa12044b852ae9e001cc74604b79c8bb0c"),
    "figs": ("StanfordMSL/FiGS", "11ad36ccf294c8423b54e4e7263a03e6a246f1b1"),
    "sousvide": ("StanfordMSL/SousVide", "a2400aa6e100c81b388bd105fdbe153710479004"),
    "splatam": ("spla-tam/SplaTAM", "da6bbcd24c248dc884ac7f49d62e91b841b26ccc"),
    "citygaussian": ("Linketic/CityGaussian", "e16c526f78ab01d377b13b3e6862dcc93ddbe953"),
}
CITY_REVISION = "4f73714cf50bdefe47ff8012e5c7ec71246adb9a"
CITY_FILES = {"MatrixCityAerial_1.zip": 5247615677, "MatrixCityAeiral_2.zip": 4068367043,
              "Building.zip": 6105357998, "Residence.zip": 4983198527,
              "Rubble.zip": 4478077459, "SciArt.zip": 851114125}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def acquire(url, target, expected_bytes=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_bytes is None or target.stat().st_size == expected_bytes):
        return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest(target), "source": url}
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "uav-mission-research/0.1", **({"Range": f"bytes={offset}-"} if offset else {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as output:
                    last_report = time.monotonic()
                    for block in iter(lambda: response.read(4 * 1024**2), b""):
                        output.write(block)
                        if time.monotonic() - last_report > 15:
                            print(json.dumps({"download": target.name, "bytes": output.tell()}), flush=True)
                            last_report = time.monotonic()
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise RuntimeError(f"Incorrect download length for {target.name}")
            partial.replace(target)
            return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest(target), "source": url}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def checkout(root, name):
    repo, revision = REPOSITORIES[name]
    path = root / "upstream" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/" + repo, str(path)], check=True)
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    # --no-checkout leaves HEAD valid but the working tree empty.
    needs_checkout = not any(item.name != ".git" for item in path.iterdir())
    if actual.stdout.strip() != revision or needs_checkout:
        changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True)
        # A brand-new --no-checkout clone presents tracked deletions; never reset user modifications.
        if changes and any(not line.startswith("D ") for line in changes.splitlines()):
            raise RuntimeError("Refusing to overwrite modified upstream checkout: " + str(path))
        subprocess.run(["git", "checkout", "--detach", revision], cwd=path, check=True)
    return {"name": name, "source": "https://github.com/" + repo, "revision": revision, "path": str(path)}


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive member " + entry.filename)
        source.extractall(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["code", "dino", "city", "qwen", "extract", "environment"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--scene", choices=list(CITY_FILES), default="MatrixCityAerial_1.zip")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "environment":
        import torch
        if torch.__version__ != "2.1.0+cu118":
            raise RuntimeError("Research environment must retain tested PyTorch2.1.0+cu118, got " + torch.__version__)
        records = [{"torch": torch.__version__, "cuda_build": torch.version.cuda}]
    elif args.command == "code":
        records = [checkout(root, name) for name in REPOSITORIES]
    elif args.command == "dino":
        records = [acquire("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", root / "weights/dinov2_vitb14.pth")]
    elif args.command == "city":
        records = [acquire(f"https://huggingface.co/TeslaYang123/CityGaussian/resolve/{CITY_REVISION}/{args.scene}", root / "assets" / args.scene, CITY_FILES[args.scene])]
    elif args.command == "qwen":
        from huggingface_hub import snapshot_download
        target = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct-AWQ", revision="e7b623934290c5a4da0ee3c6e1e57bfb6b5abbf2",
                                   local_dir=root / "weights/qwen-vl", max_workers=2)
        records = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in Path(target).glob("*.safetensors")]
    else:
        safe_extract(root / "assets" / args.scene, root / "scenes" / args.scene.removesuffix(".zip"))
        records = [{"extracted": args.scene}]
    (root / "receipts").mkdir(exist_ok=True)
    (root / "receipts" / (args.command + ("-" + args.scene if args.command in ("city", "extract") else "") + ".json")).write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
