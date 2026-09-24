"""Load pinned implementations without modifying the upstream working trees."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_revision(path, expected):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Upstream revision changed: {path}: {actual} != {expected}")
    # Tracked modifications also invalidate reference provenance; untracked build files do not.
    if subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"], text=True):
        raise RuntimeError(f"Upstream tracked sources were modified: {path}")
    return actual


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()
