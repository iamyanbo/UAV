"""Versioning, immutable artifact identities and bounded local job execution."""
import hashlib
import json
import os
from pathlib import Path
import random
import time

SCHEMA = 'photo-map-model/v3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact escapes its package')
    return path


def config():
    return read(Path(__file__).with_name('campaign.json'))


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Window:
    def __init__(self, hours):
        if not 0 < hours <= 8:
            raise ValueError('Execution windows must be in (0,8] hours')
        self.deadline = time.monotonic() + hours * 3600

    def remaining(self):
        return self.deadline - time.monotonic() > 30


def reserve_memory():
    import psutil
    if psutil.virtual_memory().available < 16 * 1024**3:
        raise RuntimeError('Need 12 GiB available RAM plus 4 GiB checkpoint headroom')


class FlightLock:
    """One active collect/train/evaluate job per campaign workspace."""
    def __init__(self, root, kind):
        self.path = Path(root) / 'active-job.lock'
        self.kind = kind

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, json.dumps(dict(pid=os.getpid(), kind=self.kind)).encode())
        return self

    def __exit__(self, *args):
        os.close(self.fd)
        self.path.unlink()

