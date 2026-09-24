"""Cross-process Spark leases. Locks serialize admission, not job execution."""
import fcntl
import json
import os
from pathlib import Path

GIB = 1024 ** 3


class ResourceLease:
    def __init__(self, root, job, kind, peak_gib, artifact=None):
        self.folder = Path(root) / 'admission'
        self.folder.mkdir(exist_ok=True)
        self.path = self.folder / (Path(job).name + '.json')
        self.record = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                           job=str(job), kind=kind, peak_bytes=int(peak_gib * GIB), artifact=artifact)
        self.held = False

    def acquire(self, available_bytes):
        with (self.folder / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = []
            for path in self.folder.glob('*.json'):
                entry = json.loads(path.read_text())
                alive = entry['boot_id'] == self.record['boot_id']
                try:
                    os.kill(entry['pid'], 0)
                except ProcessLookupError:
                    alive = False
                if alive:
                    active.append(entry)
                else:
                    path.unlink()
            kind = self.record['kind']
            if active and (kind == 'exclusive' or any(x['kind'] == 'exclusive' for x in active)):
                raise RuntimeError('An exclusive workload holds Spark admission')
            if active and (kind == 'flight' or any(x['kind'] == 'flight' for x in active)):
                raise RuntimeError('Live flight owns measured CPU/GPU timing; offline overlap is not profiled for flight')
            if kind == 'flight' and any(x['kind'] in ('flight', 'gpu') for x in active):
                raise RuntimeError('Flight requires exclusive GPU/simulator ownership')
            if kind == 'gpu' and (any(x['kind'] == 'flight' for x in active) or sum(x['kind'] == 'gpu' for x in active) >= 2):
                raise RuntimeError('Two offline GPU workers admitted, or live flight owns GPU')
            if self.record['artifact'] and any(x['artifact'] == self.record['artifact'] for x in active):
                raise RuntimeError('Another job owns the requested mutable artifact')
            # Deliberately conservative until per-job allocation profiles exist:
            # existing reservations are additional to observed current usage.
            reserved = sum(x['peak_bytes'] for x in active) + self.record['peak_bytes']
            if available_bytes - reserved < 16 * GIB:
                raise RuntimeError('Insufficient measured memory for reservations + 12 GiB floor + 4 GiB checkpoint headroom')
            self.path.write_text(json.dumps(self.record, indent=2))
            self.held = True

    def release(self):
        if self.held:
            with (self.folder / 'lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.path.unlink(missing_ok=True)
            self.held = False
