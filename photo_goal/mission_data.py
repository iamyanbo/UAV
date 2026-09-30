"""Content-addressed RGB, independent label sidecars and bounded feature cache."""
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
import io
import json
import numpy as np
import torch
from PIL import Image
from .common import digest, write, read, contained
from .mission_contracts import identity


class FeatureStore(MutableMapping):
    def __init__(self, root, cache_size=32, restore=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.restore=restore

    def _path(self, key):
        if not isinstance(key, int) or key < 0:
            raise ValueError('Invalid feature identity')
        return self.root/f'{key:09d}.pt'

    def __getitem__(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path=self._path(key)
        if not path.exists() and self.restore is not None:
            self[key]=self.restore(key)
        value = torch.load(path, map_location='cpu', weights_only=True)
        self.cache[key] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def __setitem__(self, key, value):
        path = self._path(key)
        if path.exists():
            raise ValueError('Feature identity is immutable')
        pending = path.with_suffix('.pending')
        torch.save(value.detach().cpu().half(), pending)
        pending.replace(path)

    def __delitem__(self, key):
        self._path(key).unlink()
        self.cache.pop(key, None)

    def __iter__(self):
        return (int(p.stem) for p in sorted(self.root.glob('*.pt')))

    def __len__(self):
        return sum(1 for _ in self.root.glob('*.pt'))

    def __contains__(self, key):
        return self._path(key).is_file()


class ShardWriter:
    def __init__(self, root, run_id, split='train'):
        self.root, self.run_id, self.split = Path(root), run_id, split
        if split not in ('train', 'development', 'sealed'):
            raise ValueError('Invalid split')
        for name in ('frames', 'records', 'labels'):
            (self.root/name).mkdir(parents=True, exist_ok=True)

    def image(self, path):
        with Image.open(path) as source:
            if source.size != (640, 480) or source.mode != 'RGB' or source.format != 'PNG':
                raise ValueError('Expected recorded full RGB frame')
        sha = digest(path)
        dest = self.root/'frames'/f'{sha}.png'
        if not dest.exists():
            import os
            import shutil
            tmp = dest.with_suffix('.pending')
            try:
                os.link(path, tmp)
            except OSError:
                shutil.copyfile(path, tmp)
            tmp.replace(dest)
        return dict(image='frames/'+dest.name, sha256=sha)

    def seal(self, name, records, labels):
        if Path(name).name != name or len(records) != len(labels):
            raise ValueError('Invalid shard name/label count')
        payload = dict(schema='photo-goal-city-shard/v1', run_id=self.run_id,
                       split=self.split, transitions=records)
        label_payload = dict(schema='photo-goal-city-labels/v1', labels=labels)
        label_path = self.root/'labels'/f'{name}.json'
        write(label_path, label_payload)
        payload['label_sidecar'] = dict(path='labels/'+label_path.name, sha256=digest(label_path))
        record_path = self.root/'records'/f'{name}.json'
        write(record_path, payload)
        return dict(path=str(record_path), sha256=digest(record_path),
                    transition_count=len(records), labels_sha256=digest(label_path))


def load_shard(path, require_train=True):
    path = Path(path).resolve()
    data, root = read(path), path.parent.parent
    if data.get('schema') != 'photo-goal-city-shard/v1':
        raise ValueError('Invalid shard schema')
    if require_train and data['split'] != 'train':
        raise ValueError('Development/sealed flights cannot train models')
    sidecar = contained(root, data['label_sidecar']['path'])
    if digest(sidecar) != data['label_sidecar']['sha256']:
        raise ValueError('Label sidecar hash mismatch')
    labels = read(sidecar)['labels']
    if len(labels) != len(data['transitions']):
        raise ValueError('Shard label count mismatch')
    for record in data['transitions']:
        for image in record['images']:
            resolved = contained(root, image['image'])
            if digest(resolved) != image['sha256']:
                raise ValueError('Canonical RGB hash mismatch')
    return data, labels, root


def rgb_tensor(root, image):
    with Image.open(contained(root, image['image'])) as value:
        pixels = np.asarray(value.convert('RGB')).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1)
