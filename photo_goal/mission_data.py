"""Content-addressed RGB, independent label sidecars and bounded feature cache."""
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
import io
import json
import queue
import threading
import numpy as np
import torch
from PIL import Image
from .common import digest, write, read, contained
from .mission_contracts import identity


def reward_identity(cfg):
    return identity(dict(potential=cfg.get('potential_normalization','initial_distance'),
        distance_scale=cfg.get('potential_distance_scale_m'),
        gamma_time_constant_s=cfg['gamma_time_constant_s'],
        terminal_rewards=cfg['terminal_rewards'],time_cost_per_mission=cfg['time_cost_per_mission'],
        reward_scale=cfg['reward_scale'],failure_remaining_time_charge=cfg.get('failure_remaining_time_charge',False)))


class FeatureStore(MutableMapping):
    def __init__(self, root, cache_size=32, restore=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.restore=restore
        self.pending={};self.lock=threading.RLock();self.error=None
        self.queue=queue.Queue(maxsize=64)
        self.writer=threading.Thread(target=self._write,daemon=True,name='derived-feature-writer')
        self.writer.start()

    def _write(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                key,value=item;path=self._path(key);pending=path.with_suffix('.pending')
                from .mission_space import reserve_write
                reserve_write(path,value.numel()*value.element_size()+65536)
                torch.save(value,pending);pending.replace(path)
            except BaseException as error:self.error=error
            finally:
                if item is not None:
                    with self.lock:self.pending.pop(item[0],None)
                self.queue.task_done()

    def flush(self):
        self.queue.join()
        if self.error:raise RuntimeError('Derived feature writer failed: '+str(self.error)) from self.error

    def close(self):
        self.queue.put(None);self.writer.join(timeout=30)
        if self.writer.is_alive():raise RuntimeError('Derived feature writer did not quiesce')
        if self.error:raise RuntimeError('Derived feature writer failed: '+str(self.error)) from self.error

    def _path(self, key):
        if not isinstance(key, int) or key < 0:
            raise ValueError('Invalid feature identity')
        return self.root/f'{key:09d}.pt'

    def __getitem__(self, key):
        if self.error:raise RuntimeError('Derived feature writer failed: '+str(self.error)) from self.error
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with self.lock:
            if key in self.pending:return self.pending[key]
        path=self._path(key)
        if not path.exists() and self.restore is not None:
            self[key]=self.restore(key)
            return self[key]
        value = torch.load(path, map_location='cpu', weights_only=True)
        self.cache[key] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def __setitem__(self, key, value):
        path = self._path(key)
        if self.error:raise RuntimeError('Derived feature writer failed: '+str(self.error)) from self.error
        if path.exists() or key in self.pending or key in self.cache:
            raise ValueError('Feature identity is immutable')
        value=value.detach().cpu().half().contiguous()
        self.cache[key]=value;self.cache.move_to_end(key)
        while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        with self.lock:
            self.pending[key]=value
            try:self.queue.put_nowait((key,value))
            except queue.Full:
                # This is a reconstructible cache. Canonical RGB remains in the
                # recorder; current history is retained in RAM, never block flight.
                self.pending.pop(key,None)

    def __delitem__(self, key):
        if key in self.pending:raise RuntimeError('Flush derived features before eviction')
        self._path(key).unlink(missing_ok=True)
        self.cache.pop(key, None)

    def __iter__(self):
        with self.lock:ids=set(self.pending)|set(self.cache)
        ids|={int(p.stem) for p in self.root.glob('*.pt')}
        return iter(sorted(ids))

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        return key in self.cache or key in self.pending or self._path(key).is_file()


class ShardWriter:
    def __init__(self, root, run_id, split='train', catalog=None, config=None):
        self.root, self.run_id, self.split = Path(root), run_id, split
        self.catalog=catalog
        self.config=config
        self.provider=None
        if catalog:
            from .mission_rgb_store import RGBProvider
            self.provider=RGBProvider(catalog)
        if split not in ('train', 'development', 'sealed'):
            raise ValueError('Invalid split')
        for name in ('frames', 'records', 'labels'):
            (self.root/name).mkdir(parents=True, exist_ok=True)

    def image(self, path):
        if self.provider:
            # Only current flight recordings are disposable, never task/survey/v1 files.
            removable=Path(path).resolve().is_relative_to(self.root.parent.resolve()) and any(part.startswith('worker-') for part in Path(path).parts)
            return self.provider.register(path,removable)
        with Image.open(path) as source:
            if source.size != (640, 480) or source.mode != 'RGB' or source.format != 'PNG':
                raise ValueError('Expected recorded full RGB frame')
        sha = digest(path)
        dest = self.root/'frames'/f'{sha}.png'
        if not dest.exists():
            import os
            import shutil
            from .mission_space import reserve_write
            reserve_write(dest,Path(path).stat().st_size+4096)
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
        payload = dict(schema='photo-goal-city-shard/v2' if self.provider else 'photo-goal-city-shard/v1', run_id=self.run_id,
                       split=self.split, transitions=records)
        label_payload = dict(schema='photo-goal-city-labels/v1', labels=labels)
        if self.config:
            payload.update(config_sha256=identity(self.config),reward_contract_sha256=reward_identity(self.config))
            label_payload['reward_contract_sha256']=payload['reward_contract_sha256']
        label_path = self.root/'labels'/f'{name}.json'
        write(label_path, label_payload)
        payload['label_sidecar'] = dict(path='labels/'+label_path.name, sha256=digest(label_path))
        record_path = self.root/'records'/f'{name}.json'
        write(record_path, payload)
        if self.provider:self.provider.flush()
        import os
        for path in (label_path,record_path):
            with path.open('rb') as stream:os.fsync(stream.fileno())
        return dict(path=str(record_path), sha256=digest(record_path),
                    transition_count=len(records), labels_sha256=digest(label_path))


def load_shard(path, require_train=True):
    path = Path(path).resolve()
    data, root = read(path), path.parent.parent
    if data.get('schema') not in ('photo-goal-city-shard/v1','photo-goal-city-shard/v2'):
        raise ValueError('Invalid shard schema')
    if require_train and data['split'] != 'train':
        raise ValueError('Development/sealed flights cannot train models')
    sidecar = contained(root, data['label_sidecar']['path'])
    if digest(sidecar) != data['label_sidecar']['sha256']:
        raise ValueError('Label sidecar hash mismatch')
    label_data=read(sidecar);labels = label_data['labels']
    if data.get('reward_contract_sha256')!=label_data.get('reward_contract_sha256'):
        raise ValueError('Reward contract differs between shard and labels')
    if len(labels) != len(data['transitions']):
        raise ValueError('Shard label count mismatch')
    providers={};verified=set()
    try:
        for record in data['transitions']:
            for image in record['images']:
                if image.get('schema')=='photo-goal-rgb-ref/v2':
                    from .mission_rgb_store import RGBProvider
                    project=Path(__import__('os').environ['UAV_PROJECT_ROOT']).resolve()
                    catalog=Path(image['catalog']).resolve()
                    if not catalog.is_relative_to(project):raise ValueError('RGB catalog escapes project')
                    key=(str(catalog),image['pixel_sha256'])
                    if key not in verified:
                        if str(catalog) not in providers:providers[str(catalog)]=RGBProvider(catalog)
                        providers[str(catalog)].resolve(image['pixel_sha256']);verified.add(key)
                    continue
                resolved = contained(root, image['image'])
                if digest(resolved) != image['sha256']:
                    raise ValueError('Canonical RGB hash mismatch')
    finally:
        for provider in providers.values():provider.close()
    return data, labels, root


def rgb_tensor(root, image, providers=None):
    if image.get('schema')=='photo-goal-rgb-ref/v2':
        from .mission_rgb_store import RGBProvider
        if providers is None:
            provider=RGBProvider(image['catalog'])
            try:pixels=provider.resolve(image['pixel_sha256'])
            finally:provider.close()
        else:
            key=image['catalog']
            if key not in providers:providers[key]=RGBProvider(key)
            pixels=providers[key].resolve(image['pixel_sha256'])
        return torch.from_numpy(pixels).permute(2,0,1)
    with Image.open(contained(root, image['image'])) as value:
        pixels = np.asarray(value.convert('RGB')).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1)
