"""Measured local resource admission; no discovery or connections to a server."""
import os
from pathlib import Path
import shutil
import time
import psutil
import torch


class Resources:
    def __init__(self, root, cfg, device='cuda'):
        self.root, self.cfg, self.device = Path(root), cfg['resources'], device

    def check(self, growth_bytes=0, checkpoint_bytes=0, disk=True):
        available = psutil.virtual_memory().available
        if available < self.cfg['host_available_reserve_gib']*2**30+growth_bytes:
            raise RuntimeError('12 GiB available-host-memory reserve would be breached')
        if disk:
            free = shutil.disk_usage(self.root).free
            if free < self.cfg['disk_reserve_gib']*2**30+growth_bytes+2*checkpoint_bytes:
                raise RuntimeError('100 GiB data-drive reserve/checkpoint headroom would be breached')
        result = dict(host_available_bytes=available, root=str(self.root.resolve()))
        if self.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA device required for admitted native flight workload')
            free, total = torch.cuda.mem_get_info(self.device)
            used = total-free
            if used+growth_bytes > total*self.cfg['gpu_fraction_ceiling']:
                raise RuntimeError('60% total-device VRAM ceiling would be breached')
            result.update(device_total_bytes=total, device_used_bytes=used)
        return result


class RunWindow:
    def __init__(self, hours, reserve_s=300):
        if not 0 < hours <= 8:
            raise ValueError('Resumable windows must be in (0,8] hours')
        self.deadline = time.monotonic()+hours*3600
        self.reserve_s = reserve_s
        self.stop_file=Path(os.environ['UAV_WINDOW_STOP_FILE']) if os.environ.get('UAV_WINDOW_STOP_FILE') else None

    def admits(self, estimated_s=0):
        return (not self.stop_file or not self.stop_file.exists()) and time.monotonic()+self.reserve_s+estimated_s < self.deadline


def hdd_cache_environment(root):
    """Operator calls this on an already verified HDD, before model imports/downloads."""
    root = Path(root).resolve()
    names = {'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp', 'XDG_CACHE_HOME': 'cache',
             'HF_HOME': 'cache/huggingface', 'HF_HUB_CACHE': 'cache/huggingface/hub',
             'HF_DATASETS_CACHE': 'cache/huggingface/datasets', 'TORCH_HOME': 'cache/torch',
             'PIP_CACHE_DIR': 'cache/pip', 'CUDA_CACHE_PATH': 'cache/cuda',
             'TRITON_CACHE_DIR': 'cache/triton', 'WANDB_DIR': 'cache/wandb',
             'PYTHONPYCACHEPREFIX': 'cache/pycache'}
    for key, relative in names.items():
        path = root/relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    return {key: os.environ[key] for key in names}
