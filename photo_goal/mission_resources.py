"""Measured local resource admission; no discovery or connections to a server."""
import os
import json
import math
from pathlib import Path
import shutil
import time
import psutil
import torch


class Resources:
    def __init__(self, root, cfg, device='cuda'):
        self.root, self.cfg, self.device = Path(root), cfg['resources'], device
        # Operator allowance is independent of the learning/checkpoint contract.
        self.gpu_fraction = float(os.environ.get('UAV_GPU_FRACTION_CEILING',
                                                self.cfg['gpu_fraction_ceiling']))
        if not math.isfinite(self.gpu_fraction) or not 0 < self.gpu_fraction <= 1:
            raise ValueError('GPU memory allowance must be in (0, 1]')
        self.log_memory = os.environ.get('UAV_RESOURCE_LOG') == '1'
        self.last_memory_log = 0.
        self.last_logged_used = 0

    def check(self, growth_bytes=0, checkpoint_bytes=0, disk=True):
        available = psutil.virtual_memory().available
        if available < self.cfg['host_available_reserve_gib']*2**30+growth_bytes:
            raise RuntimeError('12 GiB available-host-memory reserve would be breached')
        if disk:
            from .mission_space import guard
            space=guard()
            if space: space.check(growth_bytes+2*checkpoint_bytes)
            free = shutil.disk_usage(self.root).free
            if free < self.cfg['disk_reserve_gib']*2**30+growth_bytes+2*checkpoint_bytes:
                raise RuntimeError('100 GiB data-drive reserve/checkpoint headroom would be breached')
        result = dict(host_available_bytes=available, root=str(self.root.resolve()))
        if self.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA device required for admitted native flight workload')
            free, total = torch.cuda.mem_get_info(self.device)
            used = total-free
            exceeded = used+growth_bytes > total*self.gpu_fraction
            result.update(device_total_bytes=total, device_used_bytes=used,
                          device_free_bytes=free, gpu_fraction_ceiling=self.gpu_fraction)
            now = time.monotonic()
            if exceeded or self.log_memory and (now-self.last_memory_log >= 30 or
                                               used-self.last_logged_used >= 256*2**20):
                print(json.dumps(dict(event='gpu_memory', pid=os.getpid(), device=self.device,
                      **result, requested_growth_bytes=growth_bytes,
                      process_allocated_bytes=torch.cuda.memory_allocated(self.device),
                      process_reserved_bytes=torch.cuda.memory_reserved(self.device),
                      process_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                      allowance_exceeded=exceeded)), flush=True)
                self.last_memory_log, self.last_logged_used = now, used
            if exceeded:
                raise RuntimeError(f'{self.gpu_fraction:.0%} total-device VRAM allowance would be breached '
                                   f'(used={used}, requested_growth={growth_bytes}, total={total} bytes)')
        return result


class RunWindow:
    def __init__(self, hours, reserve_s=300):
        if not 0 < hours <= 8:
            raise ValueError('Resumable windows must be in (0,8] hours')
        self.deadline = time.monotonic()+hours*3600
        if os.environ.get('UAV_WINDOW_DEADLINE_UNIX'):
            self.deadline=min(self.deadline,time.monotonic()+float(os.environ['UAV_WINDOW_DEADLINE_UNIX'])-time.time())
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
