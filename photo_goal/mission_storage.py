"""Local block-device inspection and HDD-only run/cache placement; no SSH."""
import json
import os
from pathlib import Path
import subprocess


def inspect_hdd(root):
    root = Path(root).resolve(strict=True)
    if os.name == 'nt':
        script = """
$uavVolume = Get-Volume -FilePath $env:UAV_STORAGE_ROOT -ErrorAction Stop
$uavPartition = Get-Partition -DriveLetter $uavVolume.DriveLetter -ErrorAction Stop
$uavDisk = Get-Disk -Number $uavPartition.DiskNumber -ErrorAction Stop
$uavPhysical = @(Get-PhysicalDisk | Where-Object { $_.DeviceId -eq [string]$uavDisk.Number })
@{disk_number=$uavDisk.Number; bus_type=[string]$uavDisk.BusType; media_types=@($uavPhysical | ForEach-Object { [string]$_.MediaType })} | ConvertTo-Json -Compress
"""
        env = dict(os.environ, UAV_STORAGE_ROOT=str(root))
        result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                                env=env, capture_output=True, text=True, check=True, timeout=30)
        measured = json.loads(result.stdout)
        valid = bool(measured['media_types']) and all(x == 'HDD' for x in measured['media_types'])
    else:
        mount = json.loads(subprocess.run(['findmnt', '--json', '--target', str(root), '--output', 'SOURCE,TARGET,FSTYPE'],
                           capture_output=True, text=True, check=True, timeout=30).stdout)['filesystems'][0]
        source = mount['source'].split('[')[0]
        devices = json.loads(subprocess.run(['lsblk', '--json', '--inverse', '--output', 'NAME,PATH,TYPE,ROTA', source],
                             capture_output=True, text=True, check=True, timeout=30).stdout)['blockdevices']
        def leaves(rows):
            result = []
            for row in rows:
                result.extend(leaves(row.get('children', [])))
                if row['type'] == 'disk':
                    result.append(row)
            return result
        disks = leaves(devices)
        measured = dict(mount=mount, physical_devices=disks)
        valid = bool(disks) and all(row['rota'] in (True, 1, '1') for row in disks)
    if not valid:
        raise RuntimeError('Data path is not verified as a physical hard drive: '+json.dumps(measured))
    return dict(root=str(root), verified_hdd=True, measurement=measured)


def configure(root, outputs=()):
    receipt = inspect_hdd(root)
    root = Path(root).resolve()
    if any(not Path(path).resolve().is_relative_to(root) for path in outputs if path):
        raise ValueError('Writable training/teacher outputs must reside on the verified HDD root')
    cache = {'TMPDIR':'tmp', 'TMP':'tmp', 'TEMP':'tmp', 'XDG_CACHE_HOME':'cache',
             'HF_HOME':'cache/huggingface', 'HF_HUB_CACHE':'cache/huggingface/hub',
             'HF_DATASETS_CACHE':'cache/huggingface/datasets', 'TORCH_HOME':'cache/torch',
             'PIP_CACHE_DIR':'cache/pip', 'CUDA_CACHE_PATH':'cache/cuda',
             'TRITON_CACHE_DIR':'cache/triton', 'WANDB_DIR':'cache/wandb'}
    for variable, relative in cache.items():
        directory = root/relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    import sys
    sys.pycache_prefix = str(root/'cache'/'pycache')
    os.environ['PYTHONPYCACHEPREFIX'] = sys.pycache_prefix
    receipt['cache_environment'] = {variable: os.environ[variable] for variable in cache}
    return receipt
