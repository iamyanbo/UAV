"""CURI local CUDA guard, loaded before every worker-launched Python process."""

import os


def _install_guard() -> None:
    if os.environ.get("CURI_GPU_MEMORY_GUARD") != "1":
        return
    # CPU/download stages deliberately hide CUDA. Do not initialize the WSL
    # driver merely to discover this: older CUDA runtimes can abort on that path.
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return
    try:
        requested = float(os.environ.get("CURI_MAX_VRAM_FRACTION", "0.60"))
    except ValueError:
        requested = 0.60
    ceiling = 0.80 if os.environ.get("CURI_APPROVED_CAMPAIGN") == "idea1-mission-world-model-2026-09-21" else 0.60
    fraction = max(0.01, min(requested, ceiling))
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    # PyTorch enforces this per process. CURI also keeps local numerical
    # concurrency at one by default, so normal experiments stay within the
    # same aggregate budget.
    torch.cuda.set_per_process_memory_fraction(fraction)


_install_guard()
