"""Transparent seed interface for real-data artifact programs.

This baseline is intentionally ordinary. Research tasks should replace it only
when a named mechanism, falsifier, and independent validation justify doing so.
"""
from __future__ import annotations

import numpy as np


def signal(close: np.ndarray, horizons: tuple[int, ...] = (20, 60, 120)) -> np.ndarray:
    """Return causal equal-vote momentum targets for a [time, asset] close matrix."""
    prices = np.asarray(close, dtype=float)
    if prices.ndim != 2:
        raise ValueError("close must be [time, asset]")
    targets = np.zeros_like(prices)
    for t in range(1, len(prices)):
        votes = []
        for horizon in horizons:
            if t < horizon:
                continue
            momentum = prices[t] / prices[t - horizon] - 1.0
            scale = np.nanstd(momentum)
            votes.append(np.clip((momentum - np.nanmean(momentum)) / (scale + 1e-12), -1.0, 1.0))
        if votes:
            targets[t] = np.clip(np.nanmean(votes, axis=0), -1.0, 1.0)
    return targets
