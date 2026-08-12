"""Timestamp-aware, gap-compressed video timelines."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np


def timeline_positions(
    timestamps: Sequence[datetime],
    *,
    mode: str,
    minimum_gap_seconds: float,
    maximum_gap_seconds: float,
) -> np.ndarray:
    """Map capture times onto a monotonic visual timeline.

    Logarithmic mode preserves the ordering and relative sense of elapsed time
    while preventing a long photographic gap from dominating the finished film.
    """
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required")
    raw_gaps = [
        max((later - earlier).total_seconds(), minimum_gap_seconds)
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    ]
    if mode == "uniform":
        weights = np.ones(len(raw_gaps), dtype=np.float64)
    elif mode == "linear":
        weights = np.asarray(raw_gaps, dtype=np.float64)
    elif mode == "capped":
        weights = np.minimum(raw_gaps, maximum_gap_seconds).astype(np.float64)
    elif mode == "logarithmic":
        weights = np.log1p(np.asarray(raw_gaps, dtype=np.float64))
    else:
        raise ValueError(f"Unknown timeline mode: {mode}")
    positions = np.concatenate(([0.0], np.cumsum(weights)))
    return positions / positions[-1]


def frame_blend(
    positions: np.ndarray,
    progress: float,
) -> tuple[int, int, float]:
    """Return neighbouring source indices and a smooth interpolation fraction."""
    progress = min(max(float(progress), 0.0), 1.0)
    if progress >= 1.0:
        final = len(positions) - 1
        return final, final, 0.0
    right = int(np.searchsorted(positions, progress, side="right"))
    left = max(0, right - 1)
    right = min(right, len(positions) - 1)
    width = positions[right] - positions[left]
    linear = 0.0 if width <= 0 else (progress - positions[left]) / width
    smooth = linear * linear * (3.0 - 2.0 * linear)
    return left, right, float(smooth)
