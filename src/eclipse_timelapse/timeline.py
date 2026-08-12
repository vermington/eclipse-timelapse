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

    Linear mode preserves exact elapsed time. Logarithmic and capped modes
    preserve ordering while preventing a long gap from dominating the film.
    """
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required")
    raw_gaps = np.asarray(
        [
            (later - earlier).total_seconds()
            for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        ],
        dtype=np.float64,
    )
    if np.any(raw_gaps < 0):
        raise ValueError("Timestamps must be in chronological order")
    if mode == "uniform":
        weights = np.ones(len(raw_gaps), dtype=np.float64)
    elif mode == "linear":
        # Do not clamp short intervals: linear mode represents clock time exactly.
        weights = raw_gaps
    elif mode == "capped":
        weights = np.minimum(np.maximum(raw_gaps, minimum_gap_seconds), maximum_gap_seconds)
    elif mode == "logarithmic":
        weights = np.log1p(np.maximum(raw_gaps, minimum_gap_seconds))
    else:
        raise ValueError(f"Unknown timeline mode: {mode}")
    positions = np.concatenate(([0.0], np.cumsum(weights)))
    if mode == "linear" and positions[-1] <= 0:
        raise ValueError("Linear timeline requires at least two distinct timestamps")
    return positions / positions[-1]


def frame_blend(
    positions: np.ndarray,
    progress: float,
) -> tuple[int, int, float]:
    """Return neighbouring source indices and a linear interpolation fraction."""
    progress = min(max(float(progress), 0.0), 1.0)
    if progress >= 1.0:
        final = len(positions) - 1
        return final, final, 0.0
    right = int(np.searchsorted(positions, progress, side="right"))
    left = max(0, right - 1)
    right = min(right, len(positions) - 1)
    width = positions[right] - positions[left]
    linear = 0.0 if width <= 0 else (progress - positions[left]) / width
    return left, right, float(linear)
