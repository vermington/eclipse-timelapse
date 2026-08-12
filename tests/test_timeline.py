from datetime import datetime, timedelta

import numpy as np

from eclipse_timelapse.timeline import frame_blend, timeline_positions


def test_logarithmic_timeline_compresses_long_gaps() -> None:
    start = datetime(2026, 8, 12, 18, 0, 0)
    timestamps = [start, start + timedelta(seconds=1), start + timedelta(seconds=101)]
    positions = timeline_positions(
        timestamps,
        mode="logarithmic",
        minimum_gap_seconds=1,
        maximum_gap_seconds=30,
    )

    assert positions[0] == 0
    assert positions[-1] == 1
    assert positions[1] > 0.1


def test_frame_blend_is_smooth_and_bounded() -> None:
    positions = np.asarray([0.0, 0.5, 1.0])

    assert frame_blend(positions, -1) == (0, 1, 0.0)
    assert frame_blend(positions, 0.25) == (0, 1, 0.5)
    assert frame_blend(positions, 1) == (2, 2, 0.0)


def test_linear_timeline_uses_exact_clock_time() -> None:
    start = datetime(2026, 8, 12, 18, 0, 0)
    timestamps = [start, start + timedelta(seconds=10), start + timedelta(seconds=30)]

    positions = timeline_positions(
        timestamps,
        mode="linear",
        minimum_gap_seconds=1,
        maximum_gap_seconds=30,
    )

    np.testing.assert_allclose(positions, [0.0, 1.0 / 3.0, 1.0])
    assert frame_blend(positions, 5.0 / 30.0) == (0, 1, 0.5)


def test_linear_timeline_does_not_clamp_short_capture_gaps() -> None:
    start = datetime(2026, 8, 12, 18, 0, 0)
    timestamps = [start, start + timedelta(milliseconds=250), start + timedelta(seconds=10)]

    positions = timeline_positions(
        timestamps,
        mode="linear",
        minimum_gap_seconds=1,
        maximum_gap_seconds=30,
    )

    np.testing.assert_allclose(positions, [0.0, 0.025, 1.0])


def test_frame_blend_does_not_ease_between_photographs() -> None:
    positions = np.asarray([0.0, 1.0])

    assert frame_blend(positions, 0.25) == (0, 1, 0.25)
    assert frame_blend(positions, 0.75) == (0, 1, 0.75)
