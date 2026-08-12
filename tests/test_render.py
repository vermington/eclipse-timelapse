from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from eclipse_timelapse.config import AnalysisConfig, InputConfig, ProjectConfig, RenderConfig
from eclipse_timelapse.model import AnalysisReport, EclipseModel, FrameAnalysis
from eclipse_timelapse.render import (
    RenderError,
    _AlignedFrameCache,
    _exclusive_output_lock,
    render_project,
)


def test_output_lock_rejects_concurrent_renderer(tmp_path) -> None:
    output = tmp_path / "video.mp4"

    with (
        _exclusive_output_lock(output),
        pytest.raises(RenderError, match="already being rendered"),
        _exclusive_output_lock(output),
    ):
        pass


def test_physical_atlas_normalizes_source_colour_seams(tmp_path) -> None:
    frames = []
    colours = ((80, 100, 160), (160, 100, 80))
    for index, colour in enumerate(colours):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(image, (64, 64), 30, colour, thickness=-1)
        if index == 0:
            image[:, 64:] = 0
        else:
            image[:, :64] = 0
        filename = f"colour-{index}.png"
        cv2.imwrite(str(tmp_path / filename), image)
        frames.append(
            FrameAnalysis(
                sequence=index + 1,
                filename=filename,
                captured_at=datetime(2026, 8, 12) + timedelta(seconds=index),
                width=128,
                height=128,
                center_x=64.0,
                center_y=64.0,
                radius=30.0,
                moon_center_x=64.0,
                moon_center_y=64.0,
                moon_radius=30.0,
                moon_fit_error=0.1,
                bright_pixels=1_000,
                brightness=100.0,
                sharpness=2.0,
                blurry=False,
            )
        )

    render = RenderConfig(resolution=64, crop_size=128)
    cache = _AlignedFrameCache(tmp_path, tuple(frames), render)
    atlas = cache.prepare_physical_atlas(solar_radius=30.0)
    solar_pixels = cache.solar_mask > 0.5

    assert np.array_equal(atlas[:, :, 0][solar_pixels], atlas[:, :, 1][solar_pixels])
    assert np.array_equal(atlas[:, :, 1][solar_pixels], atlas[:, :, 2][solar_pixels])
    assert np.ptp(atlas[:, :, 0][solar_pixels]) <= 1


def test_small_video_render_is_decodable(tmp_path) -> None:
    source_size = 256
    center = 128.0
    frames = []
    for index, moon_x in enumerate((155, 125), start=1):
        image = np.zeros((source_size, source_size, 3), dtype=np.uint8)
        cv2.circle(image, (128, 128), 60, (140, 140, 140), thickness=-1)
        cv2.circle(image, (moon_x, 128), 62, (0, 0, 0), thickness=-1)
        filename = f"frame-{index}.jpg"
        cv2.imwrite(str(tmp_path / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        frames.append(
            FrameAnalysis(
                sequence=index,
                filename=filename,
                captured_at=datetime(2026, 8, 12) + timedelta(seconds=10 * index),
                width=source_size,
                height=source_size,
                center_x=center,
                center_y=center,
                radius=60.0,
                moon_center_x=float(moon_x),
                moon_center_y=center,
                moon_radius=62.0,
                moon_fit_error=0.1,
                bright_pixels=5_000,
                brightness=140.0,
                sharpness=2.0,
                blurry=False,
            )
        )

    config = ProjectConfig(
        root=tmp_path,
        input=InputConfig(directory=".", pattern="frame-*.jpg"),
        analysis=AnalysisConfig(work_directory="work"),
        render=RenderConfig(
            output="output/test.mp4",
            resolution=64,
            crop_size=256,
            duration_seconds=0.5,
            frames_per_second=8,
            interpolation="physical",
            preset="ultrafast",
        ),
    )
    report = AnalysisReport(
        schema_version=3,
        source_directory=".",
        pattern="frame-*.jpg",
        detection_threshold=20,
        blur_threshold=0.65,
        median_radius=60.0,
        eclipse_model=EclipseModel(
            reference_time=frames[0].captured_at,
            solar_radius=60.0,
            moon_radius=62.0,
            moon_x_intercept=27.0,
            moon_x_velocity=-3.0,
            moon_y_intercept=0.0,
            moon_y_velocity=0.0,
            supporting_frames=2,
        ),
        frames=tuple(frames),
    )

    output = render_project(config, report)

    capture = cv2.VideoCapture(str(output))
    decoded_frames = []
    while True:
        available, image = capture.read()
        if not available:
            break
        decoded_frames.append(image)
    capture.release()
    assert len(decoded_frames) == 4

    grid_y, grid_x = np.mgrid[:64, :64]
    solar_interior = np.hypot(grid_x - 32.0, grid_y - 32.0) <= 13.0
    checked_frames = 0
    for index, image in enumerate(decoded_frames):
        progress = index / (len(decoded_frames) - 1)
        moon_x = 32.0 + (27.0 - 30.0 * progress) / 4.0
        moon_interior = np.hypot(grid_x - moon_x, grid_y - 32.0) <= 17.0
        visible_interior = solar_interior & ~moon_interior
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.any(visible_interior):
            checked_frames += 1
            assert np.all(gray[visible_interior] > 30)
    assert checked_frames >= 2
