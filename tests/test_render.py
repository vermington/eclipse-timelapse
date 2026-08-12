from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from eclipse_timelapse.config import AnalysisConfig, InputConfig, ProjectConfig, RenderConfig
from eclipse_timelapse.model import AnalysisReport, FrameAnalysis
from eclipse_timelapse.render import RenderError, _exclusive_output_lock, render_project


def test_output_lock_rejects_concurrent_renderer(tmp_path) -> None:
    output = tmp_path / "video.mp4"

    with (
        _exclusive_output_lock(output),
        pytest.raises(RenderError, match="already being rendered"),
        _exclusive_output_lock(output),
    ):
        pass


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
            interpolation="morph",
            preset="ultrafast",
        ),
    )
    report = AnalysisReport(
        schema_version=2,
        source_directory=".",
        pattern="frame-*.jpg",
        detection_threshold=20,
        blur_threshold=0.65,
        median_radius=60.0,
        frames=tuple(frames),
    )

    output = render_project(config, report)

    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        available, _image = capture.read()
        if not available:
            break
        decoded += 1
    capture.release()
    assert decoded == 4
