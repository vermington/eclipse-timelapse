import json
from dataclasses import replace
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from eclipse_timelapse.config import AnalysisConfig, InputConfig, ProjectConfig, RenderConfig
from eclipse_timelapse.model import AnalysisReport, EclipseModel, FrameAnalysis
from eclipse_timelapse.render import (
    RenderError,
    _AlignedFrameCache,
    _DetailFeature,
    _DetailMotion,
    _exclusive_output_lock,
    _fit_detail_motion,
    _physical_frame,
    _schedule_source_anchors,
    _SourceAnchorInterpolator,
    align_frame,
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

    render = RenderConfig(resolution=64, aspect_ratio="1:1", crop_size=128)
    cache = _AlignedFrameCache(tmp_path, tuple(frames), render)
    atlas = cache.prepare_physical_atlas(solar_radius=30.0)
    solar_pixels = cache.solar_mask > 0.5

    assert np.array_equal(atlas[:, :, 0][solar_pixels], atlas[:, :, 1][solar_pixels])
    assert np.array_equal(atlas[:, :, 1][solar_pixels], atlas[:, :, 2][solar_pixels])
    assert np.ptp(atlas[:, :, 0][solar_pixels]) <= 1

    cache.atlas_detail = np.zeros((64, 64), dtype=np.float32)
    cache.atlas_detail[31:34, 19:22] = -24.0
    cache.detail_motion = _DetailMotion(
        reference_time=frames[0].captured_at,
        velocity_x_pixels_per_second=5.0,
        supporting_frames=2,
    )
    model = EclipseModel(
        reference_time=frames[0].captured_at,
        solar_radius=30.0,
        moon_radius=30.0,
        moon_x_intercept=1_000.0,
        moon_x_velocity=0.0,
        moon_y_intercept=0.0,
        moon_y_velocity=0.0,
        supporting_frames=2,
    )
    first = _physical_frame(frames[0], frames[1], 0.0, cache, model)
    second = _physical_frame(frames[0], frames[1], 1.0, cache, model)
    interior = cache.solar_mask > 0.99
    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    first_y, first_x = np.unravel_index(np.argmin(np.where(interior, first_gray, 255)), (64, 64))
    second_y, second_x = np.unravel_index(
        np.argmin(np.where(interior, second_gray, 255)),
        (64, 64),
    )
    assert second_x - first_x == 5
    assert second_y == first_y

    positions = np.asarray([0.0, 1.0])
    anchors = _schedule_source_anchors(
        positions,
        total_frames=6,
        frames_per_second=render.frames_per_second,
    )
    interpolator = _SourceAnchorInterpolator(cache, positions, anchors, total_frames=6)
    assert np.array_equal(interpolator.render(0), cache.get(0))
    assert np.array_equal(interpolator.render(5), cache.get(1))
    anchor_report = interpolator.anchor_report()
    assert anchor_report["pre_encode_pixel_identity"] is True
    assert anchor_report["encoded_pixel_identity"] is False
    assert anchor_report["anchor_count"] == 2


def test_source_anchor_schedule_is_unique_ordered_and_pins_endpoints() -> None:
    positions = np.asarray([0.0, 0.2, 0.2, 0.21, 1.0])

    anchors = _schedule_source_anchors(
        positions,
        total_frames=10,
        frames_per_second=60,
    )

    output_frames = [anchor.output_frame for anchor in anchors]
    assert output_frames[0] == 0
    assert output_frames[-1] == 9
    assert output_frames == sorted(set(output_frames))
    assert [anchor.source_index for anchor in anchors] == list(range(len(positions)))


def test_detail_motion_follows_longest_confident_track() -> None:
    reference_time = datetime(2026, 8, 12, 18, 0, 0)
    features = []
    for frame_index in range(12):
        elapsed = frame_index * 100.0
        features.extend(
            [
                _DetailFeature(
                    frame_index=frame_index,
                    elapsed_seconds=elapsed,
                    x=-170.0 + 0.004 * elapsed,
                    y=-80.0 + 0.006 * elapsed,
                    strength=500.0,
                ),
                _DetailFeature(
                    frame_index=frame_index,
                    elapsed_seconds=elapsed,
                    x=40.0,
                    y=60.0,
                    strength=20.0,
                ),
            ]
        )

    motion = _fit_detail_motion(
        features,
        reference_time=reference_time,
        position_tolerance=1.0,
    )

    assert motion.supporting_frames == 12
    assert motion.time_span_seconds == 1_100.0
    assert motion.velocity_x_pixels_per_second == pytest.approx(0.004)
    assert motion.velocity_y_pixels_per_second == pytest.approx(0.006)
    assert motion.median_residual < 1e-6


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
                blurry=index == 2,
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

    render_report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert render_report["solar_detail_motion"]["supporting_frames"] == 0
    assert render_report["source_anchors"]["anchor_count"] == 2
    assert render_report["source_anchors"]["all_source_frames_anchored"] is True
    assert render_report["excluded_blurry_frames"] == []
    assert render_report["excluded_blurry_from_reconstruction"] == ["frame-2.jpg"]

    capture = cv2.VideoCapture(str(output))
    decoded_frames = []
    while True:
        available, image = capture.read()
        if not available:
            break
        decoded_frames.append(image)
    capture.release()
    assert len(decoded_frames) == 4

    bright_frames = sum(
        np.any(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 30) for image in decoded_frames
    )
    assert bright_frames >= 2

    archive_config = replace(
        config,
        render=replace(config.render, output="output/test.mkv", codec="ffv1"),
    )
    archive_output = render_project(archive_config, report)
    archive_capture = cv2.VideoCapture(str(archive_output))
    archive_frames = []
    while True:
        available, image = archive_capture.read()
        if not available:
            break
        archive_frames.append(image)
    archive_capture.release()
    assert len(archive_frames) == 4
    for output_image, source_frame in zip(
        (archive_frames[0], archive_frames[-1]),
        (frames[0], frames[-1]),
        strict=True,
    ):
        source_image = cv2.imread(str(tmp_path / source_frame.filename), cv2.IMREAD_COLOR)
        expected_rgb = align_frame(
            source_image,
            source_frame,
            output_width=64,
            output_height=80,
            crop_size=256,
        )
        assert np.array_equal(output_image, cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR))
