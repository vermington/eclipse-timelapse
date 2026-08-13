from pathlib import Path

import pytest

from eclipse_timelapse.config import (
    ConfigurationError,
    ProjectConfig,
    RenderConfig,
    output_dimensions,
)


def test_repository_configuration_is_valid() -> None:
    config = ProjectConfig.load(Path("eclipse.toml"))

    assert config.render.resolution == 1080
    assert output_dimensions(config.render) == (1080, 1350)
    assert config.render.duration_seconds == 26.25
    assert config.render.frames_per_second == 60
    assert config.render.timeline == "linear"
    assert config.render.interpolation == "physical"
    assert config.render.source_anchors is True
    assert config.render.ingress_infill is True
    assert config.render.ingress_infill_cutoff_seconds == 19.0
    assert config.render.ingress_infill_minimum_gap_seconds == 0.75
    assert config.render.ingress_infill_interval_seconds == 0.75
    assert config.render.exclude_blurry is False


def test_odd_video_resolution_is_rejected(tmp_path) -> None:
    filename = tmp_path / "invalid.toml"
    filename.write_text("[render]\nresolution = 999\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="even integer"):
        ProjectConfig.load(filename)


def test_instagram_portrait_dimensions() -> None:
    render = RenderConfig(resolution=1080, aspect_ratio="4:5")

    assert output_dimensions(render) == (1080, 1350)


def test_invalid_aspect_ratio_is_rejected(tmp_path) -> None:
    filename = tmp_path / "invalid.toml"
    filename.write_text('[render]\naspect_ratio = "portrait"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="positive integers"):
        ProjectConfig.load(filename)


def test_ffv1_requires_an_archival_container(tmp_path) -> None:
    filename = tmp_path / "invalid.toml"
    filename.write_text(
        '[render]\ncodec = "ffv1"\noutput = "output/master.mp4"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires an .mkv or .avi"):
        ProjectConfig.load(filename)


def test_source_anchors_require_physical_interpolation(tmp_path) -> None:
    filename = tmp_path / "invalid.toml"
    filename.write_text(
        '[render]\nsource_anchors = true\ninterpolation = "crossfade"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires"):
        ProjectConfig.load(filename)


@pytest.mark.parametrize(
    "setting",
    (
        "ingress_infill_cutoff_seconds",
        "ingress_infill_minimum_gap_seconds",
        "ingress_infill_interval_seconds",
    ),
)
def test_ingress_infill_timing_settings_must_be_positive(tmp_path, setting) -> None:
    filename = tmp_path / "invalid.toml"
    filename.write_text(f"[render]\n{setting} = 0\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=setting):
        ProjectConfig.load(filename)
