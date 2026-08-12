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
    assert config.render.timeline == "linear"
    assert config.render.interpolation == "physical"
    assert config.render.exclude_blurry is True


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
