from pathlib import Path

import pytest

from eclipse_timelapse.config import ConfigurationError, ProjectConfig


def test_repository_configuration_is_valid() -> None:
    config = ProjectConfig.load(Path("eclipse.toml"))

    assert config.render.resolution == 1080
    assert config.render.interpolation == "morph"
    assert config.render.exclude_blurry is True


def test_odd_video_resolution_is_rejected(tmp_path) -> None:
    filename = tmp_path / "invalid.toml"
    filename.write_text("[render]\nresolution = 999\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="even integer"):
        ProjectConfig.load(filename)
