from pathlib import Path

from eclipse_timelapse.cli import _with_overrides, build_parser
from eclipse_timelapse.config import ProjectConfig, output_dimensions


def test_run_overrides_blur_policy_threshold_and_aspect_ratio() -> None:
    options = build_parser().parse_args(
        [
            "run",
            "--blur-threshold",
            "0.8",
            "--include-blurry",
            "--aspect-ratio",
            "4:5",
        ]
    )
    config = _with_overrides(ProjectConfig.load(Path("eclipse.toml")), options)

    assert config.analysis.blur_threshold == 0.8
    assert config.render.exclude_blurry is False
    assert output_dimensions(config.render) == (1080, 1350)
