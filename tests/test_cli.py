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
            "--no-source-anchors",
            "--no-ingress-infill",
            "--infill-cutoff",
            "18.5",
            "--infill-interval",
            "0.6",
            "--infill-min-gap",
            "0.7",
            "--aspect-ratio",
            "4:5",
            "--final-hold",
            "2",
        ]
    )
    config = _with_overrides(ProjectConfig.load(Path("eclipse.toml")), options)

    assert config.analysis.blur_threshold == 0.8
    assert config.render.exclude_blurry is False
    assert config.render.source_anchors is False
    assert config.render.ingress_infill is False
    assert config.render.ingress_infill_cutoff_seconds == 18.5
    assert config.render.ingress_infill_interval_seconds == 0.6
    assert config.render.ingress_infill_minimum_gap_seconds == 0.7
    assert config.render.final_hold_seconds == 2.0
    assert output_dimensions(config.render) == (1080, 1350)


def test_annotation_cli_accepts_explicit_artifacts() -> None:
    options = build_parser().parse_args(
        [
            "annotate-centres",
            "--video",
            "output/source.mp4",
            "--report",
            "output/source.json",
            "--output",
            "output/centres.mp4",
        ]
    )

    assert options.video == Path("output/source.mp4")
    assert options.report == Path("output/source.json")
    assert options.output == Path("output/centres.mp4")
