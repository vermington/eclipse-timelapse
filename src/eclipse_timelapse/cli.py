"""Command-line interface for the eclipse timelapse pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from eclipse_timelapse import __version__
from eclipse_timelapse.analysis import AnalysisError, analyse_project
from eclipse_timelapse.config import ConfigurationError, ProjectConfig
from eclipse_timelapse.model import AnalysisReport
from eclipse_timelapse.render import RenderError, render_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eclipse-timelapse",
        description="Analyse, align, and render a timestamp-aware eclipse timelapse.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("eclipse.toml"),
        help="project configuration file (default: eclipse.toml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="detect, align, and score all source images"
    )
    _add_analysis_overrides(analyze_parser)

    render_parser = subparsers.add_parser("render", help="render from an existing analysis")
    _add_render_overrides(render_parser)

    run_parser = subparsers.add_parser("run", help="analyse and render the complete project")
    _add_analysis_overrides(run_parser)
    _add_render_overrides(run_parser)
    return parser


def _add_analysis_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--blur-threshold",
        type=float,
        help="flag frames with a sharpness score below this value",
    )


def _add_render_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution", type=int, help="output width in pixels")
    parser.add_argument(
        "--aspect-ratio",
        help="output width:height ratio, for example 1:1, 4:5, or 9:16",
    )
    parser.add_argument("--crop-size", type=int, help="source crop width in pixels")
    parser.add_argument("--duration", type=float, help="video duration in seconds")
    parser.add_argument(
        "--final-hold",
        type=float,
        help="additional seconds to hold the final rendered frame",
    )
    parser.add_argument("--fps", type=int, help="frames per second")
    parser.add_argument(
        "--timeline",
        choices=("uniform", "linear", "logarithmic", "capped"),
        help="capture-gap mapping strategy",
    )
    parser.add_argument(
        "--interpolation",
        choices=("physical", "morph", "geometry", "crossfade"),
        help="transition strategy; physical preserves centring and clean edges",
    )
    anchor_group = parser.add_mutually_exclusive_group()
    anchor_group.add_argument(
        "--source-anchors",
        dest="source_anchors",
        action="store_true",
        default=None,
        help=(
            "render only complete aligned source photographs, with every included "
            "source on its own scheduled frame"
        ),
    )
    anchor_group.add_argument(
        "--no-source-anchors",
        dest="source_anchors",
        action="store_false",
        help="render every frame from the selected interpolation model",
    )
    infill_group = parser.add_mutually_exclusive_group()
    infill_group.add_argument(
        "--ingress-infill",
        dest="ingress_infill",
        action="store_true",
        default=None,
        help="sparsely advance the lunar boundary using only each gap's starting photo",
    )
    infill_group.add_argument(
        "--no-ingress-infill",
        dest="ingress_infill",
        action="store_false",
        help="hold only complete source photographs between anchors",
    )
    parser.add_argument(
        "--infill-cutoff",
        dest="ingress_infill_cutoff_seconds",
        type=float,
        help="stop generating subtractive ingress frames at this output time",
    )
    parser.add_argument(
        "--infill-min-gap",
        dest="ingress_infill_minimum_gap_seconds",
        type=float,
        help="minimum output gap eligible for subtractive infill",
    )
    parser.add_argument(
        "--infill-interval",
        dest="ingress_infill_interval_seconds",
        type=float,
        help="time between distinct subtractive boundary states",
    )
    parser.add_argument(
        "--codec",
        choices=("h264", "ffv1"),
        help="H.264 for delivery MP4 or lossless FFV1 for an archival MKV",
    )
    parser.add_argument("--output", type=str, help="output video path, relative to the project")
    blur_group = parser.add_mutually_exclusive_group()
    blur_group.add_argument(
        "--include-blurry",
        dest="exclude_blurry",
        action="store_false",
        default=None,
        help="include frames flagged as blurry",
    )
    blur_group.add_argument(
        "--exclude-blurry",
        dest="exclude_blurry",
        action="store_true",
        help="exclude frames flagged as blurry",
    )


def _with_overrides(config: ProjectConfig, arguments: argparse.Namespace) -> ProjectConfig:
    analysis = config.analysis
    render = config.render
    blur_threshold = getattr(arguments, "blur_threshold", None)
    if blur_threshold is not None:
        analysis = replace(analysis, blur_threshold=blur_threshold)
    replacements = {}
    for argument, field in (
        ("resolution", "resolution"),
        ("aspect_ratio", "aspect_ratio"),
        ("crop_size", "crop_size"),
        ("duration", "duration_seconds"),
        ("final_hold", "final_hold_seconds"),
        ("fps", "frames_per_second"),
        ("timeline", "timeline"),
        ("interpolation", "interpolation"),
        ("source_anchors", "source_anchors"),
        ("ingress_infill", "ingress_infill"),
        ("ingress_infill_cutoff_seconds", "ingress_infill_cutoff_seconds"),
        ("ingress_infill_minimum_gap_seconds", "ingress_infill_minimum_gap_seconds"),
        ("ingress_infill_interval_seconds", "ingress_infill_interval_seconds"),
        ("codec", "codec"),
        ("output", "output"),
    ):
        value = getattr(arguments, argument, None)
        if value is not None:
            replacements[field] = value
    exclude_blurry = getattr(arguments, "exclude_blurry", None)
    if exclude_blurry is not None:
        replacements["exclude_blurry"] = exclude_blurry
    if replacements:
        render = replace(render, **replacements)
    updated = replace(config, analysis=analysis, render=render)
    updated.validate()
    return updated


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    options = parser.parse_args(arguments)
    try:
        config = ProjectConfig.load(options.config)
        config = _with_overrides(config, options)
        if options.command in {"analyze", "run"}:
            report = analyse_project(config, progress=print)
            blurry = sum(frame.blurry for frame in report.frames)
            print(
                f"Analysis complete: {len(report.frames)} frames, {blurry} flagged blurry, "
                f"median radius {report.median_radius:.1f}px"
            )
        else:
            report = AnalysisReport.read(config.work_directory / "analysis.json")
        if options.command in {"render", "run"}:
            output = render_project(config, report, progress=print)
            print(f"Rendered {output}")
        return 0
    except (AnalysisError, ConfigurationError, RenderError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
