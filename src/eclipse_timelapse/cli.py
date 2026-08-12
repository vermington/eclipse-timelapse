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
    parser.add_argument("--output", type=str, help="output MP4 path, relative to the project")
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
        ("fps", "frames_per_second"),
        ("timeline", "timeline"),
        ("interpolation", "interpolation"),
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
