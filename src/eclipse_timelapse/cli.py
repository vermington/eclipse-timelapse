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
    subparsers.add_parser("analyze", help="detect, align, and score all source images")

    render_parser = subparsers.add_parser("render", help="render from an existing analysis")
    _add_render_overrides(render_parser)

    run_parser = subparsers.add_parser("run", help="analyse and render the complete project")
    _add_render_overrides(run_parser)
    return parser


def _add_render_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution", type=int, help="square output size in pixels")
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
        choices=("morph", "geometry", "crossfade"),
        help="transition strategy; morph avoids doubled lunar edges",
    )
    parser.add_argument("--output", type=str, help="output MP4 path, relative to the project")
    parser.add_argument(
        "--include-blurry",
        action="store_true",
        help="include frames flagged as blurry",
    )


def _with_overrides(config: ProjectConfig, arguments: argparse.Namespace) -> ProjectConfig:
    render = config.render
    replacements = {}
    for argument, field in (
        ("resolution", "resolution"),
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
    if getattr(arguments, "include_blurry", False):
        replacements["exclude_blurry"] = False
    if replacements:
        render = replace(render, **replacements)
    updated = replace(config, render=render)
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
