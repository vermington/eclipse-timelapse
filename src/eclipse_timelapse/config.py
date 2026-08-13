"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when project configuration is invalid."""


@dataclass(frozen=True, slots=True)
class InputConfig:
    directory: str = "."
    pattern: str = "Eclipse_*.JPG"


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    work_directory: str = "work"
    detection_threshold: int = 20
    minimum_component_pixels: int = 2_000
    blur_threshold: float = 0.65


@dataclass(frozen=True, slots=True)
class RenderConfig:
    output: str = "output/eclipse_timelapse_source_anchored_instagram_4x5.mp4"
    resolution: int = 1080
    aspect_ratio: str = "4:5"
    crop_size: int = 2000
    duration_seconds: float = 26.25
    frames_per_second: int = 60
    timeline: str = "linear"
    interpolation: str = "physical"
    source_anchors: bool = True
    ingress_infill: bool = True
    ingress_infill_cutoff_seconds: float = 19.0
    ingress_infill_minimum_gap_seconds: float = 0.75
    ingress_infill_interval_seconds: float = 0.75
    maximum_gap_seconds: float = 30.0
    minimum_gap_seconds: float = 1.0
    exclude_blurry: bool = False
    codec: str = "h264"
    crf: int = 16
    preset: str = "slow"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    root: Path
    input: InputConfig
    analysis: AnalysisConfig
    render: RenderConfig

    @classmethod
    def load(cls, filename: Path) -> ProjectConfig:
        filename = filename.resolve()
        try:
            with filename.open("rb") as config_file:
                raw = tomllib.load(config_file)
        except FileNotFoundError as error:
            raise ConfigurationError(f"Configuration file not found: {filename}") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigurationError(f"Invalid TOML in {filename}: {error}") from error

        try:
            project = cls(
                root=filename.parent,
                input=InputConfig(**_section(raw, "input")),
                analysis=AnalysisConfig(**_section(raw, "analysis")),
                render=RenderConfig(**_section(raw, "render")),
            )
        except TypeError as error:
            raise ConfigurationError(
                f"Unsupported or missing configuration value: {error}"
            ) from error
        project.validate()
        return project

    @property
    def input_directory(self) -> Path:
        return (self.root / self.input.directory).resolve()

    @property
    def work_directory(self) -> Path:
        return (self.root / self.analysis.work_directory).resolve()

    @property
    def output_file(self) -> Path:
        return (self.root / self.render.output).resolve()

    def validate(self) -> None:
        analysis = self.analysis
        render = self.render
        if not 0 <= analysis.detection_threshold <= 255:
            raise ConfigurationError("analysis.detection_threshold must be between 0 and 255")
        if analysis.minimum_component_pixels < 1:
            raise ConfigurationError("analysis.minimum_component_pixels must be positive")
        if analysis.blur_threshold <= 0:
            raise ConfigurationError("analysis.blur_threshold must be positive")
        if render.resolution < 64 or render.resolution % 2:
            raise ConfigurationError("render.resolution must be an even integer of at least 64")
        output_width, output_height = output_dimensions(render)
        if output_width < 64 or output_height < 64:
            raise ConfigurationError(
                "render aspect ratio produces an output smaller than 64 pixels"
            )
        if render.crop_size < 64:
            raise ConfigurationError("render.crop_size must be at least 64")
        if render.duration_seconds <= 0:
            raise ConfigurationError("render.duration_seconds must be positive")
        if render.frames_per_second < 1:
            raise ConfigurationError("render.frames_per_second must be positive")
        if render.timeline not in {"uniform", "linear", "logarithmic", "capped"}:
            raise ConfigurationError(
                "render.timeline must be uniform, linear, logarithmic, or capped"
            )
        if render.interpolation not in {"physical", "morph", "geometry", "crossfade"}:
            raise ConfigurationError(
                "render.interpolation must be physical, morph, geometry, or crossfade"
            )
        if render.source_anchors and render.interpolation != "physical":
            raise ConfigurationError(
                "render.source_anchors requires render.interpolation = 'physical'"
            )
        if render.ingress_infill_cutoff_seconds <= 0:
            raise ConfigurationError("render.ingress_infill_cutoff_seconds must be positive")
        if render.ingress_infill_minimum_gap_seconds <= 0:
            raise ConfigurationError(
                "render.ingress_infill_minimum_gap_seconds must be positive"
            )
        if render.ingress_infill_interval_seconds <= 0:
            raise ConfigurationError("render.ingress_infill_interval_seconds must be positive")
        if render.maximum_gap_seconds <= 0 or render.minimum_gap_seconds <= 0:
            raise ConfigurationError("render gap settings must be positive")
        if render.codec not in {"h264", "ffv1"}:
            raise ConfigurationError("render.codec must be h264 or ffv1")
        output_suffix = Path(render.output).suffix.lower()
        if render.codec == "ffv1" and output_suffix not in {".mkv", ".avi"}:
            raise ConfigurationError("render.codec = 'ffv1' requires an .mkv or .avi output")
        if not 0 <= render.crf <= 51:
            raise ConfigurationError("render.crf must be between 0 and 51")


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return value


def aspect_ratio_parts(value: str) -> tuple[int, int]:
    """Parse a width:height aspect ratio."""
    try:
        width_text, height_text = value.split(":", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (AttributeError, ValueError) as error:
        raise ConfigurationError(
            f"render.aspect_ratio must use positive integers such as 1:1 or 4:5, got {value!r}"
        ) from error
    if width <= 0 or height <= 0:
        raise ConfigurationError("render.aspect_ratio values must be positive")
    return width, height


def output_dimensions(render: RenderConfig) -> tuple[int, int]:
    """Return codec-safe output dimensions, treating resolution as the width."""
    ratio_width, ratio_height = aspect_ratio_parts(render.aspect_ratio)
    height = round(render.resolution * ratio_height / ratio_width)
    if height % 2:
        height += 1
    return render.resolution, height
