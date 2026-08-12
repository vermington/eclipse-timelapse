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
    output: str = "output/eclipse_timelapse_1080.mp4"
    resolution: int = 1080
    crop_size: int = 2400
    duration_seconds: float = 15.0
    frames_per_second: int = 30
    timeline: str = "logarithmic"
    interpolation: str = "morph"
    maximum_gap_seconds: float = 30.0
    minimum_gap_seconds: float = 1.0
    exclude_blurry: bool = True
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
        if render.interpolation not in {"morph", "geometry", "crossfade"}:
            raise ConfigurationError("render.interpolation must be morph, geometry, or crossfade")
        if render.maximum_gap_seconds <= 0 or render.minimum_gap_seconds <= 0:
            raise ConfigurationError("render gap settings must be positive")
        if not 0 <= render.crf <= 51:
            raise ConfigurationError("render.crf must be between 0 and 51")


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return value
