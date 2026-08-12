"""Serializable analysis models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    sequence: int
    filename: str
    captured_at: datetime
    width: int
    height: int
    center_x: float
    center_y: float
    radius: float
    moon_center_x: float
    moon_center_y: float
    moon_radius: float
    moon_fit_error: float
    bright_pixels: int
    sharpness: float
    blurry: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["captured_at"] = self.captured_at.isoformat()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FrameAnalysis:
        data = dict(value)
        data["captured_at"] = datetime.fromisoformat(data["captured_at"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    schema_version: int
    source_directory: str
    pattern: str
    detection_threshold: int
    blur_threshold: float
    median_radius: float
    frames: tuple[FrameAnalysis, ...]

    def write(self, filename: Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "source_directory": self.source_directory,
            "pattern": self.pattern,
            "detection_threshold": self.detection_threshold,
            "blur_threshold": self.blur_threshold,
            "median_radius": self.median_radius,
            "frames": [frame.to_dict() for frame in self.frames],
        }
        filename.parent.mkdir(parents=True, exist_ok=True)
        temporary = filename.with_suffix(filename.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(filename)

    @classmethod
    def read(cls, filename: Path) -> AnalysisReport:
        try:
            payload = json.loads(filename.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Analysis report not found: {filename}") from error
        if payload.get("schema_version") != 2:
            raise ValueError(f"Unsupported analysis report version in {filename}")
        payload["frames"] = tuple(FrameAnalysis.from_dict(item) for item in payload["frames"])
        return cls(**payload)
