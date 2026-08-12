"""Image discovery, solar-disc detection, and sharpness analysis."""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageOps

from eclipse_timelapse.config import ProjectConfig
from eclipse_timelapse.model import AnalysisReport, FrameAnalysis

ProgressCallback = Callable[[str], None]
EXIF_IFD = ExifTags.IFD.Exif
DATETIME_ORIGINAL = 36867
SUBSECOND_ORIGINAL = 37521


class AnalysisError(RuntimeError):
    """Raised when an input frame cannot be analysed safely."""


def read_capture_time(filename: Path) -> datetime:
    """Return EXIF DateTimeOriginal, including sub-second data when present."""
    with Image.open(filename) as image:
        exif = image.getexif()
        exif_ifd = exif.get_ifd(EXIF_IFD)
        value = exif_ifd.get(DATETIME_ORIGINAL) or exif.get(306)
        subseconds = exif_ifd.get(SUBSECOND_ORIGINAL)
    if not value:
        raise AnalysisError(f"No EXIF capture timestamp in {filename.name}")
    try:
        captured_at = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except ValueError as error:
        raise AnalysisError(
            f"Invalid EXIF capture timestamp in {filename.name}: {value}"
        ) from error
    if subseconds is not None:
        digits = "".join(character for character in str(subseconds) if character.isdigit())[:6]
        if digits:
            captured_at = captured_at.replace(microsecond=int(digits.ljust(6, "0")))
    return captured_at


def detect_frame(
    filename: Path,
    *,
    threshold: int,
    minimum_component_pixels: int,
) -> dict[str, float | int]:
    """Detect the solar limb and measure edge sharpness in one photograph."""
    gray = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise AnalysisError(f"OpenCV could not read {filename}")

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        raise AnalysisError(f"No bright component detected in {filename.name}")
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bright_pixels = int(stats[component_index, cv2.CC_STAT_AREA])
    if bright_pixels < minimum_component_pixels:
        raise AnalysisError(
            f"Bright component in {filename.name} is too small ({bright_pixels} pixels)"
        )

    component = np.uint8(labels == component_index) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    contour_points = contour[:, 0, :].astype(np.float64)
    distance_from_solar_center = np.hypot(
        contour_points[:, 0] - center_x,
        contour_points[:, 1] - center_y,
    )
    inner_limb_points = contour_points[distance_from_solar_center < radius - 8.0]
    moon_center_x, moon_center_y, moon_radius, moon_fit_error = _fit_circle_ransac(
        inner_limb_points,
        boundary_points=contour_points,
        minimum_radius=radius * 0.90,
        maximum_radius=radius * 1.15,
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    boundary = cv2.subtract(cv2.dilate(component, kernel), cv2.erode(component, kernel)).astype(
        bool
    )
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    boundary_gradient = cv2.magnitude(gradient_x, gradient_y)[boundary]
    bright_values = gray[component.astype(bool)]
    brightness = max(float(np.median(bright_values)), 1.0)
    sharpness = float(np.percentile(boundary_gradient, 90) / brightness)

    height, width = gray.shape
    return {
        "width": width,
        "height": height,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "radius": float(radius),
        "moon_center_x": moon_center_x,
        "moon_center_y": moon_center_y,
        "moon_radius": moon_radius,
        "moon_fit_error": moon_fit_error,
        "bright_pixels": bright_pixels,
        "sharpness": sharpness,
    }


def analyse_project(
    config: ProjectConfig,
    *,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    """Analyse all matching photographs and write JSON, CSV, and a contact sheet."""
    input_files = sorted(config.input_directory.glob(config.input.pattern))
    if not input_files:
        raise AnalysisError(
            f"No files matching {config.input.pattern!r} in {config.input_directory}"
        )

    timestamped = [(read_capture_time(filename), filename) for filename in input_files]
    timestamped.sort(key=lambda item: (item[0], item[1].name))
    frames: list[FrameAnalysis] = []
    for sequence, (captured_at, filename) in enumerate(timestamped, start=1):
        if progress:
            progress(f"Analysing {sequence:03d}/{len(timestamped):03d}: {filename.name}")
        metrics = detect_frame(
            filename,
            threshold=config.analysis.detection_threshold,
            minimum_component_pixels=config.analysis.minimum_component_pixels,
        )
        sharpness = float(metrics.pop("sharpness"))
        frames.append(
            FrameAnalysis(
                sequence=sequence,
                filename=filename.name,
                captured_at=captured_at,
                sharpness=sharpness,
                blurry=sharpness < config.analysis.blur_threshold,
                **metrics,
            )
        )

    report = AnalysisReport(
        schema_version=2,
        source_directory=config.input.directory,
        pattern=config.input.pattern,
        detection_threshold=config.analysis.detection_threshold,
        blur_threshold=config.analysis.blur_threshold,
        median_radius=float(np.median([frame.radius for frame in frames])),
        frames=tuple(frames),
    )
    work_directory = config.work_directory
    work_directory.mkdir(parents=True, exist_ok=True)
    report.write(work_directory / "analysis.json")
    _write_csv(report, work_directory / "analysis.csv")
    _write_contact_sheet(report, config.input_directory, work_directory / "contact-sheet.jpg")
    return report


def _write_csv(report: AnalysisReport, filename: Path) -> None:
    fields = [
        "sequence",
        "filename",
        "captured_at",
        "width",
        "height",
        "center_x",
        "center_y",
        "radius",
        "moon_center_x",
        "moon_center_y",
        "moon_radius",
        "moon_fit_error",
        "bright_pixels",
        "sharpness",
        "blurry",
    ]
    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for frame in report.frames:
            row = frame.to_dict()
            row["center_x"] = f"{frame.center_x:.3f}"
            row["center_y"] = f"{frame.center_y:.3f}"
            row["radius"] = f"{frame.radius:.3f}"
            row["moon_center_x"] = f"{frame.moon_center_x:.3f}"
            row["moon_center_y"] = f"{frame.moon_center_y:.3f}"
            row["moon_radius"] = f"{frame.moon_radius:.3f}"
            row["moon_fit_error"] = f"{frame.moon_fit_error:.4f}"
            row["sharpness"] = f"{frame.sharpness:.4f}"
            writer.writerow(row)


def _write_contact_sheet(
    report: AnalysisReport,
    source_directory: Path,
    filename: Path,
) -> None:
    columns = 4
    tile_size = 300
    label_height = 42
    rows = (len(report.frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "#090909")
    for index, frame in enumerate(report.frames):
        with Image.open(source_directory / frame.filename) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            half = 600
            crop = source.crop(
                (
                    round(frame.center_x - half),
                    round(frame.center_y - half),
                    round(frame.center_x + half),
                    round(frame.center_y + half),
                )
            )
            crop = crop.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size + label_height), "#151515")
        tile.paste(crop, (0, 0))
        draw = ImageDraw.Draw(tile)
        status = "  BLUR" if frame.blurry else ""
        colour = "#ff7066" if frame.blurry else "#eeeeee"
        draw.text(
            (8, tile_size + 5), f"{frame.sequence:03d}  {frame.sharpness:.2f}{status}", fill=colour
        )
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        sheet.paste(tile, (x, y))
    filename.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(filename, quality=92, optimize=True)


def _fit_circle_ransac(
    points: np.ndarray,
    *,
    boundary_points: np.ndarray,
    minimum_radius: float,
    maximum_radius: float,
) -> tuple[float, float, float, float]:
    """Fit the occulting lunar limb while rejecting points from the solar limb."""
    if len(points) < 20:
        raise AnalysisError("Too few inner-limb points to fit the lunar disc")
    generator = np.random.default_rng(20260812)
    best_inliers: np.ndarray | None = None
    best_count = 0
    for indices in generator.integers(0, len(points), size=(2_000, 3)):
        candidate = _circle_from_three_points(points[indices])
        if candidate is None:
            continue
        candidate_x, candidate_y, candidate_radius = candidate
        if not minimum_radius <= candidate_radius <= maximum_radius:
            continue
        # The illuminated crescent must be entirely outside the lunar disc.
        # This disambiguates the similarly sized solar and lunar arcs.
        boundary_clearance = (
            np.hypot(
                boundary_points[:, 0] - candidate_x,
                boundary_points[:, 1] - candidate_y,
            )
            - candidate_radius
        )
        if float(np.mean(boundary_clearance < -2.0)) > 0.03:
            continue
        residuals = np.abs(
            np.hypot(points[:, 0] - candidate_x, points[:, 1] - candidate_y) - candidate_radius
        )
        inliers = residuals < 1.5
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 20:
        raise AnalysisError("Could not fit a reliable lunar limb")

    inlier_points = points[best_inliers]
    design = np.column_stack(
        (2.0 * inlier_points[:, 0], 2.0 * inlier_points[:, 1], np.ones(best_count))
    )
    target = inlier_points[:, 0] ** 2 + inlier_points[:, 1] ** 2
    center_x, center_y, constant = np.linalg.lstsq(design, target, rcond=None)[0]
    radius = float(np.sqrt(constant + center_x**2 + center_y**2))
    residual = float(
        np.median(
            np.abs(
                np.hypot(
                    inlier_points[:, 0] - center_x,
                    inlier_points[:, 1] - center_y,
                )
                - radius
            )
        )
    )
    return float(center_x), float(center_y), radius, residual


def _circle_from_three_points(points: np.ndarray) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = points
    divisor = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(divisor) < 1e-9:
        return None
    center_x = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / divisor
    center_y = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / divisor
    radius = float(np.hypot(x1 - center_x, y1 - center_y))
    return float(center_x), float(center_y), radius
