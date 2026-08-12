"""Alignment, temporal interpolation, and MP4 encoding."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from eclipse_timelapse.config import ProjectConfig, RenderConfig
from eclipse_timelapse.model import AnalysisReport, FrameAnalysis
from eclipse_timelapse.timeline import frame_blend, timeline_positions

ProgressCallback = Callable[[str], None]


class RenderError(RuntimeError):
    """Raised when aligned rendering or video encoding fails."""


def align_frame(
    image: np.ndarray,
    frame: FrameAnalysis,
    *,
    resolution: int,
    crop_size: int,
) -> np.ndarray:
    """Centre the detected solar disc and produce a square RGB frame."""
    scale = resolution / crop_size
    transform = np.asarray(
        [
            [scale, 0.0, resolution / 2.0 - scale * frame.center_x],
            [0.0, scale, resolution / 2.0 - scale * frame.center_y],
        ],
        dtype=np.float64,
    )
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    aligned_bgr = cv2.warpAffine(
        image,
        transform,
        (resolution, resolution),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)


class _AlignedFrameCache:
    def __init__(
        self,
        source_directory: Path,
        frames: tuple[FrameAnalysis, ...],
        render: RenderConfig,
        capacity: int = 4,
    ) -> None:
        self.source_directory = source_directory
        self.frames = frames
        self.render = render
        self.capacity = capacity
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.mask_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.distance_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        coordinates = np.arange(render.resolution, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(coordinates, coordinates)
        self.scale = render.resolution / render.crop_size
        self.solar_mask: np.ndarray | None = None
        self.atlas: np.ndarray | None = None

    def get(self, index: int) -> np.ndarray:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        frame = self.frames[index]
        image = cv2.imread(str(self.source_directory / frame.filename), cv2.IMREAD_COLOR)
        if image is None:
            raise RenderError(f"Could not read {frame.filename} during rendering")
        aligned = align_frame(
            image,
            frame,
            resolution=self.render.resolution,
            crop_size=self.render.crop_size,
        )
        self.cache[index] = aligned
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return aligned

    def visible_mask(self, index: int) -> np.ndarray:
        if index in self.mask_cache:
            self.mask_cache.move_to_end(index)
            return self.mask_cache[index]
        # Use observed luminance for source availability. A circle fitted to a
        # short arc can slightly overestimate the visible region; treating that
        # synthetic region as real would dilute the crescent with black pixels.
        luminance = np.max(self.get(index), axis=2).astype(np.float32)
        visible = np.asarray(luminance >= 30.0, dtype=np.float32)
        self.mask_cache[index] = visible
        if len(self.mask_cache) > self.capacity:
            self.mask_cache.popitem(last=False)
        return visible

    def signed_distance(self, index: int) -> np.ndarray:
        if index in self.distance_cache:
            self.distance_cache.move_to_end(index)
            return self.distance_cache[index]
        binary = np.uint8(self.visible_mask(index) >= 0.25)
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distance = inside - outside
        self.distance_cache[index] = distance
        if len(self.distance_cache) > self.capacity:
            self.distance_cache.popitem(last=False)
        return distance

    def build_atlas(self, progress: ProgressCallback | None = None) -> np.ndarray:
        """Reconstruct available solar texture for newly revealed morph pixels."""
        if self.atlas is not None:
            return self.atlas
        atlas = np.zeros((self.render.resolution, self.render.resolution, 3), dtype=np.uint8)
        best_luminance = np.zeros((self.render.resolution, self.render.resolution), dtype=np.uint8)
        for index, _frame in enumerate(self.frames):
            image = self.get(index)
            luminance = np.max(image, axis=2)
            replace_pixels = (luminance > 20) & (luminance > best_luminance)
            atlas[replace_pixels] = image[replace_pixels]
            best_luminance[replace_pixels] = luminance[replace_pixels]
            if progress and (index % 10 == 0 or index + 1 == len(self.frames)):
                progress(f"Building texture atlas {index + 1:03d}/{len(self.frames):03d}")
        self.atlas = atlas
        return atlas


def render_project(
    config: ProjectConfig,
    report: AnalysisReport,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Render a configured MP4 and a machine-readable render report."""
    output_file = config.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output_file):
        return _render_project_unlocked(config, report, progress=progress)


def _render_project_unlocked(
    config: ProjectConfig,
    report: AnalysisReport,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    render = config.render
    included = tuple(
        frame for frame in report.frames if not (render.exclude_blurry and frame.blurry)
    )
    excluded = tuple(frame for frame in report.frames if render.exclude_blurry and frame.blurry)
    if len(included) < 2:
        raise RenderError("Fewer than two usable frames remain after filtering")

    positions = timeline_positions(
        [frame.captured_at for frame in included],
        mode=render.timeline,
        minimum_gap_seconds=render.minimum_gap_seconds,
        maximum_gap_seconds=render.maximum_gap_seconds,
    )
    output_file = config.output_file
    temporary_file = output_file.with_name(f".{output_file.stem}.partial{output_file.suffix}")
    cache = _AlignedFrameCache(config.input_directory, included, render)
    if render.interpolation == "morph":
        cache.build_atlas(progress)
    total_frames = max(2, round(render.duration_seconds * render.frames_per_second))

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{render.resolution}x{render.resolution}",
        "-r",
        str(render.frames_per_second),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        render.preset,
        "-crf",
        str(render.crf),
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        str(temporary_file),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    poster: np.ndarray | None = None
    try:
        assert process.stdin is not None
        for output_index in range(total_frames):
            video_progress = output_index / (total_frames - 1)
            left, right, alpha = frame_blend(positions, video_progress)
            left_image = cache.get(left)
            if right == left or alpha <= 0:
                image = left_image
            elif alpha >= 1:
                image = cache.get(right)
            elif render.interpolation == "morph":
                image = _distance_morph_blend(
                    left_image,
                    cache.get(right),
                    left,
                    right,
                    alpha,
                    cache,
                )
            elif render.interpolation == "geometry":
                image = _geometry_blend(
                    left_image,
                    cache.get(right),
                    included[left],
                    included[right],
                    alpha,
                    cache,
                    report.median_radius,
                )
            else:
                image = cv2.addWeighted(left_image, 1.0 - alpha, cache.get(right), alpha, 0.0)
            if output_index == total_frames // 2:
                poster = image.copy()
            process.stdin.write(image.tobytes())
            if progress and (
                output_index % render.frames_per_second == 0 or output_index + 1 == total_frames
            ):
                progress(f"Rendering frame {output_index + 1:04d}/{total_frames:04d}")
        process.stdin.close()
        return_code = process.wait()
        error_output = (
            process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        )
    except (BrokenPipeError, OSError) as error:
        process.kill()
        process.wait()
        error_output = (
            process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        )
        temporary_file.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg failed while encoding: {error_output or error}") from error
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        temporary_file.unlink(missing_ok=True)
        raise
    if return_code != 0:
        temporary_file.unlink(missing_ok=True)
        raise RenderError(f"FFmpeg exited with status {return_code}: {error_output}")

    temporary_file.replace(output_file)
    if poster is not None:
        poster_file = output_file.with_name(f"{output_file.stem}-poster.jpg")
        cv2.imwrite(
            str(poster_file),
            cv2.cvtColor(poster, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    _write_render_report(output_file, render, included, excluded, total_frames)
    return output_file


@contextmanager
def _exclusive_output_lock(output_file: Path) -> Iterator[None]:
    """Prevent concurrent encoders from racing on an output and its report."""
    lock_file = output_file.with_name(f".{output_file.name}.lock")
    descriptor: int | None = None
    for _attempt in range(2):
        try:
            descriptor = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(descriptor, f"{os.getpid()}\n".encode())
            break
        except FileExistsError as error:
            try:
                owner_pid = int(lock_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                owner_pid = None
            if owner_pid is not None and not _process_is_running(owner_pid):
                lock_file.unlink(missing_ok=True)
                continue
            owner = f" by process {owner_pid}" if owner_pid is not None else ""
            raise RenderError(f"Output is already being rendered{owner}: {output_file}") from error
    if descriptor is None:
        raise RenderError(f"Could not acquire render lock for {output_file}")
    try:
        yield
    finally:
        os.close(descriptor)
        lock_file.unlink(missing_ok=True)


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _distance_morph_blend(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_index: int,
    right_index: int,
    alpha: float,
    cache: _AlignedFrameCache,
) -> np.ndarray:
    """Interpolate observed crescent silhouettes without double-exposed edges."""
    left_distance = cache.signed_distance(left_index)
    right_distance = cache.signed_distance(right_index)
    target_distance = (1.0 - alpha) * left_distance + alpha * right_distance
    target_mask = np.clip((target_distance + 1.0) / 2.0, 0.0, 1.0)

    left_mask = cache.visible_mask(left_index)
    right_mask = cache.visible_mask(right_index)
    left_weight = (1.0 - alpha) * left_mask
    right_weight = alpha * right_mask
    total_weight = left_weight + right_weight
    safe_weight = np.maximum(total_weight, 1e-6)
    blended = (
        left_image.astype(np.float32) * left_weight[:, :, None]
        + right_image.astype(np.float32) * right_weight[:, :, None]
    ) / safe_weight[:, :, None]

    missing = total_weight < 0.05
    if np.any(missing):
        assert cache.atlas is not None
        blended[missing] = cache.atlas[missing]
    blended *= target_mask[:, :, None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _geometry_blend(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_frame: FrameAnalysis,
    right_frame: FrameAnalysis,
    alpha: float,
    cache: _AlignedFrameCache,
    solar_radius: float,
) -> np.ndarray:
    """Morph the fitted lunar disc while blending only genuinely visible pixels."""
    if cache.solar_mask is None:
        cache.solar_mask = _disc_coverage(
            cache.grid_x,
            cache.grid_y,
            cache.render.resolution / 2.0,
            cache.render.resolution / 2.0,
            solar_radius * cache.scale,
        )
    left_mask = cache.visible_mask(cache.frames.index(left_frame))
    right_mask = cache.visible_mask(cache.frames.index(right_frame))
    left_x, left_y, left_radius = _aligned_moon_geometry(left_frame, cache.render)
    right_x, right_y, right_radius = _aligned_moon_geometry(right_frame, cache.render)
    moon_x = left_x + (right_x - left_x) * alpha
    moon_y = left_y + (right_y - left_y) * alpha
    moon_radius = left_radius + (right_radius - left_radius) * alpha
    target_moon = _disc_coverage(cache.grid_x, cache.grid_y, moon_x, moon_y, moon_radius)
    assert cache.solar_mask is not None
    target_mask = cache.solar_mask * (1.0 - target_moon)

    left_weight = (1.0 - alpha) * left_mask
    right_weight = alpha * right_mask
    total_weight = left_weight + right_weight
    safe_weight = np.maximum(total_weight, 1e-6)
    blended = (
        left_image.astype(np.float32) * left_weight[:, :, None]
        + right_image.astype(np.float32) * right_weight[:, :, None]
    ) / safe_weight[:, :, None]
    blended *= target_mask[:, :, None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _aligned_moon_geometry(
    frame: FrameAnalysis,
    render: RenderConfig,
) -> tuple[float, float, float]:
    scale = render.resolution / render.crop_size
    center = render.resolution / 2.0
    return (
        center + (frame.moon_center_x - frame.center_x) * scale,
        center + (frame.moon_center_y - frame.center_y) * scale,
        frame.moon_radius * scale,
    )


def _disc_coverage(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    distance = np.hypot(grid_x - center_x, grid_y - center_y)
    # A two-pixel transition is a stable approximation of the source anti-aliasing.
    return np.clip((radius + 1.0 - distance) / 2.0, 0.0, 1.0)


def _write_render_report(
    output_file: Path,
    render: RenderConfig,
    included: tuple[FrameAnalysis, ...],
    excluded: tuple[FrameAnalysis, ...],
    total_frames: int,
) -> None:
    digest = hashlib.sha256()
    with output_file.open("rb") as video_file:
        for block in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(block)
    payload = {
        "output": output_file.name,
        "sha256": digest.hexdigest(),
        "parameters": asdict(render),
        "encoded_frames": total_frames,
        "included_source_frames": [frame.filename for frame in included],
        "excluded_blurry_frames": [frame.filename for frame in excluded],
    }
    report_file = output_file.with_suffix(".json")
    report_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
