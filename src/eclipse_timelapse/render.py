"""Alignment, temporal interpolation, and MP4 encoding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from eclipse_timelapse.config import ProjectConfig, RenderConfig, output_dimensions
from eclipse_timelapse.model import AnalysisReport, EclipseModel, FrameAnalysis
from eclipse_timelapse.timeline import frame_blend, timeline_positions

ProgressCallback = Callable[[str], None]


class RenderError(RuntimeError):
    """Raised when aligned rendering or video encoding fails."""


@dataclass(frozen=True, slots=True)
class _DetailFeature:
    """One confidently detected high-frequency solar feature."""

    frame_index: int
    elapsed_seconds: float
    x: float
    y: float
    strength: float


@dataclass(frozen=True, slots=True)
class _DetailMotion:
    """Linear apparent motion of solar detail in output pixels."""

    reference_time: datetime
    velocity_x_pixels_per_second: float = 0.0
    velocity_y_pixels_per_second: float = 0.0
    supporting_frames: int = 0
    time_span_seconds: float = 0.0
    median_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class _SourceAnchor:
    """One source photograph placed on a unique output frame."""

    source_index: int
    ideal_frame: float
    output_frame: int
    timing_offset_seconds: float


@dataclass(frozen=True, slots=True)
class _IngressInfillState:
    """One sparse subtractive boundary state inside a source gap."""

    output_frame: int
    gap_fraction: float


@dataclass(frozen=True, slots=True)
class _IngressInfillGap:
    """A source interval eligible for sparse ingress-only infill."""

    left_anchor: _SourceAnchor
    right_anchor: _SourceAnchor
    states: tuple[_IngressInfillState, ...]
    active_end_frame: int


def align_frame(
    image: np.ndarray,
    frame: FrameAnalysis,
    *,
    output_width: int,
    output_height: int,
    crop_size: int,
) -> np.ndarray:
    """Centre the detected solar disc and produce one RGB frame."""
    scale = output_width / crop_size
    transform = np.asarray(
        [
            [scale, 0.0, output_width / 2.0 - scale * frame.center_x],
            [0.0, scale, output_height / 2.0 - scale * frame.center_y],
        ],
        dtype=np.float64,
    )
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    aligned_bgr = cv2.warpAffine(
        image,
        transform,
        (output_width, output_height),
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
        self.output_width, self.output_height = output_dimensions(render)
        x_coordinates = np.arange(self.output_width, dtype=np.float32)
        y_coordinates = np.arange(self.output_height, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x_coordinates, y_coordinates)
        self.scale = render.resolution / render.crop_size
        self.solar_mask: np.ndarray | None = None
        self.atlas: np.ndarray | None = None
        self.atlas_brightness: float | None = None
        self.atlas_chroma: np.ndarray | None = None
        self.atlas_detail: np.ndarray | None = None
        self.detail_motion = _DetailMotion(reference_time=frames[0].captured_at)

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
            output_width=self.output_width,
            output_height=self.output_height,
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
        atlas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        best_luminance = np.zeros((self.output_height, self.output_width), dtype=np.uint8)
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

    def prepare_physical_atlas(
        self,
        solar_radius: float,
        progress: ProgressCallback | None = None,
        *,
        skip_blurry: bool = False,
    ) -> np.ndarray:
        """Build a colour-consistent texture whose reliable detail moves over time."""
        self.prepare_solar_mask(solar_radius)
        assert self.solar_mask is not None
        solar_pixels = self.solar_mask > 0.0

        detail_blur_sigma = max(2.0, solar_radius * self.scale * 0.04)
        detail_edge_margin = max(2.0, solar_radius * self.scale * 0.04)
        reference_time = self.frames[0].captured_at
        features: list[_DetailFeature] = []
        frame_chromas: list[np.ndarray] = []
        for index, frame in enumerate(self.frames):
            if skip_blurry and frame.blurry:
                continue
            image = self.get(index)
            layer = _extract_solar_detail(
                image,
                frame,
                solar_pixels,
                blur_sigma=detail_blur_sigma,
                edge_margin=detail_edge_margin,
            )
            if layer is not None:
                detail, weight, chroma = layer
                frame_chromas.append(chroma)
                elapsed_seconds = (frame.captured_at - reference_time).total_seconds()
                features.extend(
                    _detect_detail_features(
                        detail,
                        weight,
                        frame_index=index,
                        elapsed_seconds=elapsed_seconds,
                        center_x=self.output_width / 2.0,
                        center_y=self.output_height / 2.0,
                        solar_radius=solar_radius * self.scale,
                    )
                )
            if progress and (index % 10 == 0 or index + 1 == len(self.frames)):
                progress(f"Tracking solar detail {index + 1:03d}/{len(self.frames):03d}")

        if not frame_chromas:
            raise RenderError("Could not estimate a usable solar colour")
        output_solar_radius = solar_radius * self.scale
        self.detail_motion = _fit_detail_motion(
            features,
            reference_time=reference_time,
            position_tolerance=max(3.0, output_solar_radius * 0.032),
            maximum_speed=max(0.02, output_solar_radius * 0.00028),
        )

        detail_sum = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        detail_weight = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        for index, frame in enumerate(self.frames):
            if skip_blurry and frame.blurry:
                continue
            image = self.get(index)
            layer = _extract_solar_detail(
                image,
                frame,
                solar_pixels,
                blur_sigma=detail_blur_sigma,
                edge_margin=detail_edge_margin,
            )
            if layer is not None:
                detail, weight, _chroma = layer
                elapsed_seconds = (frame.captured_at - reference_time).total_seconds()
                transform = np.asarray(
                    [
                        [
                            1.0,
                            0.0,
                            -self.detail_motion.velocity_x_pixels_per_second * elapsed_seconds,
                        ],
                        [
                            0.0,
                            1.0,
                            -self.detail_motion.velocity_y_pixels_per_second * elapsed_seconds,
                        ],
                    ],
                    dtype=np.float32,
                )
                canonical_detail = cv2.warpAffine(
                    detail,
                    transform,
                    (self.output_width, self.output_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                canonical_weight = cv2.warpAffine(
                    weight,
                    transform,
                    (self.output_width, self.output_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                detail_sum += canonical_detail * canonical_weight
                detail_weight += canonical_weight
            if progress and (index % 10 == 0 or index + 1 == len(self.frames)):
                progress(f"Building moving texture atlas {index + 1:03d}/{len(self.frames):03d}")

        if not np.any(detail_weight > 0):
            raise RenderError("Could not reconstruct a usable solar texture atlas")

        self.atlas_detail = detail_sum / np.maximum(detail_weight, 1e-6)
        atlas_luminance = 128.0 + self.atlas_detail
        atlas_luminance[~solar_pixels] = 0.0
        atlas_gray = np.uint8(np.clip(np.rint(atlas_luminance), 0, 255))
        atlas = np.repeat(atlas_gray[:, :, None], 3, axis=2)

        atlas_chroma = np.median(np.asarray(frame_chromas), axis=0)
        chroma_luminance = float(
            0.299 * atlas_chroma[0] + 0.587 * atlas_chroma[1] + 0.114 * atlas_chroma[2]
        )
        self.atlas_chroma = atlas_chroma / chroma_luminance
        self.atlas_brightness = max(float(np.median(atlas_luminance[solar_pixels])), 1.0)
        self.atlas = atlas
        return atlas

    def prepare_solar_mask(self, solar_radius: float) -> np.ndarray:
        """Prepare only the aligned solar-disc mask without reconstructing texture."""
        self.solar_mask = _disc_coverage(
            self.grid_x,
            self.grid_y,
            self.output_width / 2.0,
            self.output_height / 2.0,
            solar_radius * self.scale,
        )
        return self.solar_mask


def _extract_solar_detail(
    image: np.ndarray,
    frame: FrameAnalysis,
    solar_pixels: np.ndarray,
    *,
    blur_sigma: float,
    edge_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Separate reliable high-frequency detail from exposure and colour."""
    luminance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    visible_pixels = solar_pixels & (luminance >= 30.0)
    if not np.any(visible_pixels):
        return None

    normalized_luminance = np.clip(
        luminance * (128.0 / max(frame.brightness, 1.0)),
        0.0,
        255.0,
    )
    visible_float = visible_pixels.astype(np.float32)
    local_average = cv2.GaussianBlur(
        normalized_luminance * visible_float,
        (0, 0),
        blur_sigma,
    ) / np.maximum(
        cv2.GaussianBlur(visible_float, (0, 0), blur_sigma),
        1e-3,
    )
    detail = np.clip(normalized_luminance - local_average, -24.0, 24.0)
    boundary_distance = cv2.distanceTransform(
        np.uint8(visible_pixels),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    weight = np.clip(boundary_distance - edge_margin, 0.0, 12.0)

    frame_colour = np.median(image[visible_pixels], axis=0)
    frame_colour_luminance = float(
        0.299 * frame_colour[0] + 0.587 * frame_colour[1] + 0.114 * frame_colour[2]
    )
    if frame_colour_luminance <= 0:
        return None
    return detail, weight, frame_colour / frame_colour_luminance


def _detect_detail_features(
    detail: np.ndarray,
    weight: np.ndarray,
    *,
    frame_index: int,
    elapsed_seconds: float,
    center_x: float,
    center_y: float,
    solar_radius: float,
) -> list[_DetailFeature]:
    """Detect compact dark features while excluding all photographed limbs."""
    candidate = np.uint8((detail <= -3.5) & (weight > 0.0))
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )
    scale_area = (solar_radius / 220.0) ** 2
    minimum_area = max(2, round(3.0 * scale_area))
    maximum_area = max(100, round(solar_radius**2 * 0.015))
    minimum_strength = max(12.0, 20.0 * scale_area)
    features: list[_DetailFeature] = []
    for component_index in range(1, count):
        area = int(statistics[component_index, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= maximum_area:
            continue
        component = labels == component_index
        strength = float(np.sum(-detail[component]))
        if strength < minimum_strength:
            continue
        feature_x, feature_y = centroids[component_index]
        features.append(
            _DetailFeature(
                frame_index=frame_index,
                elapsed_seconds=elapsed_seconds,
                x=float(feature_x - center_x),
                y=float(feature_y - center_y),
                strength=strength,
            )
        )
    features.sort(key=lambda feature: feature.strength, reverse=True)
    return features[:8]


def _fit_detail_motion(
    features: list[_DetailFeature],
    *,
    reference_time: datetime,
    position_tolerance: float = 7.0,
    maximum_speed: float = 0.06,
) -> _DetailMotion:
    """Fit the longest consistent feature track with deterministic RANSAC."""
    by_frame: dict[int, list[_DetailFeature]] = defaultdict(list)
    for feature in features:
        by_frame[feature.frame_index].append(feature)
    if len(by_frame) < 8 or len(features) < 8:
        return _DetailMotion(reference_time=reference_time)

    def select_track(
        velocity_x: float,
        velocity_y: float,
        intercept_x: float,
        intercept_y: float,
    ) -> list[_DetailFeature]:
        selected: list[_DetailFeature] = []
        for frame_features in by_frame.values():
            elapsed = frame_features[0].elapsed_seconds
            predicted_x = intercept_x + velocity_x * elapsed
            predicted_y = intercept_y + velocity_y * elapsed
            nearest = min(
                frame_features,
                key=lambda feature: np.hypot(
                    feature.x - predicted_x,
                    feature.y - predicted_y,
                ),
            )
            distance = np.hypot(nearest.x - predicted_x, nearest.y - predicted_y)
            if distance <= position_tolerance:
                selected.append(nearest)
        return selected

    generator = np.random.default_rng(20260812)
    best_track: list[_DetailFeature] = []
    best_score = (0, 0.0, 0.0)
    for first_index, second_index in generator.integers(
        0,
        len(features),
        size=(4_000, 2),
    ):
        first = features[int(first_index)]
        second = features[int(second_index)]
        elapsed_delta = second.elapsed_seconds - first.elapsed_seconds
        if first.frame_index == second.frame_index or abs(elapsed_delta) < 30.0:
            continue
        velocity_x = (second.x - first.x) / elapsed_delta
        velocity_y = (second.y - first.y) / elapsed_delta
        if np.hypot(velocity_x, velocity_y) > maximum_speed:
            continue
        intercept_x = first.x - velocity_x * first.elapsed_seconds
        intercept_y = first.y - velocity_y * first.elapsed_seconds
        track = select_track(velocity_x, velocity_y, intercept_x, intercept_y)
        if not track:
            continue
        time_span = max(feature.elapsed_seconds for feature in track) - min(
            feature.elapsed_seconds for feature in track
        )
        strength = float(sum(np.log1p(feature.strength) for feature in track))
        score = (len(track), time_span, strength)
        if score > best_score:
            best_score = score
            best_track = track

    if len(best_track) < 8:
        return _DetailMotion(reference_time=reference_time)

    velocity_x = velocity_y = intercept_x = intercept_y = 0.0
    for _iteration in range(3):
        elapsed = np.asarray([feature.elapsed_seconds for feature in best_track])
        design = np.column_stack((elapsed, np.ones(len(best_track))))
        x_values = np.asarray([feature.x for feature in best_track])
        y_values = np.asarray([feature.y for feature in best_track])
        velocity_x, intercept_x = np.linalg.lstsq(design, x_values, rcond=None)[0]
        velocity_y, intercept_y = np.linalg.lstsq(design, y_values, rcond=None)[0]
        best_track = select_track(
            float(velocity_x),
            float(velocity_y),
            float(intercept_x),
            float(intercept_y),
        )
        if len(best_track) < 8:
            return _DetailMotion(reference_time=reference_time)

    time_span = max(feature.elapsed_seconds for feature in best_track) - min(
        feature.elapsed_seconds for feature in best_track
    )
    residuals = [
        np.hypot(
            feature.x - (intercept_x + velocity_x * feature.elapsed_seconds),
            feature.y - (intercept_y + velocity_y * feature.elapsed_seconds),
        )
        for feature in best_track
    ]
    median_residual = float(np.median(residuals))
    if (
        time_span < 300.0
        or median_residual > position_tolerance * 0.75
        or np.hypot(velocity_x, velocity_y) > maximum_speed
    ):
        return _DetailMotion(reference_time=reference_time)
    return _DetailMotion(
        reference_time=reference_time,
        velocity_x_pixels_per_second=float(velocity_x),
        velocity_y_pixels_per_second=float(velocity_y),
        supporting_frames=len(best_track),
        time_span_seconds=float(time_span),
        median_residual=median_residual,
    )


def _schedule_source_anchors(
    positions: np.ndarray,
    *,
    total_frames: int,
    frames_per_second: int,
) -> tuple[_SourceAnchor, ...]:
    """Assign every source to a unique ordered frame with minimum squared error."""
    source_count = len(positions)
    if source_count > total_frames:
        raise RenderError(
            f"Cannot place {source_count} source anchors in only {total_frames} output frames"
        )
    if source_count < 2:
        raise RenderError("At least two source anchors are required")

    ideal_frames = np.asarray(positions, dtype=np.float64) * (total_frames - 1)
    costs = np.full(total_frames, np.inf, dtype=np.float64)
    costs[0] = ideal_frames[0] ** 2
    parents = np.full((source_count, total_frames), -1, dtype=np.int32)

    for source_index in range(1, source_count):
        prefix_best = np.empty(total_frames, dtype=np.int32)
        best_index = 0
        for output_frame in range(total_frames):
            if costs[output_frame] < costs[best_index]:
                best_index = output_frame
            prefix_best[output_frame] = best_index

        updated = np.full(total_frames, np.inf, dtype=np.float64)
        minimum_frame = source_index
        maximum_frame = total_frames - (source_count - source_index)
        if source_index == source_count - 1:
            minimum_frame = total_frames - 1
        for output_frame in range(minimum_frame, maximum_frame + 1):
            parent = int(prefix_best[output_frame - 1])
            if np.isfinite(costs[parent]):
                updated[output_frame] = costs[parent] + (
                    output_frame - ideal_frames[source_index]
                ) ** 2
                parents[source_index, output_frame] = parent
        costs = updated

    output_frame = total_frames - 1
    if not np.isfinite(costs[output_frame]):
        raise RenderError("Could not construct a strictly ordered source-anchor schedule")
    assigned_frames = [output_frame]
    for source_index in range(source_count - 1, 0, -1):
        output_frame = int(parents[source_index, output_frame])
        if output_frame < 0:
            raise RenderError("Source-anchor schedule backtracking failed")
        assigned_frames.append(output_frame)
    assigned_frames.reverse()

    return tuple(
        _SourceAnchor(
            source_index=source_index,
            ideal_frame=float(ideal_frames[source_index]),
            output_frame=assigned_frame,
            timing_offset_seconds=(assigned_frame - ideal_frames[source_index])
            / frames_per_second,
        )
        for source_index, assigned_frame in enumerate(assigned_frames)
    )


def _schedule_ingress_infill(
    anchors: tuple[_SourceAnchor, ...],
    *,
    total_frames: int,
    render: RenderConfig,
) -> tuple[_IngressInfillGap, ...]:
    """Schedule a small number of subtractive states before the configured cutoff."""
    if not render.ingress_infill:
        return ()

    frames_per_second = render.frames_per_second
    interval_frames = max(1, round(render.ingress_infill_interval_seconds * frames_per_second))
    cutoff_frame = min(
        total_frames,
        math.ceil(render.ingress_infill_cutoff_seconds * frames_per_second - 1e-9),
    )
    gaps = []
    for left_anchor, right_anchor in zip(anchors, anchors[1:], strict=False):
        gap_frames = right_anchor.output_frame - left_anchor.output_frame
        gap_seconds = gap_frames / frames_per_second
        if gap_seconds < render.ingress_infill_minimum_gap_seconds:
            continue
        active_end_frame = min(right_anchor.output_frame, cutoff_frame)
        state_frames = range(
            left_anchor.output_frame + interval_frames,
            active_end_frame,
            interval_frames,
        )
        states = tuple(
            _IngressInfillState(
                output_frame=output_frame,
                gap_fraction=(output_frame - left_anchor.output_frame) / gap_frames,
            )
            for output_frame in state_frames
        )
        if states:
            gaps.append(
                _IngressInfillGap(
                    left_anchor=left_anchor,
                    right_anchor=right_anchor,
                    states=states,
                    active_end_frame=active_end_frame,
                )
            )
    return tuple(gaps)


class _SourceAnchoredTimeline:
    """Render exact anchors with optional sparse, subtractive ingress infill."""

    def __init__(
        self,
        cache: _AlignedFrameCache,
        anchors: tuple[_SourceAnchor, ...],
        *,
        total_frames: int,
        stable_moon_radius: float | None = None,
    ) -> None:
        self.cache = cache
        self.anchors = anchors
        self.total_frames = total_frames
        self.output_frames = np.asarray(
            [anchor.output_frame for anchor in anchors],
            dtype=np.int32,
        )
        self.infill_gaps = _schedule_ingress_infill(
            anchors,
            total_frames=total_frames,
            render=cache.render,
        )
        self.infill_by_left_frame = {
            gap.left_anchor.output_frame: gap for gap in self.infill_gaps
        }
        self.infill_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.stable_moon_radius = (
            float(stable_moon_radius)
            if stable_moon_radius is not None
            else float(np.median([frame.moon_radius for frame in cache.frames])) * cache.scale
        )

    def render(self, output_frame: int) -> np.ndarray:
        """Return an exact source or a start-source image with added black occlusion."""
        insertion = int(np.searchsorted(self.output_frames, output_frame, side="left"))
        if insertion == 0:
            return self.cache.get(self.anchors[0].source_index).copy()
        if insertion == len(self.anchors):
            return self.cache.get(self.anchors[-1].source_index).copy()

        right = self.anchors[insertion]
        if right.output_frame == output_frame:
            return self.cache.get(right.source_index).copy()
        left = self.anchors[insertion - 1]
        infill_gap = self.infill_by_left_frame.get(left.output_frame)
        if infill_gap is not None and output_frame < infill_gap.active_end_frame:
            state_frames = [state.output_frame for state in infill_gap.states]
            state_index = int(np.searchsorted(state_frames, output_frame, side="right")) - 1
            if state_index >= 0:
                return self._render_infill_state(infill_gap, state_index).copy()
            return self.cache.get(left.source_index).copy()

        left_distance = output_frame - left.output_frame
        right_distance = right.output_frame - output_frame
        nearest = left if left_distance <= right_distance else right
        return self.cache.get(nearest.source_index).copy()

    def _render_infill_state(
        self,
        gap: _IngressInfillGap,
        state_index: int,
    ) -> np.ndarray:
        """Blacken only the cumulative lunar advance in the gap's starting image."""
        state = gap.states[state_index]
        key = (gap.left_anchor.source_index, state.output_frame)
        if key in self.infill_cache:
            self.infill_cache.move_to_end(key)
            return self.infill_cache[key]

        left_frame = self.cache.frames[gap.left_anchor.source_index]
        right_frame = self.cache.frames[gap.right_anchor.source_index]
        left_x, left_y, _ = _aligned_moon_geometry(left_frame, self.cache.render)
        right_x, right_y, _ = _aligned_moon_geometry(right_frame, self.cache.render)
        # Include the source's anti-aliased solar limb in the permitted region.
        # The source image already contains that limb coverage; the lunar mask
        # only needs to attenuate it smoothly rather than cutting it on another
        # whole-pixel boundary.
        solar_pixels = _disc_coverage(
            self.cache.grid_x,
            self.cache.grid_y,
            self.cache.output_width / 2.0,
            self.cache.output_height / 2.0,
            left_frame.radius * self.cache.scale,
        ) > 0.0
        new_occlusion = np.zeros(solar_pixels.shape, dtype=np.float32)
        for scheduled_state in gap.states[: state_index + 1]:
            fraction = scheduled_state.gap_fraction
            target_coverage = _disc_coverage(
                self.cache.grid_x,
                self.cache.grid_y,
                left_x + (right_x - left_x) * fraction,
                left_y + (right_y - left_y) * fraction,
                self.stable_moon_radius,
            )
            np.maximum(new_occlusion, target_coverage, out=new_occlusion)
        new_occlusion *= solar_pixels

        source = self.cache.get(gap.left_anchor.source_index)
        image = np.rint(
            source.astype(np.float32) * (1.0 - new_occlusion[:, :, None])
        ).astype(np.uint8)
        self.infill_cache[key] = image
        if len(self.infill_cache) > 2:
            self.infill_cache.popitem(last=False)
        return image

    def anchor_report(self) -> dict[str, object]:
        """Return an auditable record of source placement and pre-encode identity."""
        records = []
        for anchor in self.anchors:
            frame = self.cache.frames[anchor.source_index]
            aligned = self.cache.get(anchor.source_index)
            records.append(
                {
                    "filename": frame.filename,
                    "captured_at": frame.captured_at.isoformat(),
                    "blurry": frame.blurry,
                    "ideal_output_time_seconds": anchor.ideal_frame
                    / self.cache.render.frames_per_second,
                    "output_frame": anchor.output_frame,
                    "output_time_seconds": anchor.output_frame
                    / self.cache.render.frames_per_second,
                    "timing_offset_milliseconds": anchor.timing_offset_seconds * 1_000.0,
                    "aligned_rgb_sha256": hashlib.sha256(aligned.tobytes()).hexdigest(),
                }
            )
        offsets = np.asarray(
            [anchor.timing_offset_seconds for anchor in self.anchors],
            dtype=np.float64,
        )
        infill_records = []
        for gap in self.infill_gaps:
            left_frame = self.cache.frames[gap.left_anchor.source_index]
            right_frame = self.cache.frames[gap.right_anchor.source_index]
            infill_records.append(
                {
                    "start_filename": left_frame.filename,
                    "end_filename": right_frame.filename,
                    "start_output_frame": gap.left_anchor.output_frame,
                    "end_output_frame": gap.right_anchor.output_frame,
                    "gap_duration_seconds": (
                        gap.right_anchor.output_frame - gap.left_anchor.output_frame
                    )
                    / self.cache.render.frames_per_second,
                    "active_end_frame": gap.active_end_frame,
                    "states": [
                        {
                            "output_frame": state.output_frame,
                            "output_time_seconds": state.output_frame
                            / self.cache.render.frames_per_second,
                            "gap_fraction": state.gap_fraction,
                        }
                        for state in gap.states
                    ],
                }
            )
        distinct_infill_frames = sum(len(gap.states) for gap in self.infill_gaps)
        synthetic_output_frames = sum(
            gap.active_end_frame - gap.states[0].output_frame for gap in self.infill_gaps
        )
        has_infill = distinct_infill_frames > 0
        return {
            "policy": (
                "source-anchored-with-sparse-subtractive-ingress-infill"
                if has_infill
                else "aligned-source-pass-through"
            ),
            "intermediate_texture_policy": (
                "gap-start-source-darkened-only-by-new-lunar-subpixel-coverage"
                if has_infill
                else "nearest-complete-source-frame-hold"
            ),
            "synthetic_frame_count": distinct_infill_frames,
            "synthetic_distinct_frame_count": distinct_infill_frames,
            "synthetic_output_frame_count": synthetic_output_frames,
            "source_output_frame_count": self.total_frames - synthetic_output_frames,
            "pre_encode_all_frames_are_complete_sources": not has_infill,
            "pre_encode_all_anchor_frames_are_complete_sources": True,
            "pre_encode_all_unoccluded_pixels_match_gap_start_sources": True,
            "pre_encode_all_surviving_pixels_match_gap_start_sources": not has_infill,
            "all_synthetic_frames_derived_from_gap_start": True,
            "all_synthetic_pixel_changes_are_blackening_only": not has_infill,
            "all_synthetic_pixel_changes_are_subtractive_occlusion_only": True,
            "synthetic_boundary_antialiasing": (
                "two-pixel subpixel lunar coverage band" if has_infill else None
            ),
            "synthetic_moon_radius_output_pixels": (
                self.stable_moon_radius if has_infill else None
            ),
            "post_cutoff_synthetic_frame_count": 0,
            "ingress_infill_cutoff_seconds": self.cache.render.ingress_infill_cutoff_seconds,
            "pre_encode_pixel_identity": True,
            "encoded_pixel_identity": self.cache.render.codec == "ffv1",
            "geometric_operations": [
                "solar-centre alignment",
                f"{self.cache.render.aspect_ratio} crop",
                "resize",
            ],
            "photometric_operations": (
                [
                    "attenuate newly occulted pixels by computed lunar coverage; "
                    "fully covered pixels become RGB (0, 0, 0)"
                ]
                if has_infill
                else []
            ),
            "infill_geometry": (
                "linear interpolation of detected endpoint lunar centres with one "
                "globally fitted constant lunar radius"
                if has_infill
                else None
            ),
            "infill_gaps": infill_records,
            "anchor_count": len(self.anchors),
            "all_source_frames_anchored": len(self.anchors) == len(self.cache.frames),
            "maximum_absolute_timing_offset_milliseconds": float(
                np.max(np.abs(offsets)) * 1_000.0
            ),
            "rms_timing_offset_milliseconds": float(
                np.sqrt(np.mean(offsets**2)) * 1_000.0
            ),
            "frames": records,
        }

def render_project(
    config: ProjectConfig,
    report: AnalysisReport,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Render a configured video and a machine-readable render report."""
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
    if render.source_anchors:
        included = report.frames
        excluded: tuple[FrameAnalysis, ...] = ()
        reconstruction_excluded: tuple[FrameAnalysis, ...] = ()
    else:
        included = tuple(
            frame for frame in report.frames if not (render.exclude_blurry and frame.blurry)
        )
        excluded = tuple(frame for frame in report.frames if render.exclude_blurry and frame.blurry)
        reconstruction_excluded = excluded
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
    if render.interpolation == "physical" and not render.source_anchors:
        cache.prepare_physical_atlas(
            report.eclipse_model.solar_radius,
            progress,
        )
    elif render.interpolation == "morph":
        cache.build_atlas(progress)
    total_frames = max(2, round(render.duration_seconds * render.frames_per_second))
    output_width, output_height = output_dimensions(render)
    anchors = (
        _schedule_source_anchors(
            positions,
            total_frames=total_frames,
            frames_per_second=render.frames_per_second,
        )
        if render.source_anchors
        else ()
    )
    source_timeline = (
        _SourceAnchoredTimeline(
            cache,
            anchors,
            total_frames=total_frames,
            stable_moon_radius=report.eclipse_model.moon_radius * cache.scale,
        )
        if anchors
        else None
    )

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
        f"{output_width}x{output_height}",
        "-r",
        str(render.frames_per_second),
        "-i",
        "-",
        "-an",
    ]
    if render.codec == "h264":
        command.extend(
            [
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
            ]
        )
    else:
        command.extend(
            [
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-pix_fmt",
                "gbrp",
            ]
        )
    command.append(str(temporary_file))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    poster: np.ndarray | None = None
    try:
        assert process.stdin is not None
        for output_index in range(total_frames):
            if source_timeline is not None:
                image = source_timeline.render(output_index)
            else:
                video_progress = output_index / (total_frames - 1)
                left, right, alpha = frame_blend(positions, video_progress)
                if render.interpolation == "physical":
                    image = _physical_frame(
                        included[left],
                        included[right],
                        alpha,
                        cache,
                        report.eclipse_model,
                    )
                else:
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
                        image = cv2.addWeighted(
                            left_image,
                            1.0 - alpha,
                            cache.get(right),
                            alpha,
                            0.0,
                        )
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
    detail_motion = (
        cache.detail_motion
        if render.interpolation == "physical" and not render.source_anchors
        else None
    )
    anchor_report = source_timeline.anchor_report() if source_timeline else None
    _write_render_report(
        output_file,
        render,
        included,
        excluded,
        total_frames,
        detail_motion=detail_motion,
        source_anchor_report=anchor_report,
        reconstruction_excluded=reconstruction_excluded,
    )
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


def _physical_frame(
    left_frame: FrameAnalysis,
    right_frame: FrameAnalysis,
    alpha: float,
    cache: _AlignedFrameCache,
    model: EclipseModel,
) -> np.ndarray:
    """Render one exposure from a fixed Sun and a clock-linear lunar trajectory."""
    assert cache.solar_mask is not None

    captured_at = _interpolated_capture_time(left_frame, right_frame, alpha)
    moon_offset_x, moon_offset_y = model.moon_center_at(captured_at)
    moon_x = cache.output_width / 2.0 + moon_offset_x * cache.scale
    moon_y = cache.output_height / 2.0 + moon_offset_y * cache.scale
    moon_mask = _disc_coverage(
        cache.grid_x,
        cache.grid_y,
        moon_x,
        moon_y,
        model.moon_radius * cache.scale,
    )
    texture = _physical_texture(left_frame, right_frame, alpha, cache)
    texture *= 1.0 - moon_mask[:, :, None]
    return np.uint8(np.clip(np.rint(texture), 0, 255))


def _physical_texture(
    left_frame: FrameAnalysis,
    right_frame: FrameAnalysis,
    alpha: float,
    cache: _AlignedFrameCache,
) -> np.ndarray:
    """Render the moving solar texture without applying the lunar silhouette."""
    assert cache.atlas is not None
    assert cache.atlas_brightness is not None
    assert cache.atlas_chroma is not None
    assert cache.atlas_detail is not None
    assert cache.solar_mask is not None

    target_brightness = left_frame.brightness * (1.0 - alpha) + right_frame.brightness * alpha
    brightness_gain = target_brightness / cache.atlas_brightness
    captured_at = _interpolated_capture_time(left_frame, right_frame, alpha)
    detail_elapsed = (captured_at - cache.detail_motion.reference_time).total_seconds()
    if (
        cache.detail_motion.velocity_x_pixels_per_second
        or cache.detail_motion.velocity_y_pixels_per_second
    ):
        transform = np.asarray(
            [
                [
                    1.0,
                    0.0,
                    cache.detail_motion.velocity_x_pixels_per_second * detail_elapsed,
                ],
                [
                    0.0,
                    1.0,
                    cache.detail_motion.velocity_y_pixels_per_second * detail_elapsed,
                ],
            ],
            dtype=np.float32,
        )
        moving_detail = cv2.warpAffine(
            cache.atlas_detail,
            transform,
            (cache.output_width, cache.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    else:
        moving_detail = cache.atlas_detail
    texture_luminance = 128.0 + moving_detail
    image = texture_luminance[:, :, None] * brightness_gain * cache.atlas_chroma[None, None, :]
    image *= cache.solar_mask[:, :, None]
    return image


def _interpolated_capture_time(
    left_frame: FrameAnalysis,
    right_frame: FrameAnalysis,
    alpha: float,
) -> datetime:
    interval_seconds = (right_frame.captured_at - left_frame.captured_at).total_seconds()
    return left_frame.captured_at + timedelta(seconds=interval_seconds * alpha)


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
            cache.output_width / 2.0,
            cache.output_height / 2.0,
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
    output_width, output_height = output_dimensions(render)
    return (
        output_width / 2.0 + (frame.moon_center_x - frame.center_x) * scale,
        output_height / 2.0 + (frame.moon_center_y - frame.center_y) * scale,
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
    *,
    detail_motion: _DetailMotion | None = None,
    source_anchor_report: dict[str, object] | None = None,
    reconstruction_excluded: tuple[FrameAnalysis, ...] = (),
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
        "excluded_blurry_from_reconstruction": [
            frame.filename for frame in reconstruction_excluded
        ],
    }
    if detail_motion is not None:
        motion_payload = asdict(detail_motion)
        motion_payload["reference_time"] = detail_motion.reference_time.isoformat()
        payload["solar_detail_motion"] = motion_payload
    if source_anchor_report is not None:
        payload["source_anchors"] = source_anchor_report
    report_file = output_file.with_suffix(".json")
    report_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
