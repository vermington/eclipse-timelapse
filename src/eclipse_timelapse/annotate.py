"""Diagnostic overlays for rendered eclipse videos."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from eclipse_timelapse.model import AnalysisReport
from eclipse_timelapse.render import _exclusive_output_lock

ProgressCallback = Callable[[str], None]

_BLACK = (0, 0, 0)
_BLUE = (255, 0, 0)
_RED = (0, 0, 255)


class AnnotationError(RuntimeError):
    """Raised when a diagnostic video cannot be annotated."""


def annotate_centres(
    analysis: AnalysisReport,
    *,
    input_video: Path,
    render_report_file: Path,
    output_file: Path,
    progress: ProgressCallback | None = None,
) -> Path:
    """Overlay the solar centre and measured lunar-centre trajectory."""
    input_video = input_video.resolve()
    render_report_file = render_report_file.resolve()
    output_file = output_file.resolve()
    if input_video == output_file:
        raise AnnotationError("Annotation output must differ from the input video")
    if output_file.with_suffix(".json") == render_report_file:
        raise AnnotationError("Annotation report must not overwrite the source render report")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output_file):
        return _annotate_centres_unlocked(
            analysis,
            input_video=input_video,
            render_report_file=render_report_file,
            output_file=output_file,
            progress=progress,
        )


def _annotate_centres_unlocked(
    analysis: AnalysisReport,
    *,
    input_video: Path,
    render_report_file: Path,
    output_file: Path,
    progress: ProgressCallback | None,
) -> Path:
    payload = _read_render_report(render_report_file)
    parameters = payload["parameters"]
    source_anchors = payload.get("source_anchors")
    if not isinstance(source_anchors, dict) or not source_anchors.get("frames"):
        raise AnnotationError("Centre annotation requires a source-anchored render report")

    try:
        width = int(parameters["resolution"])
        ratio_width, ratio_height = map(int, parameters["aspect_ratio"].split(":"))
        height = round(width * ratio_height / ratio_width)
        if height % 2:
            height += 1
        crop_size = float(parameters["crop_size"])
        frames_per_second = int(parameters["frames_per_second"])
        encoded_frames = int(payload["encoded_frames"])
        timeline_frames = int(payload.get("timeline_frames", encoded_frames))
    except (KeyError, TypeError, ValueError) as error:
        raise AnnotationError(f"Invalid render parameters in {render_report_file}") from error
    if not 1 <= timeline_frames <= encoded_frames:
        raise AnnotationError("Render report has invalid timeline and encoded frame counts")

    path = _interpolated_moon_path(
        analysis,
        source_anchors["frames"],
        timeline_frames=timeline_frames,
        output_width=width,
        output_height=height,
        crop_size=crop_size,
    )
    sun = (round(width / 2), round(height / 2))
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise AnnotationError(f"Cannot open input video: {input_video}")
    _validate_input_video(
        capture,
        input_video,
        width=width,
        height=height,
        frames_per_second=frames_per_second,
        encoded_frames=encoded_frames,
    )

    temporary_file = output_file.with_name(f".{output_file.stem}.partial{output_file.suffix}")
    command = _encoder_command(
        temporary_file,
        width=width,
        height=height,
        frames_per_second=frames_per_second,
        crf=int(parameters.get("crf", 16)),
        preset=str(parameters.get("preset", "slow")),
    )
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert encoder.stdin is not None
        for output_index in range(encoded_frames):
            available, image = capture.read()
            if not available:
                raise AnnotationError(
                    f"Input video ended at frame {output_index}; expected {encoded_frames}"
                )
            timeline_index = min(output_index, timeline_frames - 1)
            _draw_centres(image, sun=sun, moon_path=path, timeline_index=timeline_index)
            encoder.stdin.write(image.tobytes())
            if progress and (
                output_index % frames_per_second == 0 or output_index + 1 == encoded_frames
            ):
                progress(f"Annotating frame {output_index + 1:04d}/{encoded_frames:04d}")
        if capture.read()[0]:
            raise AnnotationError(
                f"Input video contains more than the reported {encoded_frames} frames"
            )
        encoder.stdin.close()
        return_code = encoder.wait()
        error_output = (
            encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        )
    except (BrokenPipeError, OSError) as error:
        encoder.kill()
        encoder.wait()
        error_output = (
            encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        )
        temporary_file.unlink(missing_ok=True)
        raise AnnotationError(f"FFmpeg failed while annotating: {error_output or error}") from error
    except BaseException:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        temporary_file.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
    if return_code != 0:
        temporary_file.unlink(missing_ok=True)
        raise AnnotationError(f"FFmpeg exited with status {return_code}: {error_output}")

    temporary_file.replace(output_file)
    _write_annotation_report(
        output_file,
        input_video=input_video,
        render_report_file=render_report_file,
        encoded_frames=encoded_frames,
        timeline_frames=timeline_frames,
        sun=sun,
    )
    return output_file


def _read_render_report(filename: Path) -> dict[str, object]:
    try:
        payload = json.loads(filename.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AnnotationError(f"Render report not found: {filename}") from error
    except json.JSONDecodeError as error:
        raise AnnotationError(f"Invalid render report: {filename}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
        raise AnnotationError(f"Invalid render report: {filename}")
    return payload


def _interpolated_moon_path(
    analysis: AnalysisReport,
    anchor_records: list[object],
    *,
    timeline_frames: int,
    output_width: int,
    output_height: int,
    crop_size: float,
) -> np.ndarray:
    frames_by_name = {frame.filename: frame for frame in analysis.frames}
    anchor_indices: list[int] = []
    moon_x: list[float] = []
    moon_y: list[float] = []
    scale = output_width / crop_size
    sun_x = output_width / 2.0
    sun_y = output_height / 2.0
    for raw_anchor in anchor_records:
        if not isinstance(raw_anchor, dict):
            raise AnnotationError("Render report contains an invalid source anchor")
        try:
            filename = str(raw_anchor["filename"])
            output_frame = int(raw_anchor["output_frame"])
            frame = frames_by_name[filename]
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError("Render report contains an unknown source anchor") from error
        anchor_indices.append(output_frame)
        moon_x.append(sun_x + (frame.moon_center_x - frame.center_x) * scale)
        moon_y.append(sun_y + (frame.moon_center_y - frame.center_y) * scale)
    if anchor_indices != sorted(set(anchor_indices)):
        raise AnnotationError("Source-anchor output frames must be unique and ordered")
    if anchor_indices[0] != 0 or anchor_indices[-1] != timeline_frames - 1:
        raise AnnotationError("Source anchors do not span the complete render timeline")
    timeline_indices = np.arange(timeline_frames)
    return np.column_stack(
        (
            np.interp(timeline_indices, anchor_indices, moon_x),
            np.interp(timeline_indices, anchor_indices, moon_y),
        )
    ).round().astype(np.int32)


def _validate_input_video(
    capture: cv2.VideoCapture,
    filename: Path,
    *,
    width: int,
    height: int,
    frames_per_second: int,
    encoded_frames: int,
) -> None:
    actual_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    actual_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (actual_width, actual_height) != (width, height):
        raise AnnotationError(
            f"Input video is {actual_width}x{actual_height}; report expects {width}x{height}"
        )
    if abs(actual_fps - frames_per_second) > 0.01:
        raise AnnotationError(
            f"Input video is {actual_fps:g} FPS; report expects {frames_per_second} FPS"
        )
    if actual_frames != encoded_frames:
        raise AnnotationError(
            f"Input video has {actual_frames} frames; report expects {encoded_frames}: {filename}"
        )


def _draw_centres(
    image: np.ndarray,
    *,
    sun: tuple[int, int],
    moon_path: np.ndarray,
    timeline_index: int,
) -> None:
    trail = moon_path[: timeline_index + 1].reshape((-1, 1, 2))
    if len(trail) > 1:
        cv2.polylines(image, [trail], False, _BLACK, 7, cv2.LINE_AA)
        cv2.polylines(image, [trail], False, _BLUE, 3, cv2.LINE_AA)
    moon = tuple(int(value) for value in moon_path[timeline_index])
    cv2.circle(image, moon, 12, _BLACK, -1, cv2.LINE_AA)
    cv2.circle(image, moon, 8, _BLUE, -1, cv2.LINE_AA)
    cv2.circle(image, sun, 15, _BLACK, 7, cv2.LINE_AA)
    cv2.circle(image, sun, 15, _RED, 3, cv2.LINE_AA)
    cv2.drawMarker(image, sun, _BLACK, cv2.MARKER_CROSS, 42, 7, cv2.LINE_AA)
    cv2.drawMarker(image, sun, _RED, cv2.MARKER_CROSS, 42, 3, cv2.LINE_AA)
    _draw_label(image, "SUN CENTRE", (sun[0] + 24, sun[1] - 22), _RED)
    _draw_label(image, "MOON CENTRE + PATH", (moon[0] + 18, moon[1] + 30), _BLUE)


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, 0.62, _BLACK, 5, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, 0.62, colour, 2, cv2.LINE_AA)


def _encoder_command(
    output_file: Path,
    *,
    width: int,
    height: int,
    frames_per_second: int,
    crf: int,
    preset: str,
) -> list[str]:
    return [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(frames_per_second),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
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
        str(output_file),
    ]


def _write_annotation_report(
    output_file: Path,
    *,
    input_video: Path,
    render_report_file: Path,
    encoded_frames: int,
    timeline_frames: int,
    sun: tuple[int, int],
) -> None:
    digest = hashlib.sha256()
    with output_file.open("rb") as video_file:
        for block in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(block)
    payload = {
        "output": output_file.name,
        "sha256": digest.hexdigest(),
        "input_video": input_video.name,
        "render_report": render_report_file.name,
        "encoded_frames": encoded_frames,
        "timeline_frames": timeline_frames,
        "annotations": {
            "sun_centre_output_pixels": list(sun),
            "sun_centre_colour": "red",
            "moon_centre_colour": "blue",
            "moon_path": "linear between measured source-anchor centres",
        },
    }
    report_file = output_file.with_suffix(".json")
    report_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
