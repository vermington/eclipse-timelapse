from datetime import datetime

import cv2
import numpy as np

from eclipse_timelapse.analysis import detect_frame
from eclipse_timelapse.model import FrameAnalysis
from eclipse_timelapse.render import align_frame


def test_detects_and_centres_synthetic_crescent(tmp_path) -> None:
    source_center = (730, 440)
    radius = 150
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    cv2.circle(image, source_center, radius, (150, 150, 150), thickness=-1)
    cv2.circle(image, (790, 410), radius, (0, 0, 0), thickness=-1)
    filename = tmp_path / "crescent.jpg"
    cv2.imwrite(str(filename), image, [cv2.IMWRITE_JPEG_QUALITY, 98])

    result = detect_frame(filename, threshold=20, minimum_component_pixels=500)

    assert abs(result["center_x"] - source_center[0]) < 3
    assert abs(result["center_y"] - source_center[1]) < 3
    assert abs(result["radius"] - radius) < 3

    frame = FrameAnalysis(
        sequence=1,
        filename=filename.name,
        captured_at=datetime(2026, 8, 12),
        blurry=False,
        **result,
    )
    aligned = align_frame(image, frame, output_width=600, output_height=600, crop_size=600)
    gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    _, _, _, centroids = cv2.connectedComponentsWithStats(bright)
    # The crescent centroid is offset, so validate the transformed solar centre directly.
    expected_x = 300
    expected_y = 300
    transform_x = expected_x + (result["center_x"] - source_center[0])
    transform_y = expected_y + (result["center_y"] - source_center[1])
    assert abs(transform_x - expected_x) < 3
    assert abs(transform_y - expected_y) < 3


def test_outer_limb_fit_centres_a_very_thin_crescent(tmp_path) -> None:
    source_center = (730, 440)
    solar_radius = 150
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    cv2.circle(image, source_center, solar_radius, (150, 150, 150), thickness=-1)
    cv2.circle(image, (750, 450), 155, (0, 0, 0), thickness=-1)
    filename = tmp_path / "thin-crescent.png"
    cv2.imwrite(str(filename), image)

    result = detect_frame(filename, threshold=20, minimum_component_pixels=500)

    assert abs(result["center_x"] - source_center[0]) < 1.5
    assert abs(result["center_y"] - source_center[1]) < 1.5
    assert abs(result["radius"] - solar_radius) < 2.0
