# Eclipse Timelapse

A programmatic pipeline for turning a hand-held sequence of eclipse photographs
into an aligned, square-cropped timelapse.

The source set contains 92 photographs captured at 300 mm over approximately
37 minutes. Their EXIF timestamps are unevenly spaced, so the intended pipeline
will preserve the overall chronology while smoothing long gaps rather than
holding a single frame for an excessive time.

## Planned pipeline

1. Read and validate capture times from EXIF metadata.
2. Measure sharpness and produce a review report for blurry frames.
3. Detect the solar disc and align its centre across the sequence.
4. Create consistent 1:1 crops without modifying the originals.
5. Place frames according to capture time and smooth long gaps.
6. Encode preview and final MP4 videos.

## Repository layout

The original photographs and generated media are intentionally excluded from
Git. Processing code and configuration will live in this repository; generated
files will be written beneath `work/` and `output/`.

## Status

Project scaffold created. The processing pipeline is the next step.
