# Eclipse Timelapse

A non-destructive command-line pipeline that turns hand-held eclipse photographs
into an aligned, timestamp-aware timelapse in square, portrait, or landscape
formats.

It detects the solar and lunar limbs, centres every exposure, reports soft
frames, reconstructs a clean solar texture, and renders the eclipse from a
globally fitted physical model into an H.264 MP4. Output resolution, crop,
duration, frame rate, quality, timing, and interpolation are configurable.

## Why this exists

An ordinary image sequence assigns every photograph the same duration. That
distorts an irregularly photographed event. The default clock-linear timeline
instead anchors every photograph at its real normalized capture time. If two
photographs are 10 seconds apart, the generated eclipse is exactly 25% of the
way between them after 2.5 seconds, 50% after 5 seconds, and 75% after 7.5
seconds.

The default physical renderer holds the Sun fixed and moves a fitted lunar disc
at constant measured velocity. A clean texture atlas assembled only from the
aligned source photographs prevents photographed lunar edges from leaking into
intermediate frames. Source brightness is retained, while colour is normalized
before reconstruction so different in-camera processing does not create seams
within the solar crescent. Compact solar features are tracked across reliable
observations and rendered as a moving detail layer; when no sufficiently long,
consistent track exists, detail remains neutral rather than acquiring invented
motion. Optional compressed timelines remain available when a long real-world
gap would otherwise occupy more of the finished film than desired.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or another Python installer

FFmpeg is supplied by the `imageio-ffmpeg` dependency; no system FFmpeg install
is required.

## Quick start

```sh
uv sync --extra dev
uv run eclipse-timelapse run
```

The defaults live in [`eclipse.toml`](eclipse.toml). Source photographs are found
using their filename pattern but sorted by EXIF `DateTimeOriginal`, with the
filename used as a deterministic tie-breaker.

Three commands are available:

```sh
uv run eclipse-timelapse analyze
uv run eclipse-timelapse render
uv run eclipse-timelapse run
```

`analyze` writes `work/analysis.json`, `work/analysis.csv`, and a labelled contact
sheet. `render` consumes the JSON report. `run` performs both steps.

## Aspect ratio and resolution

The default is a 4:5 Instagram portrait, which uses more screen area while
keeping the eclipse visually prominent:

```sh
uv run eclipse-timelapse render \
  --aspect-ratio 4:5 \
  --resolution 1080 \
  --crop-size 2000 \
  --output output/eclipse_timelapse_instagram_4x5.mp4
```

`resolution` is the output width, so that command produces 1080×1350. A 9:16
ratio is also accepted for Reels, as are arbitrary positive integer ratios.

CLI options override the tracked configuration without changing it:

```sh
uv run eclipse-timelapse render \
  --aspect-ratio 4:5 \
  --resolution 2160 \
  --crop-size 3000 \
  --output output/eclipse_timelapse_4x5_2160.mp4
```

The source crop width and output width are independent. Keeping `crop-size` at
or above `resolution` avoids upscaling. The corresponding heights are derived
from the chosen aspect ratio, and codec dimensions are kept even for broad H.264
compatibility.

Other useful controls include:

```sh
uv run eclipse-timelapse render --duration 20 --fps 60
uv run eclipse-timelapse render --timeline capped
uv run eclipse-timelapse render --interpolation crossfade
uv run eclipse-timelapse render --include-blurry
uv run eclipse-timelapse render --exclude-blurry
```

Supported timeline modes are `uniform`, `linear`, `capped`, and `logarithmic`.
`linear` is the default and preserves clock time exactly. Supported interpolation
modes are `physical` (default), `morph`, `geometry`, and `crossfade`.
Every timeline mode uses a linear fraction within each pair of photographs; the
mode changes only how much of the finished clip is allocated to each capture
gap.

## Blur controls

Every analysis records a normalized sharpness score in JSON and CSV and labels
flagged frames in red on the contact sheet. A frame is considered blurry when
its score is below `analysis.blur_threshold` (default `0.65`). Higher thresholds
flag more photographs:

```sh
uv run eclipse-timelapse run --blur-threshold 0.8 --exclude-blurry
uv run eclipse-timelapse render --include-blurry
```

The threshold is applied during `analyze` or `run`. The render policy never
deletes a photograph; it only chooses whether flagged frames participate in the
video.

## Default workflow

1. Validate EXIF capture times and sort chronologically.
2. Isolate the largest bright component against the dark sky.
3. Fit the solar limb and robustly disambiguate the occulting lunar limb.
4. Score edge sharpness and flag frames below the configured threshold.
5. Align the solar centre with a single affine resampling operation.
6. Reconstruct available solar texture from the aligned observations.
7. Fit one smooth lunar trajectory across the geometrically reliable frames.
8. Map each photograph to its exact clock-linear position and render analytic
   solar and lunar masks.
9. Stream RGB frames directly into H.264 with BT.709 colour metadata.

The original photographs are never edited. Blurry frames are reported and
excluded from the default render, but never deleted.

## Outputs

- `work/analysis.json`: complete machine-readable detection report
- `work/analysis.csv`: spreadsheet-friendly frame measurements
- `work/contact-sheet.jpg`: aligned visual review with blur flags
- `output/*.mp4`: rendered video
- `output/*-poster.jpg`: representative still
- `output/*.json`: render parameters, frame selection, and SHA-256 digest

The `work/`, `output/`, original photographs, and local virtual environment are
ignored by Git. This keeps the repository publishable without accidentally
uploading the source media.

## Development

```sh
uv sync --extra dev
uv run pytest
```

The dependency lockfile is committed for reproducible development. The package
can also be installed with standard `pip` tooling from `pyproject.toml`.

## License

MIT. See [`LICENSE`](LICENSE).
