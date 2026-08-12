# Eclipse Timelapse

A non-destructive command-line pipeline that turns hand-held eclipse photographs
into an aligned, square, timestamp-aware timelapse.

It detects the solar and lunar limbs, centres every exposure, reports soft
frames, compresses irregular capture gaps, morphs between observed crescent
silhouettes, and produces an H.264 MP4. Output resolution, crop, duration, frame
rate, quality, timing, and interpolation are configurable.

## Why this exists

An ordinary image sequence assigns every photograph the same duration. That
distorts an irregularly photographed event. A strictly timestamp-linear video
has the opposite problem: one long capture gap can freeze the screen for several
seconds.

The default logarithmic timeline keeps the temporal character of the sequence
while compressing outlying gaps. Signed-distance morphing prevents the doubled
lunar edge that a normal crossfade creates. A texture atlas assembled only from
the aligned source photographs fills the small regions revealed between two
exposures.

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

## Rendering at a higher resolution

CLI options override the tracked configuration without changing it:

```sh
uv run eclipse-timelapse render \
  --resolution 2160 \
  --crop-size 3000 \
  --output output/eclipse_timelapse_2160.mp4
```

The source crop and output resolution are independent. Keeping `crop-size` at or
above `resolution` avoids upscaling. Both dimensions are square; the output size
must be even for broad H.264 compatibility.

Other useful controls include:

```sh
uv run eclipse-timelapse render --duration 20 --fps 60
uv run eclipse-timelapse render --timeline capped
uv run eclipse-timelapse render --interpolation crossfade
uv run eclipse-timelapse render --include-blurry
```

Supported timeline modes are `uniform`, `linear`, `capped`, and `logarithmic`.
Supported interpolation modes are `morph` (default), `geometry`, and
`crossfade`.

## Default workflow

1. Validate EXIF capture times and sort chronologically.
2. Isolate the largest bright component against the dark sky.
3. Fit the solar limb and robustly disambiguate the occulting lunar limb.
4. Score edge sharpness and flag frames below the configured threshold.
5. Align the solar centre with a single affine resampling operation.
6. Reconstruct available solar texture from the aligned observations.
7. Map capture gaps to a compressed visual timeline and morph silhouettes.
8. Stream RGB frames directly into H.264 with BT.709 colour metadata.

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
