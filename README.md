# Eclipse Timelapse

A non-destructive command-line pipeline that turns hand-held eclipse photographs
into an aligned, timestamp-aware timelapse in square, portrait, or landscape
formats.

It detects the solar and lunar limbs, centres every exposure, and reports soft
frames. The default source-anchored renderer preserves every complete aligned
photograph and adds only a few audited, subtractive ingress states in long early
gaps. Optional synthetic modes can reconstruct a solar texture and use a
globally fitted physical model. H.264 delivery and lossless FFV1 archival
outputs are both supported.

The repository intentionally contains no photographs or rendered media. Your
source files, EXIF-bearing analysis data, previews, and final videos stay local
unless you deliberately publish them separately.

## Why this exists

An ordinary image sequence assigns every photograph the same duration. That
distorts an irregularly photographed event. The default clock-linear timeline
instead places every photograph at its real normalized capture time. In the
source-anchored timeline, every original remains on its exact assigned frame.
Ordinary gaps switch to the nearest source at their midpoint. Eligible ingress
gaps receive sparse boundary states at their clock-linear positions. With
source anchors disabled, an interpolated mode is 25% of the way through a
10-second gap after 2.5 seconds, 50% after 5 seconds, and 75% after 7.5 seconds.

With source anchors disabled, the physical renderer holds the Sun fixed and
moves a fitted lunar disc at constant measured velocity. A clean texture atlas
assembled only from the aligned source photographs prevents photographed lunar
edges from leaking into intermediate frames. Source brightness is retained,
while colour is normalized
before reconstruction so different in-camera processing does not create seams
within the solar crescent. Compact solar features are tracked across reliable
observations and rendered as a moving detail layer; when no sufficiently long,
consistent track exists, detail remains neutral rather than acquiring invented
motion. Optional compressed timelines remain available when a long real-world
gap would otherwise occupy more of the finished film than desired.

The project configuration enables source anchors. Each original is given
its own ordered output frame as close as possible to its ideal clock-linear
position. At that frame, the renderer emits only the aligned 4:5 crop of that
photograph—no texture atlas, synthetic lunar geometry, mask, local retouching,
or blend.

Before the configured 19-second cutoff, gaps of at least 0.75 seconds receive a
new boundary state every 0.75 seconds. Each state starts again from the gap's
first photograph and changes pixels in exactly one permitted way: computed
lunar coverage darkens newly covered solar pixels, with a narrow two-pixel
subpixel transition at the Moon's edge and fully covered pixels set to RGB
black. Synthetic states use one robust globally fitted lunar radius rather than
allowing uncertain short-arc fits to make the Moon appear to change size. Its
centre still moves linearly between the two observed endpoint positions. It
never borrows endpoint texture, blends source photographs, moves detail,
brightens a pixel, changes an unoccluded pixel, or reveals a pixel that was
previously removed within the gap. The defaults add 18 distinct states
across six gaps. At and after 19 seconds there is no generated infill; the
nearest complete photograph is held. The JSON report records every source
assignment, infill state, timing offset, blur flag, and source SHA-256 digest.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or another Python installer

FFmpeg is supplied by the `imageio-ffmpeg` dependency; no system FFmpeg install
is required.

## Quick start

Place your eclipse photographs in the repository directory, then adjust the
input filename pattern and render defaults in [`eclipse.toml`](eclipse.toml).
Images must contain EXIF `DateTimeOriginal` timestamps; using a consistent focal
length gives the alignment model the best input.

```sh
uv sync --locked --extra dev
uv run eclipse-timelapse run
```

The defaults live in [`eclipse.toml`](eclipse.toml). Source photographs are found
using their filename pattern but sorted by EXIF `DateTimeOriginal`, with the
filename used as a deterministic tie-breaker.

Four commands are available:

```sh
uv run eclipse-timelapse analyze
uv run eclipse-timelapse render
uv run eclipse-timelapse run
uv run eclipse-timelapse annotate-centres
```

`analyze` writes `work/analysis.json`, `work/analysis.csv`, and a labelled contact
sheet. `render` consumes the JSON report. `run` performs both steps.
`annotate-centres` creates a separate diagnostic video with the stabilized Sun
centre in red and the clock-linear path through measured Moon centres in blue.

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

Crop centring uses a robust fit of the visible outer solar limb. This remains
stable when only a thin crescent is visible, where a minimum circle around the
illuminated shape would otherwise jump between photographs. The corrected
centre is applied inside the existing single affine crop, so stabilization does
not add another resampling pass.

Other useful controls include:

```sh
uv run eclipse-timelapse render --duration 20 --fps 60
uv run eclipse-timelapse render --final-hold 2
uv run eclipse-timelapse render --timeline capped
uv run eclipse-timelapse render --interpolation crossfade --no-source-anchors
uv run eclipse-timelapse render --source-anchors
uv run eclipse-timelapse render --no-source-anchors
uv run eclipse-timelapse render --ingress-infill --infill-interval 0.75
uv run eclipse-timelapse render --no-ingress-infill
uv run eclipse-timelapse render --include-blurry
uv run eclipse-timelapse render --exclude-blurry
```

Supported timeline modes are `uniform`, `linear`, `capped`, and `logarithmic`.
`linear` is the default and preserves clock time exactly. Supported interpolation
modes are `physical` (default), `morph`, `geometry`, and `crossfade`.
Every timeline mode uses a linear fraction within each pair of photographs; the
mode changes only how much of the finished clip is allocated to each capture
gap.

## Auditable master and Instagram copy

The tracked defaults produce a 26.25-second, 60 FPS, 1080×1350 H.264 MP4. That
1,575-frame grid gives all 92 photographs a distinct ordered frame; the render
report records the small unavoidable grid offsets around the one pair of files
whose EXIF timestamps are identical.

Create the Instagram delivery file with:

```sh
uv run eclipse-timelapse render
```

To append a two-second hold of the exact final rendered frame without stretching
the clock-linear eclipse timeline, use `--final-hold 2`. The render report records
the timeline and hold frame counts separately.

Create a lossless archival master with the identical frame sequence using:

```sh
uv run eclipse-timelapse render \
  --codec ffv1 \
  --output output/eclipse_timelapse_source_anchored_master_4x5.mkv
```

FFV1 preserves every rendered RGB pixel exactly, including the exact source
anchors and the sparse subtractive states. H.264 and Instagram both re-encode
pixels, and Instagram may also convert 60 FPS video to 30 FPS; the lossless
master and its JSON audit report are therefore the authoritative artifacts.

Create a viewing-only centre trace from any rendered video and its matching JSON
report with:

```sh
uv run eclipse-timelapse annotate-centres \
  --video output/eclipse_timelapse_source_anchored_instagram_4x5.mp4 \
  --output output/eclipse_timelapse_centres_diagnostic_4x5.mp4
```

The input video is never modified. The red marker is the stabilized solar centre;
the blue marker and cumulative trail move linearly between the measured lunar
centres at their source-anchor times. A separate JSON file records the diagnostic
input, frame counts, colours, path policy, and SHA-256 digest.

## Blur controls

Every analysis records a normalized sharpness score in JSON and CSV and labels
flagged frames in red on the contact sheet. A frame is considered blurry when
its score is below `analysis.blur_threshold` (default `0.65`). Higher thresholds
flag more photographs:

```sh
uv run eclipse-timelapse run --blur-threshold 0.8 --exclude-blurry
uv run eclipse-timelapse render --include-blurry
uv run eclipse-timelapse render --exclude-blurry \
  --output output/eclipse_timelapse_sharp_only_instagram_4x5.mp4
```

The threshold is applied during `analyze` or `run`. The default render includes
every photograph. `--exclude-blurry` omits flagged sources from either renderer;
in the source-anchored mode, every retained photograph keeps its original
clock-linear anchor and the nearest sharp photograph is held through each
omitted slot. `--include-blurry` restores all sources. The files on disk are
never deleted or edited.

## Default workflow

1. Validate EXIF capture times and sort chronologically.
2. Isolate the largest bright component against the dark sky.
3. Robustly fit the outer solar limb, then disambiguate the occulting lunar limb.
4. Score edge sharpness and flag frames below the configured threshold.
5. Align the solar centre with a single affine resampling operation.
6. Assign every photograph a unique, minimum-error frame near its ideal
   clock-linear position.
7. Add sparse ingress-only states to qualifying pre-cutoff gaps by darkening
   only newly occulted pixels in each gap's starting photograph, using a narrow
   subpixel edge transition, a robust constant lunar radius, and pure black for
   full coverage.
8. Hold the nearest complete aligned photograph everywhere else, including all
   frames at and after the cutoff.
9. Stream RGB frames into either H.264 with BT.709 metadata or lossless FFV1.

The original files are never edited. Blur-flagged photographs remain in the
default source-anchored render unless explicitly excluded.

## Outputs

- `work/analysis.json`: complete machine-readable detection report
- `work/analysis.csv`: spreadsheet-friendly frame measurements
- `work/contact-sheet.jpg`: aligned visual review with blur flags
- `output/*.mp4`: Instagram-oriented H.264 video
- `output/*.mkv`: lossless FFV1 archival master
- `output/*-poster.jpg`: representative still
- `output/*.json`: render parameters, frame selection, and SHA-256 digest

The `work/`, `output/`, original photographs, and local virtual environment are
ignored by Git. This keeps the repository publishable without accidentally
uploading the source media.

Analysis reports contain source filenames and capture timestamps. They are
ignored for the same reason. Always inspect `git status` before publishing if
you have changed the ignore rules or force-added any artifact.

## Development

```sh
uv sync --locked --extra dev
uv run ruff check .
uv run pytest
```

The dependency lockfile is committed for reproducible development. The package
can also be installed with standard `pip` tooling from `pyproject.toml`.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development and pull-request
workflow. Notable changes are recorded in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT. See [`LICENSE`](LICENSE).
