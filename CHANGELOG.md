# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Configurable final-frame holds that extend the output without stretching the
  timestamp-aware eclipse timeline.

## [0.6.4] - 2026-08-13

### Added

- Source-anchored, clock-linear rendering that places every included photograph
  on an auditable output frame.
- Sparse ingress infill derived exclusively from each gap's starting image.
- Configurable 4:5, square, landscape, and vertical output dimensions.
- Blur scoring, selective exclusion, H.264 delivery, and lossless FFV1 export.
- Machine-readable render reports with frame assignments and SHA-256 hashes.

### Changed

- Stabilized thin-crescent crop centring with robust outer-solar-limb fitting.
- Smoothed the synthetic lunar edge with subpixel coverage.
- Fixed synthetic lunar curvature to one robust global radius while leaving all
  original-photo anchor frames untouched.

### Fidelity guarantees

- Synthetic ingress frames only darken pixels from the gap-start photograph.
- No generated infill is used at or after the configured cutoff.
- Lossless FFV1 output preserves the renderer's RGB frame sequence exactly.
