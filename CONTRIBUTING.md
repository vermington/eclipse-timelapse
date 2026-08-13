# Contributing

Thanks for helping improve Eclipse Timelapse.

## Development setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended:

```sh
uv sync --locked --extra dev
uv run ruff check .
uv run pytest
```

Keep changes focused, add regression coverage for behavioural changes, and run
both checks before opening a pull request. Commit messages should describe the
observable outcome in the imperative mood.

## Photograph and metadata privacy

Do not commit source photographs, rendered media, analysis reports, or contact
sheets. Apart from being large, photographs and generated reports can contain
identifying EXIF timestamps, filenames, or location metadata. The repository's
ignore rules cover the usual formats and generated directories; verify
`git status` before every commit.

Tests should create small synthetic images inside pytest temporary directories.
Only add real photographic material when its owner has explicitly licensed it
for redistribution and the repository maintainers have agreed to include it.

## Pull requests

A useful pull request includes:

- a concise explanation of the user-visible change;
- tests for new behaviour or a reproducible description of the defect;
- documentation updates when commands, configuration, or fidelity guarantees
  change; and
- confirmation that unrelated generated media is not included.
