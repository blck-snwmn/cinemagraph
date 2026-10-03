# cinemagraph

Procedural cinemagraph experiments: a still image stays as is and only masked
regions are animated with seamless, periodic effects.

## Usage

```sh
uv sync
uv run python -m cinemagraph.render  # writes output/night_study.mp4 and .webp
```

Options: `--period` (loop length in seconds, default 6), `--fps` (default 24).

## Effects

- `steam`: steam rising from the coffee cup
