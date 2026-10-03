"""Measure how much the sky changes over the loop (clouds effect).

    PYTHONPATH=review uv run python review/sky_metrics.py output/night_study.mp4
"""

import sys

import numpy as np
from PIL import Image

from cinemagraph.effects.clouds import Clouds, CloudsConfig
from loop_metrics import SOURCE, frames


def main() -> None:
    video = frames(sys.argv[1])
    base = np.asarray(Image.open(SOURCE).convert("RGB"), np.float32) / 255
    clouds = Clouds(CloudsConfig(), base, 6.0)
    sky = clouds.weight[..., 0] > 0.5
    y0, y1, x0, x1 = clouds.y0, clouds.y1, clouds.x0, clouds.x1
    luma = video[:, y0:y1, x0:x1].mean(axis=3)
    swing = luma.max(axis=0) - luma.min(axis=0)
    half = np.abs(luma[len(luma) // 2] - luma[0])
    print(f"sky swing median={np.median(swing[sky]):.1f} p90={np.percentile(swing[sky], 90):.1f}")
    print(f"frame 0 vs mid median={np.median(half[sky]):.1f} p90={np.percentile(half[sky], 90):.1f}")


if __name__ == "__main__":
    main()
