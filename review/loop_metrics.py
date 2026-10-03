"""Measure steam visibility and loop seam smoothness of a rendered video.

    uv run python review/loop_metrics.py output/night_study.mp4
"""

import subprocess
import sys

import numpy as np
from PIL import Image

SOURCE = "assets/source/night_study.png"
BOX = (830, 560, 955, 770)  # x0, y0, x1, y1 around the steam


def frames(path: str) -> np.ndarray:
    w, h = Image.open(SOURCE).size
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def main() -> None:
    video = frames(sys.argv[1])
    base = np.asarray(Image.open(SOURCE).convert("RGB"), np.float32)
    x0, y0, x1, y1 = BOX
    crop = video[:, y0:y1, x0:x1].mean(axis=3)
    lift = crop - base[y0:y1, x0:x1].mean(axis=2)
    per_frame = np.clip(lift, 0, None).mean(axis=(1, 2))
    peaks = np.percentile(lift.reshape(len(lift), -1), 99, axis=1)
    steps = np.abs(np.diff(crop, axis=0)).mean(axis=(1, 2))
    seam = np.abs(crop[0] - crop[-1]).mean()
    print(f"frames={len(video)}")
    print(f"mean lift={per_frame.mean():.2f}  min frame lift={per_frame.min():.2f} "
          f"(frame {per_frame.argmin()})  min/mean={per_frame.min() / per_frame.mean():.2f}")
    print(f"p99 lift mean={peaks.mean():.1f}  min={peaks.min():.1f}")
    print(f"frame step max={steps.max():.3f}  mean={steps.mean():.3f}  seam={seam:.3f}")


if __name__ == "__main__":
    main()
