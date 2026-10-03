"""Check that motion advances evenly from frame to frame.

    PYTHONPATH=review uv run python review/smoothness.py output/night_study.mp4

Prints the coefficient of variation of the frame-to-frame change in the steam
and sky regions. Uneven steps (a high value) read as stutter.
"""

import sys

import numpy as np

from loop_metrics import frames

REGIONS = {"steam": (830, 560, 955, 770), "sky": (680, 10, 1010, 90)}


def main() -> None:
    for path in sys.argv[1:]:
        video = frames(path)
        out = [path.rsplit("/", 1)[-1], f"frames={len(video)}"]
        for name, (x0, y0, x1, y1) in REGIONS.items():
            crop = video[:, y0:y1, x0:x1].mean(axis=3)
            steps = np.abs(np.diff(crop, axis=0)).mean(axis=(1, 2))
            out.append(f"{name} cv={steps.std() / steps.mean():.2f}")
        print("  ".join(out))


if __name__ == "__main__":
    main()
