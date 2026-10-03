"""Render the night study cinemagraph to a looping video.

    python -m cinemagraph.render
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from cinemagraph.effects.city_lights import CityLights, CityLightsConfig
from cinemagraph.effects.clouds import Clouds, CloudsConfig
from cinemagraph.effects.lamp import Lamp, LampConfig
from cinemagraph.effects.steam import Steam, SteamConfig

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/night_study.png"


def build_effects(base: np.ndarray, period: float) -> list:
    return [
        Lamp(LampConfig(), base, period),
        Clouds(CloudsConfig(), base, period),
        CityLights(CityLightsConfig(), base, period),
        Steam(SteamConfig(source_x=890, source_y=792), period),
    ]


def render(out: Path, period: float, fps: int, preview: Path | None) -> None:
    base = np.asarray(Image.open(SOURCE).convert("RGB"), dtype=np.float32) / 255
    height, width = base.shape[:2]
    effects = build_effects(base, period)
    n_frames = int(round(period * fps))

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        # Near-lossless constant QP: at normal quality the first (I) frame
        # carries different grain from the last (P) frame, which shows as a
        # flicker at the loop seam. Static pixels keep the file small anyway.
        # No B-frames either: they get a coarser quantizer than P-frames, so
        # the grain pulses every few frames and motion looks uneven.
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-qp", "4", "-bf", "0",
        "-movflags", "+faststart",
        str(out),
    ]  # fmt: skip
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for i in range(n_frames):
            frame = base.copy()
            for effect in effects:
                effect.apply(frame, i / fps)
            proc.stdin.write((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        proc.stdin.close()
    if proc.returncode:
        raise SystemExit(f"ffmpeg failed with code {proc.returncode}")

    if preview:
        # A small looping WebP that is easy to view anywhere.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                "-vf", "scale=724:-2", "-loop", "0",
                # Below ~95 the animated WebP encoder treats the faint sky
                # changes as "unchanged" and skips them, so the clouds stall
                # for several frames and then jump.
                "-quality", "95",
                str(preview),
            ],  # fmt: skip
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/night_study.mp4")
    parser.add_argument("--preview", type=Path, default=ROOT / "output/night_study.webp")
    parser.add_argument("--period", type=float, default=6.0, help="loop length in seconds")
    # 30 fps divides evenly into 60 Hz displays (2 refreshes per frame); 24 fps
    # alternates 2 and 3 refreshes, which shows as judder in GIF/WebP viewers.
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    render(args.out, args.period, args.fps, args.preview)


if __name__ == "__main__":
    main()
