"""Steam rising from a cup.

Every time-dependent term is periodic in the loop duration, so the last frame
flows straight back into the first.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from cinemagraph.noise import fractal_noise


@dataclass
class SteamConfig:
    source_x: int  # center of the liquid surface
    source_y: int
    height: int = 260  # how far the plume rises, in pixels
    half_width: int = 110  # half width of the working patch
    base_width: float = 9.0  # ribbon thickness at the source
    top_width: float = 26.0  # ribbon thickness at the top
    strength: float = 0.55  # peak opacity
    color: tuple[float, float, float] = (1.0, 0.95, 0.88)  # RGB, 0..1
    seed: int = 7


class Steam:
    def __init__(self, cfg: SteamConfig, period: float):
        self.cfg = cfg
        self.period = period
        h, w = cfg.height, cfg.half_width * 2
        self.x0 = cfg.source_x - cfg.half_width
        self.y0 = cfg.source_y - cfg.height
        # distance above the source, 0 at the bottom row
        self.rise = (cfg.height - np.arange(h))[:, None].astype(np.float64)
        self.xs = np.arange(w)[None, :].astype(np.float64) - cfg.half_width
        self.breakup = fractal_noise(h, w, scale=18, seed=cfg.seed)
        self.drift = fractal_noise(h, w, scale=40, seed=cfg.seed + 10, octaves=2)
        rng = np.random.default_rng(cfg.seed)
        # (horizontal offset, wavelength, phase) for each ribbon
        self.ribbons = [
            (rng.uniform(-14, 14), rng.uniform(90, 140), rng.uniform(0, 2 * np.pi))
            for _ in range(3)
        ]

    def alpha(self, t: float) -> np.ndarray:
        cfg = self.cfg
        h = cfg.height
        u = self.rise / h  # 0 at source, 1 at top
        phase = 2 * np.pi * t / self.period

        # Scroll the noise upward by exactly one tile per loop.
        shift = int(round(h * t / self.period)) % h
        breakup = np.roll(self.breakup, -shift, axis=0)
        drift = np.roll(self.drift, -shift, axis=0)

        width = cfg.base_width + (cfg.top_width - cfg.base_width) * u
        sway = 4 + 30 * u**1.3  # wider swing higher up
        density = np.zeros((h, cfg.half_width * 2))
        for offset, wavelength, phi in self.ribbons:
            center = (
                offset * (1 - 0.5 * u)
                + sway * np.sin(2 * np.pi * self.rise / wavelength - phase + phi)
                + 22 * u * (drift - 0.5)
            )
            density += np.exp(-((self.xs - center) ** 2) / (2 * width**2))

        envelope = np.clip(self.rise / 22, 0, 1) * (1 - u) ** 1.6
        a = density * (0.25 + breakup**1.5) * envelope
        a = np.clip(a * cfg.strength, 0, 1)
        return cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2.0)

    def apply(self, frame: np.ndarray, t: float) -> None:
        """Screen-blend the steam onto an RGB float frame in place."""
        a = self.alpha(t)[..., None]
        h, w = a.shape[:2]
        region = frame[self.y0 : self.y0 + h, self.x0 : self.x0 + w]
        color = np.array(self.cfg.color, dtype=np.float32)
        region[:] = 1 - (1 - region) * (1 - color * a)
