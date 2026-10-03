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
    surface_half_width: int = 38  # wisps start anywhere across the surface
    height: int = 270  # how far the plume rises, in pixels
    half_width: int = 120  # half width of the working patch
    fade_start: float = 12.0  # invisible right at the liquid surface...
    fade_length: float = 80.0  # ...and fully formed this far above it
    base_width: float = 4.0  # wisp thickness near the surface
    top_width: float = 16.0  # wisp thickness at the top
    wisps: int = 4
    lean: float = 10.0  # overall horizontal drift at the top, in pixels
    strength: float = 3.0  # peak opacity
    # Slightly cool white so the steam separates from the warm room light.
    color: tuple[float, float, float] = (0.94, 0.97, 1.0)  # RGB, 0..1
    seed: int = 7


@dataclass
class Wisp:
    offset: float  # where on the surface it starts, relative to center
    wavelength: float
    phase: float
    reach: float  # fraction of the full height it rises before vanishing
    pulse: int  # how many times it swells and fades per loop
    pulse_phase: float


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
        self.breakup = fractal_noise(h, w, scale=22, seed=cfg.seed, stretch=2.5, octaves=2)
        self.drift = fractal_noise(h, w, scale=40, seed=cfg.seed + 10, octaves=2)
        rng = np.random.default_rng(cfg.seed)
        # Spread the start points evenly across the surface, with jitter, so
        # the steam does not collapse into a single column in the middle.
        slots = np.linspace(-1, 1, cfg.wisps)
        rng.shuffle(slots)
        self.wisps = [
            Wisp(
                offset=cfg.surface_half_width
                * np.clip(slot + rng.uniform(-0.15, 0.15), -1, 1),
                wavelength=rng.uniform(80, 150),
                phase=rng.uniform(0, 2 * np.pi),
                reach=rng.uniform(0.45, 1.0),
                pulse=int(rng.integers(1, 3)),
                # Stagger the pulses so the wisps never all thin out at once.
                pulse_phase=2 * np.pi * i / cfg.wisps + rng.uniform(-0.3, 0.3),
            )
            for i, slot in enumerate(slots)
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
        sway = 3 + 26 * u**1.3  # wider swing higher up
        # Steam is invisible at the surface and condenses as it rises.
        formed = np.clip((self.rise - cfg.fade_start) / cfg.fade_length, 0, 1)
        formed = formed * formed * (3 - 2 * formed)

        density = np.zeros((h, cfg.half_width * 2))
        for wisp in self.wisps:
            center = (
                wisp.offset * (1 - 0.35 * u)
                + cfg.lean * u
                + sway * np.sin(2 * np.pi * self.rise / wisp.wavelength - phase + wisp.phase)
                + 22 * u * (drift - 0.5)
            )
            fade = np.clip(1 - u / wisp.reach, 0, 1) ** 1.4
            # Integer pulse count keeps this periodic in the loop length.
            swell = 0.8 + 0.2 * np.sin(wisp.pulse * phase + wisp.pulse_phase)
            density += swell * fade * np.exp(-((self.xs - center) ** 2) / (2 * width**2))

        # Densest just above the rim, thinning out as it rises.
        body = 1.3 - 0.8 * np.clip(u / 0.6, 0, 1)
        a = density * (0.15 + breakup**2.2) * formed * body
        a = np.clip(a * cfg.strength, 0, 1)
        return cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2.0)

    def apply(self, frame: np.ndarray, t: float) -> None:
        """Blend the steam onto an RGB float frame in place.

        Mixes screen (glow on dark areas) with plain alpha-over (haze that
        stays visible on the light sweater behind the cup).
        """
        a = self.alpha(t)[..., None]
        h, w = a.shape[:2]
        region = frame[self.y0 : self.y0 + h, self.x0 : self.x0 + w]
        color = np.array(self.cfg.color, dtype=np.float32)
        screen = 1 - (1 - region) * (1 - color * a)
        region[:] = 0.5 * screen + 0.5 * (region * (1 - a) + color * a)
