"""The desk lamp's light breathing very gently.

Only what the lamp already lights changes: brightness is scaled by a falloff
around the bulb, weighted by how bright the pixel already is, so shadows and
the dark room stay put. The bulb's halo swings a little more than the pool of
light on the desk. Every component is periodic in the loop length.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LampConfig:
    bulb_x: int = 255
    bulb_y: int = 468
    reach_x: float = 420.0  # falloff radius across the desk
    reach_y: float = 330.0
    halo_radius: float = 55.0
    depth: float = 0.025  # brightness swing of the lit desk, as a fraction
    halo_depth: float = 0.05  # brightness swing of the bulb's halo


class Lamp:
    def __init__(self, cfg: LampConfig, base: np.ndarray, period: float):
        self.cfg = cfg
        self.period = period
        h, w = base.shape[:2]
        # Work only on the area the lamp can reach.
        self.x0 = max(int(cfg.bulb_x - 2 * cfg.reach_x), 0)
        self.x1 = min(int(cfg.bulb_x + 2 * cfg.reach_x), w)
        self.y0 = max(int(cfg.bulb_y - 2 * cfg.reach_y), 0)
        self.y1 = min(int(cfg.bulb_y + 2 * cfg.reach_y), h)
        ys, xs = np.mgrid[self.y0 : self.y1, self.x0 : self.x1].astype(np.float32)
        dx, dy = xs - cfg.bulb_x, ys - cfg.bulb_y
        falloff = np.exp(-((dx / cfg.reach_x) ** 2 + (dy / cfg.reach_y) ** 2))
        lum = base[self.y0 : self.y1, self.x0 : self.x1] @ np.array(
            [0.299, 0.587, 0.114], dtype=np.float32
        )
        lit = np.clip((lum - 0.25) / 0.45, 0, 1)
        halo = np.exp(-(dx**2 + dy**2) / (2 * cfg.halo_radius**2))
        # Scale only what is already lit, so the dark lamp shade beside the
        # bulb does not glow.
        gain = (cfg.depth * falloff + cfg.halo_depth * halo) * lit
        self.gain = gain[..., None].astype(np.float32)

    def level(self, t: float) -> float:
        """Brightness offset in [-1, 1]: one even bright-dark-bright breath.

        Faster terms read as a flame or a failing bulb, and an uneven mix of
        slow terms reads as the lamp dimming and struggling to recover.
        """
        return float(np.cos(2 * np.pi * t / self.period))

    def apply(self, frame: np.ndarray, t: float) -> None:
        m = self.level(t)
        region = frame[self.y0 : self.y1, self.x0 : self.x1]
        region *= 1 + m * self.gain
