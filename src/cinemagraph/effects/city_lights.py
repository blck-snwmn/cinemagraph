"""Lit windows of the city seen through the room's window.

Each lit window is found automatically and given its own behavior: a gentle
shimmer, a TV flickering in one room, and a blinking rooftop beacon. Every
curve is periodic in the loop duration.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CityLightsConfig:
    # Glass panes as (x0, y0, x1, y1); lights outside them are ignored.
    panes: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: [(660, 0, 810, 178), (857, 0, 1027, 195)]
    )
    beacon_near: tuple[int, int] = (992, 106)  # red light on a rooftop
    tv_near: tuple[int, int] = (787, 171)  # the room with a TV on
    shimmer_count: int = 6
    seed: int = 3


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class CityLights:
    def __init__(self, cfg: CityLightsConfig, base: np.ndarray, period: float):
        self.cfg = cfg
        self.period = period
        xs = [p[0] for p in cfg.panes] + [p[2] for p in cfg.panes]
        ys = [p[1] for p in cfg.panes] + [p[3] for p in cfg.panes]
        self.x0, self.y0 = min(xs) - 8, max(min(ys) - 8, 0)
        self.x1, self.y1 = max(xs) + 8, max(ys) + 8
        patch = base[self.y0 : self.y1, self.x0 : self.x1]

        rgb = patch * 255
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        local = cv2.blur(lum, (25, 25))
        inside = np.zeros(lum.shape, bool)
        for px0, py0, px1, py1 in cfg.panes:
            inside[py0 - self.y0 : py1 - self.y0, px0 - self.x0 : px1 - self.x0] = True
        lit = (lum - local > 12) & (rgb[..., 0] > rgb[..., 2] + 15) & (lum > 70) & inside
        n, labels, stats, centers = cv2.connectedComponentsWithStats(lit.astype(np.uint8), 8)

        # What the scene looks like with every light switched off.
        hole = cv2.dilate(lit.astype(np.uint8), np.ones((5, 5), np.uint8))
        unlit = cv2.inpaint((np.clip(patch, 0, 1) * 255).astype(np.uint8), hole, 4, cv2.INPAINT_TELEA)
        self.unlit = unlit.astype(np.float32) / 255

        centers = centers + [self.x0, self.y0]
        ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]

        def nearest(point: tuple[int, int]) -> int:
            return min(ids, key=lambda i: np.hypot(*(centers[i] - point)))

        self.beacon = nearest(cfg.beacon_near)
        self.tv = nearest(cfg.tv_near)
        rng = np.random.default_rng(cfg.seed)
        others = [i for i in ids if i not in (self.beacon, self.tv)]
        chosen = rng.choice(others, size=min(cfg.shimmer_count, len(others)), replace=False)
        self.shimmer = [
            (
                int(i),
                rng.uniform(0.25, 0.45),  # depth of the dip
                int(rng.integers(1, 3)),  # slow cycles per loop
                int(rng.integers(3, 6)),  # faster cycles per loop
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
            )
            for i in chosen
        ]

        def soft(i: int, grow: int) -> np.ndarray:
            m = cv2.dilate((labels == i).astype(np.uint8), np.ones((grow, grow), np.uint8))
            return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]

        self.masks = {i: soft(i, 5) for i in ids}
        self.beacon_glow = cv2.GaussianBlur(
            (labels == self.beacon).astype(np.float32), (0, 0), 1.5
        )[..., None]
        self.beacon_glow /= self.beacon_glow.max()

        # TV: brightness and tint change in steps, like scene cuts.
        self.tv_steps = 12
        self.tv_dim = rng.uniform(0.0, 0.45, self.tv_steps)
        self.tv_blue = rng.uniform(0.2, 0.7, self.tv_steps)

    def _tv(self, t: float) -> tuple[float, float]:
        pos = (t / self.period % 1) * self.tv_steps
        i = int(pos) % self.tv_steps
        j = (i + 1) % self.tv_steps
        # Hold each shot, then cut over quickly.
        mix = _smoothstep((pos - int(pos) - 0.85) / 0.15)
        dim = self.tv_dim[i] * (1 - mix) + self.tv_dim[j] * mix
        blue = self.tv_blue[i] * (1 - mix) + self.tv_blue[j] * mix
        dim += 0.04 * np.sin(2 * np.pi * 37 * t / self.period)
        return float(np.clip(dim, 0, 1)), float(blue)

    def apply(self, frame: np.ndarray, t: float) -> None:
        region = frame[self.y0 : self.y1, self.x0 : self.x1]
        phase = 2 * np.pi * t / self.period
        off = np.zeros(region.shape[:2] + (1,), np.float32)

        for i, depth, slow, fast, p1, p2 in self.shimmer:
            k = depth * (0.5 - 0.5 * np.cos(slow * phase + p1)) * (0.7 + 0.3 * np.sin(fast * phase + p2))
            off = np.maximum(off, self.masks[i] * k)

        # Beacon: three slow blinks per loop.
        cycle = (3 * t / self.period) % 1
        on = _smoothstep(cycle / 0.12) * (1 - _smoothstep((cycle - 0.4) / 0.15))
        off = np.maximum(off, self.masks[self.beacon] * 0.9 * (1 - on))

        dim, blue = self._tv(t)
        tv_mask = self.masks[self.tv]
        off = np.maximum(off, tv_mask * dim)

        region[:] = region * (1 - off) + self.unlit * off
        # Cool cast from the TV screen, and a soft red halo when the beacon is lit.
        tv_tint = np.array([-0.06, -0.01, 0.07], np.float32) * blue
        region[:] += tv_mask * tv_tint * (1 - dim)
        region[:] += self.beacon_glow * np.array([0.14, 0.02, 0.04], np.float32) * on
