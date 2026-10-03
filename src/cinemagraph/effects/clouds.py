"""Clouds shifting slowly in the night sky outside the window.

The sky is split into a soft layer (clouds) and a detail layer (stars, paint
grain). Only the soft layer moves, so the stars stay put. Over a loop of a few
seconds real clouds barely travel, so instead of translating them (which can
only loop by cross-fading, and that reads as blurring in place) the soft layer
is warped by a slow wave that travels to the right, and broad patches of light
and shade scroll right across it. Bands of cloud swell and brighten as they
pass. Both are periodic in the loop length.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from cinemagraph.noise import periodic_noise


@dataclass
class CloudsConfig:
    # Glass panes as (x0, x1, bottom_y); sky is searched from the top down.
    panes: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(664, 808, 178), (860, 1024, 195)]
    )
    sway_x: float = 16.0  # horizontal warp amplitude, in pixels
    sway_y: float = 7.0  # vertical warp amplitude, in pixels
    # Patches of light and shade that drift right by one tile per loop.
    shade_tile: int = 120  # pixels, so they travel tile / loop length
    shade_depth: float = 0.3  # brightness swing, as a fraction
    wavelength: float = 260.0  # pixels; the wave crosses one of these per loop
    edge_fade: float = 22.0  # motion fades out this close to buildings/frame


class Clouds:
    def __init__(self, cfg: CloudsConfig, base: np.ndarray, period: float):
        self.cfg = cfg
        self.period = period
        self.x0 = min(p[0] for p in cfg.panes) - 4
        self.x1 = max(p[1] for p in cfg.panes) + 4
        self.y0, self.y1 = 0, max(p[2] for p in cfg.panes) + 4
        patch = base[self.y0 : self.y1, self.x0 : self.x1]

        sky = self._sky_mask(patch)
        # Fade motion out near anything that is not sky, so the shifted sky
        # never drags building or frame pixels along with it.
        dist = cv2.distanceTransform(sky.astype(np.uint8), cv2.DIST_L2, 5)
        weight = np.clip(dist / cfg.edge_fade, 0, 1)
        self.weight = (weight * weight * (3 - 2 * weight))[..., None].astype(np.float32)

        # Median first so stars and specks stay in the static detail layer.
        smooth = cv2.medianBlur((np.clip(patch, 0, 1) * 255).astype(np.uint8), 9)
        self.soft = cv2.GaussianBlur(smooth.astype(np.float32) / 255, (0, 0), 2.0)
        self.detail = patch - self.soft
        h, w = patch.shape[:2]
        self.gy, self.gx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Uneven phase so the wave front is not a straight vertical line.
        self.jitter = (2.0 * periodic_noise(h, w, scale=40, seed=5)).astype(np.float32)
        # Broad, flat patches; periodic across the tile so the scroll wraps.
        tile = periodic_noise(h, cfg.shade_tile, scale=30, seed=11, stretch=0.5)
        self.shade = (tile - tile.mean()).astype(np.float32)

    def _sky_mask(self, patch: np.ndarray) -> np.ndarray:
        rgb = patch * 255
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # The sky is a dusky violet; buildings are darker and greyer.
        skylike = ((rgb[..., 2] - rgb[..., 1]) > 13) & (lum > 46)
        skylike = cv2.morphologyEx(skylike.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = np.zeros(skylike.shape, np.uint8)
        for px0, px1, bottom in self.cfg.panes:
            for x in range(px0 - self.x0, px1 - self.x0):
                gap = 0
                end = bottom
                for y in range(bottom):
                    gap = 0 if skylike[y, x] else gap + 1
                    if gap >= 5:  # first solid run of non-sky is the skyline
                        end = y - 4
                        break
                mask[:end, x] = 1
        return cv2.medianBlur(mask * 255, 5) > 0

    def apply(self, frame: np.ndarray, t: float) -> None:
        phase = float(2 * np.pi * t / self.period)
        travel = phase - 2 * np.pi * self.gx / self.cfg.wavelength + self.jitter
        map_x = self.gx + self.cfg.sway_x * np.sin(travel)
        map_y = self.gy + self.cfg.sway_y * np.sin(travel + 1.3 + 0.8 * self.jitter)
        moved = cv2.remap(
            self.soft, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        tile = self.cfg.shade_tile
        offset = int(round(tile * t / self.period))
        cols = (np.arange(moved.shape[1]) - offset) % tile
        moved = moved * (1 + self.cfg.shade_depth * self.shade[:, cols, None])
        region = frame[self.y0 : self.y1, self.x0 : self.x1]
        region[:] = region + self.weight * (moved + self.detail - region)
