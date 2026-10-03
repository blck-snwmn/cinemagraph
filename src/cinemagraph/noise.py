"""Tileable noise helpers used to build seamless loops."""

import numpy as np


def periodic_noise(height: int, width: int, scale: float, seed: int) -> np.ndarray:
    """Smooth noise that wraps on both axes, normalized to [0, 1].

    White noise is low-pass filtered in the frequency domain, which makes the
    result periodic by construction. `scale` is the feature size in pixels.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    sigma = 1.0 / max(scale, 1.0)
    lowpass = np.exp(-(fx**2 + fy**2) / (2 * sigma**2))
    smooth = np.real(np.fft.ifft2(np.fft.fft2(white) * lowpass))
    smooth -= smooth.min()
    return smooth / smooth.max()


def fractal_noise(
    height: int, width: int, scale: float, seed: int, octaves: int = 3
) -> np.ndarray:
    """Sum of periodic noise octaves, normalized to [0, 1]."""
    total = np.zeros((height, width))
    amp = 1.0
    for i in range(octaves):
        total += amp * periodic_noise(height, width, scale / 2**i, seed + i)
        amp *= 0.5
    total -= total.min()
    return total / total.max()
