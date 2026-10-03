"""Tileable noise helpers used to build seamless loops."""

import numpy as np


def periodic_noise(
    height: int, width: int, scale: float, seed: int, stretch: float = 1.0
) -> np.ndarray:
    """Smooth noise that wraps on both axes, normalized to [0, 1].

    White noise is low-pass filtered in the frequency domain, which makes the
    result periodic by construction. `scale` is the feature size in pixels;
    `stretch` > 1 elongates features vertically.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    sigma = 1.0 / max(scale, 1.0)
    lowpass = np.exp(-(fx**2 + (fy * stretch) ** 2) / (2 * sigma**2))
    smooth = np.real(np.fft.ifft2(np.fft.fft2(white) * lowpass))
    smooth -= smooth.min()
    return smooth / smooth.max()


def fractal_noise(
    height: int,
    width: int,
    scale: float,
    seed: int,
    octaves: int = 3,
    stretch: float = 1.0,
) -> np.ndarray:
    """Sum of periodic noise octaves, normalized to [0, 1]."""
    total = np.zeros((height, width))
    amp = 1.0
    for i in range(octaves):
        total += amp * periodic_noise(height, width, scale / 2**i, seed + i, stretch)
        amp *= 0.5
    total -= total.min()
    return total / total.max()



def roll_smooth(a: np.ndarray, shift: float, axis: int) -> np.ndarray:
    """np.roll with a fractional shift, for arrays that wrap along `axis`.

    Rounding the shift to whole pixels makes slow scrolls advance in uneven
    0/1/2 pixel steps, which reads as stutter. Blending neighbouring pixels
    instead softens the image every half pixel, which pulses. Shifting the
    phase in the frequency domain is exact for periodic noise.
    """
    n = a.shape[axis]
    freqs = np.fft.fftfreq(n).reshape([-1 if i == axis else 1 for i in range(a.ndim)])
    spectrum = np.fft.fft(a, axis=axis) * np.exp(-2j * np.pi * freqs * shift)
    return np.real(np.fft.ifft(spectrum, axis=axis)).astype(a.dtype)
