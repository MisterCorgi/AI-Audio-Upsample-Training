"""Minimal STFT/iSTFT with exact reconstruction.

Hand-rolled (rather than scipy.signal) so the framing, padding, and
normalization are fully controlled and unit-tested — the neural models in
later phases must share this exact transform.
"""

import numpy as np


def _hann(n_fft: int) -> np.ndarray:
    # Periodic Hann window (COLA-compliant for hop = n_fft / 4)
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)


def stft(x: np.ndarray, n_fft: int = 4096, hop: int = 1024) -> np.ndarray:
    """STFT of a mono signal (num_samples,) -> (num_frames, n_fft // 2 + 1) complex."""
    if x.ndim != 1:
        raise ValueError(f"stft expects a mono signal, got shape {x.shape}")
    pad = n_fft // 2
    x = np.pad(x, (pad, pad))
    num_frames = 1 + (len(x) - n_fft) // hop
    window = _hann(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[:: hop][:num_frames]
    return np.fft.rfft(frames * window, axis=-1)


def istft(spec: np.ndarray, length: int, n_fft: int = 4096, hop: int = 1024) -> np.ndarray:
    """Inverse of `stft`. `length` is the original signal length."""
    window = _hann(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=-1) * window
    total = (spec.shape[0] - 1) * hop + n_fft
    out = np.zeros(total)
    norm = np.zeros(total)
    win_sq = window**2
    for i in range(spec.shape[0]):
        start = i * hop
        out[start : start + n_fft] += frames[i]
        norm[start : start + n_fft] += win_sq
    out /= np.maximum(norm, 1e-10)
    pad = n_fft // 2
    return out[pad : pad + length]
