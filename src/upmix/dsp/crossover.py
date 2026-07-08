"""LFE channel derivation.

The LFE channel is derived from the front bed with a low-pass crossover
(default 120 Hz, 4th-order Butterworth, zero-phase). The fronts are left
full-range: bass management is the receiver's job, and removing bass from
the mains would break playback on systems without a subwoofer.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def lfe_from_fronts(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 120.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Derive an LFE signal from the front left/right pair."""
    sos = butter(4, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    mono = 0.5 * (left + right)
    return gain * sosfiltfilt(sos, mono)
