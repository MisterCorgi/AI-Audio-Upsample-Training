"""Multichannel audio file I/O.

All in-memory audio in this project is float64/float32 numpy arrays of shape
(num_samples, num_channels), channel order as defined in `upmix.io.layouts`.
"""

from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file.

    Returns (data, sample_rate) with data of shape (num_samples, num_channels),
    float64 in [-1, 1].
    """
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data, sr


def write_audio(
    path: str | Path,
    data: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> None:
    """Write (num_samples, num_channels) audio to a file.

    The container is inferred from the extension (.wav, .flac). Multichannel
    WAV written by libsndfile uses WAVEFORMATEXTENSIBLE with the standard
    speaker order, matching `upmix.io.layouts`.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected (num_samples, num_channels) array, got shape {data.shape}")
    sf.write(str(path), data, sample_rate, subtype=subtype)
