import numpy as np
from tests.conftest import noise

from upmix.dsp.stft import istft, stft


def test_perfect_reconstruction(sr):
    x = noise(0.5, sr)
    spec = stft(x)
    y = istft(spec, length=len(x))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_reconstruction_odd_length():
    x = noise(0.1)[:4801]
    y = istft(stft(x), length=len(x))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_tone_lands_in_correct_bin(sr):
    n_fft = 4096
    freq = sr / n_fft * 100  # exactly bin 100
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * freq * t)
    spec = stft(x, n_fft=n_fft)
    mid_frame = np.abs(spec[spec.shape[0] // 2])
    assert np.argmax(mid_frame) == 100
