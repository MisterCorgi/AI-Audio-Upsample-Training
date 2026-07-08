import numpy as np
from tests.conftest import tone

from upmix.dsp.crossover import lfe_from_fronts


def _gain_at(freq: float, sr: int) -> float:
    sig = tone(freq, 0.5, sr, amp=1.0)
    lfe = lfe_from_fronts(sig, sig, sr, gain=1.0)
    # steady-state region only (skip filter edges)
    core = slice(len(sig) // 4, -len(sig) // 4)
    return float(np.sqrt(np.mean(lfe[core] ** 2) / np.mean(sig[core] ** 2)))


def test_bass_passes(sr):
    assert _gain_at(50, sr) > 0.9


def test_midrange_blocked(sr):
    assert _gain_at(1000, sr) < 0.01


def test_gain_applied(sr):
    sig = tone(50, 0.5, sr)
    half = lfe_from_fronts(sig, sig, sr, gain=0.5)
    full = lfe_from_fronts(sig, sig, sr, gain=1.0)
    np.testing.assert_allclose(half, full * 0.5, atol=1e-12)
