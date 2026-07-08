import numpy as np
import pytest
from tests.conftest import energy, noise, tone

from upmix.dsp.baseline import upmix_baseline
from upmix.io.layouts import get_layout

L51 = get_layout("5.1")
L71 = get_layout("7.1")


def _ch(out, layout, name):
    return out[:, layout.index(name)]


def test_output_shape_and_length(sr):
    stereo = np.stack([noise(0.3, sr, seed=1), noise(0.3, sr, seed=2)], axis=1)
    out51 = upmix_baseline(stereo, sr, "5.1")
    out71 = upmix_baseline(stereo, sr, "7.1")
    assert out51.shape == (stereo.shape[0], 6)
    assert out71.shape == (stereo.shape[0], 8)


def test_centered_content_goes_to_center(sr):
    sig = tone(440, 1.0, sr)
    stereo = np.stack([sig, sig], axis=1)  # identical L/R = phantom center
    out = upmix_baseline(stereo, sr, "5.1")
    center = energy(_ch(out, L51, "FC"))
    fronts = energy(_ch(out, L51, "FL")) + energy(_ch(out, L51, "FR"))
    surrounds = energy(_ch(out, L51, "BL")) + energy(_ch(out, L51, "BR"))
    assert center > 5 * fronts
    assert center > 20 * surrounds


def test_hard_left_stays_front_left(sr):
    stereo = np.stack([tone(440, 1.0, sr), np.zeros(sr)], axis=1)
    out = upmix_baseline(stereo, sr, "5.1")
    fl = energy(_ch(out, L51, "FL"))
    assert fl > 20 * energy(_ch(out, L51, "FR"))
    assert fl > 20 * energy(_ch(out, L51, "FC"))
    assert fl > 20 * energy(_ch(out, L51, "BL"))


def test_decorrelated_content_goes_to_surrounds(sr):
    # Independent noise in L and R = diffuse ambience
    stereo = np.stack([noise(1.0, sr, seed=10), noise(1.0, sr, seed=20)], axis=1)
    out = upmix_baseline(stereo, sr, "5.1")
    surrounds = energy(_ch(out, L51, "BL")) + energy(_ch(out, L51, "BR"))
    center = energy(_ch(out, L51, "FC"))
    assert surrounds > 2 * center
    assert surrounds > 0.1 * energy(stereo[:, 0])


def test_lfe_contains_only_bass(sr):
    # 60 Hz + 2 kHz mixed; LFE should keep the former, reject the latter
    sig = tone(60, 1.0, sr) + tone(2000, 1.0, sr)
    stereo = np.stack([sig, sig], axis=1)
    out = upmix_baseline(stereo, sr, "5.1")
    lfe = _ch(out, L51, "LFE")
    spectrum = np.abs(np.fft.rfft(lfe))
    freqs = np.fft.rfftfreq(len(lfe), 1 / sr)
    bass = spectrum[np.abs(freqs - 60) < 5].max()
    mid = spectrum[np.abs(freqs - 2000) < 5].max()
    assert bass > 100 * mid


def test_7_1_sides_lead_backs(sr):
    stereo = np.stack([noise(0.5, sr, seed=3), noise(0.5, sr, seed=4)], axis=1)
    out = upmix_baseline(stereo, sr, "7.1")
    sides = energy(_ch(out, L71, "SL")) + energy(_ch(out, L71, "SR"))
    backs = energy(_ch(out, L71, "BL")) + energy(_ch(out, L71, "BR"))
    assert sides > backs > 0


def test_rejects_non_stereo(sr):
    with pytest.raises(ValueError, match="stereo"):
        upmix_baseline(np.zeros((1000, 3)), sr)
