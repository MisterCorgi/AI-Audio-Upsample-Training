"""Classical DSP baseline upmixer (stereo -> 5.1 / 7.1).

Frequency-domain ambience/center extraction in the spirit of Avendano & Jot:
per time-frequency bin, time-smoothed auto/cross spectra of the L/R pair give

  * similarity  — how correlated L and R are (1 = same source, 0 = independent)
  * balance     — where the correlated content is panned (-1 = left, +1 = right)

From these we build soft masks:

  * ambience  = (1 - similarity) * energy-balance factor  -> surround channels
    (the energy factor keeps hard-panned single-channel content out of the
    surrounds: a source only in L has zero similarity but is direct, not
    ambience)
  * center    = similarity * (1 - |balance|)              -> center channel

Direct sound minus the extracted center stays in the front L/R. Surrounds are
gently low-passed and delayed (precedence effect keeps the front image
stable). The LFE is crossover-derived downstream, not extracted here.

This is the reference implementation the neural models must beat, and the
fallback renderer that ships in the CLI from day one.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from upmix.dsp.crossover import lfe_from_fronts
from upmix.dsp.stft import istft, stft
from upmix.io.layouts import ChannelLayout, get_layout

_EPS = 1e-12


@dataclass(frozen=True)
class BaselineConfig:
    n_fft: int = 4096
    hop: int = 1024
    smoothing: float = 0.85  # EMA forgetting factor for spectral statistics
    center_gain: float = 1.0
    surround_gain: float = 0.9
    lfe_gain: float = 0.5
    lfe_cutoff_hz: float = 120.0
    surround_lowpass_hz: float = 8000.0
    side_delay_ms: float = 8.0  # SL/SR (7.1 only)
    back_delay_ms: float = 14.0  # BL/BR


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average along the frame axis (works on complex)."""
    return lfilter([1.0 - alpha], [1.0, -alpha], x, axis=0)


def _delay(x: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return x
    return np.concatenate([np.zeros(samples), x[:-samples]])


def upmix_baseline(
    stereo: np.ndarray,
    sample_rate: int,
    layout: str | ChannelLayout = "5.1",
    config: BaselineConfig | None = None,
) -> np.ndarray:
    """Upmix (num_samples, 2) stereo audio to the given surround layout.

    Returns (num_samples, layout.num_channels) in the layout's channel order.
    """
    cfg = config or BaselineConfig()
    if isinstance(layout, str):
        layout = get_layout(layout)
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError(f"Expected (num_samples, 2) stereo input, got shape {stereo.shape}")

    length = stereo.shape[0]
    left, right = stereo[:, 0], stereo[:, 1]

    spec_l = stft(left, cfg.n_fft, cfg.hop)
    spec_r = stft(right, cfg.n_fft, cfg.hop)

    # Time-smoothed auto/cross spectral statistics
    p_ll = _ema(np.abs(spec_l) ** 2, cfg.smoothing).real
    p_rr = _ema(np.abs(spec_r) ** 2, cfg.smoothing).real
    p_lr = _ema(spec_l * np.conj(spec_r), cfg.smoothing)

    total = p_ll + p_rr + _EPS
    similarity = np.clip(2.0 * np.abs(p_lr) / total, 0.0, 1.0)
    balance = (p_ll - p_rr) / total
    energy_balance = 2.0 * np.minimum(p_ll, p_rr) / total

    ambience_mask = np.clip((1.0 - similarity) * energy_balance, 0.0, 1.0)
    center_mask = np.clip(similarity * (1.0 - np.abs(balance)) * cfg.center_gain, 0.0, 1.0)

    amb_l = ambience_mask * spec_l
    amb_r = ambience_mask * spec_r
    dir_l = spec_l - amb_l
    dir_r = spec_r - amb_r

    center = center_mask * 0.5 * (dir_l + dir_r)
    front_l = dir_l * (1.0 - center_mask)
    front_r = dir_r * (1.0 - center_mask)

    # Gentle low-pass on the surround feed: ambience is dull by nature, and
    # bright surrounds pull the image backwards.
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / sample_rate)
    surround_shelf = 1.0 / np.sqrt(1.0 + (freqs / cfg.surround_lowpass_hz) ** 8)
    amb_l = amb_l * surround_shelf
    amb_r = amb_r * surround_shelf

    fl = istft(front_l, length, cfg.n_fft, cfg.hop)
    fr = istft(front_r, length, cfg.n_fft, cfg.hop)
    fc = istft(center, length, cfg.n_fft, cfg.hop)
    surr_l = istft(amb_l, length, cfg.n_fft, cfg.hop) * cfg.surround_gain
    surr_r = istft(amb_r, length, cfg.n_fft, cfg.hop) * cfg.surround_gain

    # LFE draws from the full front bed (center folded in at -3 dB) so that
    # centered content — the common case for voice — still feeds the sub.
    fold = 1.0 / np.sqrt(2.0)
    lfe = lfe_from_fronts(
        fl + fold * fc, fr + fold * fc, sample_rate, cfg.lfe_cutoff_hz, cfg.lfe_gain
    )

    back_delay = int(cfg.back_delay_ms * sample_rate / 1000)
    out = {
        "FL": fl,
        "FR": fr,
        "FC": fc,
        "LFE": lfe,
        "BL": _delay(surr_l, back_delay),
        "BR": _delay(surr_r, back_delay),
    }
    if "SL" in layout.channels:
        side_delay = int(cfg.side_delay_ms * sample_rate / 1000)
        # 7.1: split the ambience between sides (earlier, fuller) and backs
        # (later, softer) so the surround field wraps around the listener.
        out["SL"] = _delay(surr_l, side_delay)
        out["SR"] = _delay(surr_r, side_delay)
        out["BL"] = _delay(surr_l * 0.7, back_delay)
        out["BR"] = _delay(surr_r * 0.7, back_delay)

    return np.stack([out[name] for name in layout.channels], axis=1)
