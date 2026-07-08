"""ITU-R BS.775 downmix of surround layouts back to stereo.

Used both as a quality check (fold the upmix back down and compare with the
original input) and, in later phases, as a training-time consistency loss.
"""

import numpy as np

from upmix.io.layouts import ChannelLayout

# ITU-R BS.775: center and surrounds fold in at -3 dB. LFE is conventionally
# dropped in studio downmixes; we include it at -3 dB so no content is lost
# when verifying our own crossover-derived LFE.
_COEF = 1.0 / np.sqrt(2.0)

_LEFT_MIX = {"FL": 1.0, "FC": _COEF, "LFE": _COEF, "BL": _COEF, "SL": _COEF}
_RIGHT_MIX = {"FR": 1.0, "FC": _COEF, "LFE": _COEF, "BR": _COEF, "SR": _COEF}


def downmix_to_stereo(data: np.ndarray, layout: ChannelLayout) -> np.ndarray:
    """Downmix (num_samples, num_channels) surround audio to (num_samples, 2)."""
    if data.shape[1] != layout.num_channels:
        raise ValueError(
            f"Data has {data.shape[1]} channels but layout {layout.name!r} "
            f"has {layout.num_channels}"
        )
    left = np.zeros(data.shape[0])
    right = np.zeros(data.shape[0])
    for name, gain in _LEFT_MIX.items():
        if name in layout.channels:
            left += gain * data[:, layout.index(name)]
    for name, gain in _RIGHT_MIX.items():
        if name in layout.channels:
            right += gain * data[:, layout.index(name)]
    return np.stack([left, right], axis=1)
