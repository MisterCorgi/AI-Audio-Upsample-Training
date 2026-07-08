import numpy as np
import pytest
from tests.conftest import tone

from upmix.dsp.downmix import downmix_to_stereo
from upmix.io.layouts import get_layout


def _surround_with(layout_name: str, channel: str, signal: np.ndarray) -> np.ndarray:
    layout = get_layout(layout_name)
    data = np.zeros((len(signal), layout.num_channels))
    data[:, layout.index(channel)] = signal
    return data


def test_center_folds_at_minus_3db(sr):
    sig = tone(440, 0.1, sr)
    stereo = downmix_to_stereo(_surround_with("5.1", "FC", sig), get_layout("5.1"))
    np.testing.assert_allclose(stereo[:, 0], sig / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(stereo[:, 1], sig / np.sqrt(2), atol=1e-12)


def test_front_left_passes_through(sr):
    sig = tone(440, 0.1, sr)
    stereo = downmix_to_stereo(_surround_with("5.1", "FL", sig), get_layout("5.1"))
    np.testing.assert_allclose(stereo[:, 0], sig, atol=1e-12)
    np.testing.assert_allclose(stereo[:, 1], 0, atol=1e-12)


def test_side_channels_fold_in_7_1(sr):
    sig = tone(440, 0.1, sr)
    stereo = downmix_to_stereo(_surround_with("7.1", "SR", sig), get_layout("7.1"))
    np.testing.assert_allclose(stereo[:, 1], sig / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(stereo[:, 0], 0, atol=1e-12)


def test_channel_count_mismatch_raises():
    with pytest.raises(ValueError, match="channels"):
        downmix_to_stereo(np.zeros((100, 6)), get_layout("7.1"))
