import numpy as np
import pytest
from tests.conftest import tone

from upmix.io.audio import read_audio, write_audio


def test_multichannel_roundtrip_preserves_channel_identity(tmp_path, sr):
    # Each channel gets a distinct frequency; after write+read, each channel
    # must still contain its own frequency (i.e., no channel reordering).
    freqs = [100, 200, 400, 800, 1600, 3200]
    data = np.stack([tone(f, 0.25, sr) for f in freqs], axis=1)
    path = tmp_path / "six.wav"
    write_audio(path, data, sr, subtype="FLOAT")

    loaded, loaded_sr = read_audio(path)
    assert loaded_sr == sr
    assert loaded.shape == data.shape
    np.testing.assert_allclose(loaded, data, atol=1e-6)


def test_read_always_2d(tmp_path, sr):
    path = tmp_path / "mono.wav"
    write_audio(path, tone(440, 0.1, sr)[:, None], sr)
    loaded, _ = read_audio(path)
    assert loaded.ndim == 2
    assert loaded.shape[1] == 1


def test_write_rejects_1d(tmp_path, sr):
    with pytest.raises(ValueError, match="num_samples, num_channels"):
        write_audio(tmp_path / "bad.wav", np.zeros(100), sr)
