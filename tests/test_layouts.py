import pytest

from upmix.io.layouts import LAYOUTS, get_layout


def test_wav_channel_orders():
    # These orders are the WAVEFORMATEXTENSIBLE standard; if they change,
    # every file we ever wrote plays through the wrong speakers.
    assert LAYOUTS["stereo"].channels == ("FL", "FR")
    assert LAYOUTS["5.1"].channels == ("FL", "FR", "FC", "LFE", "BL", "BR")
    assert LAYOUTS["7.1"].channels == ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR")


def test_index_lookup():
    layout = get_layout("5.1")
    assert layout.index("FL") == 0
    assert layout.index("LFE") == 3
    assert layout.num_channels == 6


def test_unknown_layout():
    with pytest.raises(ValueError, match="Unknown layout"):
        get_layout("22.2")
