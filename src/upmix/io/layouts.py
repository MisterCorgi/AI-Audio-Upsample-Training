"""Channel layout definitions.

Channel order follows the WAVEFORMATEXTENSIBLE / SMPTE convention used by
multichannel WAV files. Getting this order wrong is a silent, catastrophic
bug (audio plays from the wrong speakers), so every layout is defined once
here and everything else refers to channels by name, never by bare index.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelLayout:
    name: str
    channels: tuple[str, ...]

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    def index(self, channel: str) -> int:
        return self.channels.index(channel)


STEREO = ChannelLayout("stereo", ("FL", "FR"))

# WAV standard order for 5.1: FL FR FC LFE BL BR
SURROUND_5_1 = ChannelLayout("5.1", ("FL", "FR", "FC", "LFE", "BL", "BR"))

# WAV standard order for 7.1: FL FR FC LFE BL BR SL SR
SURROUND_7_1 = ChannelLayout("7.1", ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"))

LAYOUTS: dict[str, ChannelLayout] = {
    "stereo": STEREO,
    "5.1": SURROUND_5_1,
    "7.1": SURROUND_7_1,
}


def get_layout(name: str) -> ChannelLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"Unknown layout {name!r}; available: {sorted(LAYOUTS)}") from None
