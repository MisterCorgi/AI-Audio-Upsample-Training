import numpy as np
import pytest

SR = 48000


@pytest.fixture
def sr() -> int:
    return SR


def tone(freq: float, duration: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def noise(duration: float = 1.0, sr: int = SR, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(duration * sr))


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x) ** 2))
