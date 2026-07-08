import numpy as np
from tests.conftest import tone

from upmix.cli import main
from upmix.io.audio import read_audio, write_audio


def _make_stereo(tmp_path, sr):
    path = tmp_path / "in.wav"
    data = np.stack([tone(440, 0.5, sr), tone(880, 0.5, sr)], axis=1)
    write_audio(path, data, sr)
    return path


def test_convert_5_1(tmp_path, sr):
    src = _make_stereo(tmp_path, sr)
    out = tmp_path / "out.wav"
    assert main(["convert", str(src), "-o", str(out), "--layout", "5.1"]) == 0
    data, out_sr = read_audio(out)
    assert data.shape[1] == 6
    assert out_sr == sr


def test_convert_7_1_default_output_name(tmp_path, sr):
    src = _make_stereo(tmp_path, sr)
    assert main(["convert", str(src), "--layout", "7.1"]) == 0
    data, _ = read_audio(tmp_path / "in.7_1.wav")
    assert data.shape[1] == 8


def test_info(tmp_path, sr, capsys):
    src = _make_stereo(tmp_path, sr)
    assert main(["info", str(src)]) == 0
    captured = capsys.readouterr().out
    assert "2 channels" in captured
    assert "correlation" in captured


def test_train_stub(capsys):
    assert main(["train"]) == 2
    assert "Phase 3" in capsys.readouterr().out
