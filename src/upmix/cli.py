"""The `upmix` command-line interface.

    upmix convert input.wav --layout 5.1 -o output.wav
    upmix info input.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from upmix import __version__
from upmix.dsp.baseline import BaselineConfig, upmix_baseline
from upmix.io.audio import read_audio, write_audio
from upmix.io.layouts import get_layout

PROFILES = ("baseline",)  # neural profiles land in Phase 3/4


def _cmd_convert(args: argparse.Namespace) -> int:
    layout = get_layout(args.layout)
    data, sr = read_audio(args.input)

    if data.shape[1] == 1:
        print("note: mono input, duplicating to stereo before upmix")
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] != 2:
        print(f"error: expected mono or stereo input, got {data.shape[1]} channels")
        return 1

    output = args.output
    if output is None:
        stem = Path(args.input).with_suffix("")
        output = Path(f"{stem}.{layout.name.replace('.', '_')}.wav")

    config = BaselineConfig(
        center_gain=args.center_gain,
        surround_gain=args.surround_gain,
        lfe_gain=args.lfe_gain,
    )
    print(f"converting {args.input} -> {output} ({layout.name}, profile={args.profile})")
    result = upmix_baseline(data, sr, layout, config)

    peak = np.max(np.abs(result))
    if peak > 0.99:
        result = result * (0.99 / peak)
        print(f"note: output peaked at {peak:.2f}, normalized to 0.99")

    write_audio(output, result.astype(np.float32), sr)
    print(f"wrote {output} ({layout.num_channels} channels, {sr} Hz)")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    data, sr = read_audio(args.input)
    n, ch = data.shape
    print(f"{args.input}: {ch} channels, {sr} Hz, {n / sr:.2f} s, peak {np.max(np.abs(data)):.3f}")
    if ch == 2:
        left, right = data[:, 0], data[:, 1]
        denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
        corr = float(np.sum(left * right) / denom) if denom > 0 else 1.0
        print(f"L/R correlation: {corr:+.3f} (1 = mono-like, ~0 = wide/decorrelated)")
    return 0


def _cmd_stub(name: str) -> int:
    print(f"`upmix {name}` is not implemented yet — it arrives with the neural models (Phase 3).")
    print("See PLAN.md for the roadmap.")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="upmix",
        description="Upmix stereo/binaural audio to 5.1 and 7.1 surround.",
    )
    parser.add_argument("--version", action="version", version=f"upmix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert a stereo file to surround")
    convert.add_argument("input", help="Input audio file (wav/flac, mono or stereo)")
    convert.add_argument("-o", "--output", help="Output file (default: <input>.<layout>.wav)")
    convert.add_argument("--layout", default="5.1", choices=["5.1", "7.1"])
    convert.add_argument("--profile", default="baseline", choices=PROFILES)
    convert.add_argument("--center-gain", type=float, default=1.0)
    convert.add_argument("--surround-gain", type=float, default=0.9)
    convert.add_argument("--lfe-gain", type=float, default=0.5)
    convert.set_defaults(func=_cmd_convert)

    info = sub.add_parser("info", help="Show information about an audio file")
    info.add_argument("input")
    info.set_defaults(func=_cmd_info)

    for name, help_text in [
        ("train", "Train an upmix model (Phase 3)"),
        ("evaluate", "Evaluate a model on the test set (Phase 3)"),
    ]:
        stub = sub.add_parser(name, help=help_text)
        stub.set_defaults(func=lambda _args, _name=name: _cmd_stub(_name))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
