# AI Audio Upsample Training

Neural upmixing of 2-channel audio into **5.1 and 7.1 surround**, with a flagship focus on
**binaural voice recordings (ASMR)**: decoding the spatial cues baked into binaural audio and
re-rendering them onto physical surround speakers for true room-scale positioning.

## Status

📋 **Planning phase.** The complete project plan — architecture, data strategy, roadmap, and
risks — lives in [PLAN.md](PLAN.md).

## What this will become

```bash
# the end goal
upmix convert asmr_recording.wav --layout 5.1 --profile binaural-voice -o output.wav
```

- Offline CLI converter: stereo/binaural file in → 5.1/7.1 file out
- Two model profiles: generic **music/film upmix** and a specialized **binaural decoder**
  that reads interaural cues (ITD/ILD/HRTF fingerprints) to place sources where they
  actually were
- Trainable end-to-end on a single consumer GPU using an on-the-fly synthetic data engine
  (HRTF-rendered binaural inputs paired with VBAP-rendered surround targets)

## Roadmap at a glance

| Phase | Deliverable |
|---|---|
| 0 | Repo scaffolding, multichannel I/O, CI |
| 1 | Working CLI with classical DSP upmix baseline |
| 2 | Synthetic paired-data engine (HRTF ↔ VBAP) |
| 3 | Model v1: generic stereo → 5.1 |
| 4 | Binaural/ASMR model, 7.1 support |
| 5 | Listening tests & tuning |
| 6 | v0.1.0 release, checkpoints on Hugging Face Hub |

See [PLAN.md](PLAN.md) for the full details.

## License

[MIT](LICENSE)
