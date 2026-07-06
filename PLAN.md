# Project Plan: AI Stereo → 5.1 / 7.1 Surround Upsampling

**Goal:** Train neural models that upmix 2-channel audio into 5.1 and 7.1 surround, with the
flagship use case being **binaural voice recordings (ASMR)** rendered onto physical surround
speakers so the spatial cues baked into the binaural recording become real, room-scale
positioning.

**Constraints and decisions (locked in):**

| Decision | Choice |
|---|---|
| Training hardware | Single consumer GPU (RTX-class, ~12–24 GB VRAM) |
| Deliverable | Offline CLI converter (`file in → multichannel file out`) |
| Playback target | Physical 5.1 / 7.1 speaker systems (ITU-R BS.775 layouts) |
| Stack | Python 3.11+, PyTorch, torchaudio |

---

## 1. Why this is two problems, not one

It is important to plan around the fact that the two inputs carry very different information:

1. **Generic stereo (music, film mixes).** The surround version doesn't exist and can't be
   "recovered" — upmixing is a *creative* task. The model learns plausible conventions:
   ambience and reverb to the surrounds, dialog/lead to the center, dry instruments in the
   front stage. Ground truth comes from real 5.1 masters and from synthetic mixes.

2. **Binaural recordings (the ASMR goal).** These *do* contain real spatial information:
   interaural time differences (ITD), level differences (ILD), and the spectral fingerprints
   of the dummy-head HRTF. Here the task is well-posed: **decode the encoded positions, then
   re-render the sources at those positions on a speaker layout.** This is where the project
   can genuinely outperform classical upmixers (Dolby Pro Logic II, DTS Neural:X), which
   ignore HRTF cues entirely.

The plan builds the generic upmixer first (it's the easier data problem and produces all the
shared infrastructure), then specializes into the binaural decoder, which is the differentiator.

**Key insight for training data:** for the binaural task we can synthesize *unlimited perfectly
paired data*. Take a dry mono source (voice, mouth sounds, brushes, taps), place it at a known
position, render it two ways:

- through an HRTF → the **binaural input**
- through VBAP speaker panning → the **5.1/7.1 ground truth**

Same source, same trajectory, two renderings. The model learns the mapping between them.

---

## 2. System architecture

### 2.1 Processing pipeline (inference)

```
input.wav (2ch)
   │
   ├─ analysis: detect binaural vs plain stereo (heuristic or tiny classifier)
   │
   ├─ neural upmix model (per profile: "music", "binaural-voice")
   │     2ch STFT in → 6ch / 8ch STFT out (complex masks + direct mapping)
   │
   ├─ post-processing
   │     • LFE: crossover-derived (<120 Hz from front bed), configurable
   │     • center-channel policy: hard / phantom / blend (ASMR often prefers phantom)
   │     • surround ambience level trim
   │     • downmix-consistency check (fold back to stereo, compare with input)
   │
   └─ output.wav / .flac (5.1 or 7.1, SMPTE channel order, 48 kHz)
```

### 2.2 Model design (v1)

- **Backbone:** band-split RNN (BSRNN-style) or hybrid U-Net operating on complex STFT.
  These are state of the art for source separation at sizes (10–40 M params) that train
  comfortably on one consumer GPU with mixed precision and 5–10 s crops.
- **Output head:** 6 (5.1) or 8 (7.1) channel complex spectrogram; LFE predicted but
  post-processing can override it with a DSP crossover (safer for real speakers).
- **Losses:**
  - multi-resolution STFT loss per output channel
  - SI-SDR per channel on the synthetic pairs
  - **spatial losses:** inter-channel level/time differences and acoustic energy-vector
    direction vs. ground truth — this is what makes positions land correctly
  - downmix-consistency loss: the ITU downmix of the output should reconstruct the input
- **Binaural variant:** same backbone, but input features are augmented with explicit
  interaural cues (IPD/ILD per T-F bin), which strongly help localization decoding.

### 2.3 Why not a modular pipeline (separate → localize → re-pan)?

A modular DSP+ML pipeline is more interpretable, and we *will* build a small version of it as
the Phase-1 baseline. But error compounds across stages, and end-to-end training with spatial
losses lets the network trade off separation vs. spatialization jointly. The baseline stays in
the repo forever as a comparison point and as a fallback renderer.

---

## 3. Data strategy

### 3.1 Source material (all free/research-licensed)

| Purpose | Datasets |
|---|---|
| Dry speech / voice | VCTK, LibriTTS-R, EARS (anechoic, 48 kHz) |
| ASMR-like foley (taps, brushes, crinkles, mouth sounds) | FSD50K subsets, ESC-50, custom recordings |
| Music stems (generic upmixer) | MUSDB18-HQ, MoisesDB |
| Real 5.1 references (eval + fine-tune) | Any owned surround masters; used for evaluation only if licensing is unclear |
| HRTFs | SADIE II, HUTUBS, CIPIC, Neumann KU100 (the de-facto ASMR dummy head) |

### 3.2 Synthetic pair generation (the core asset)

A `spatializer` module that, given dry sources:

1. samples positions/trajectories (static, slow orbits, ear-to-ear passes — typical ASMR moves),
2. samples a room (pyroomacoustics image-source RIRs; also anechoic, since much ASMR is close-mic),
3. renders **binaural input** by HRTF convolution (randomly drawn HRTF per example — this is
   the key augmentation that prevents overfitting to one dummy head),
4. renders **5.1/7.1 target** by VBAP on the ITU layout, with the room's diffuse tail sent
   decorrelated to the surrounds,
5. mixes 1–5 simultaneous sources with realistic level distributions.

Generation happens **on the fly during training** (CPU workers feeding the GPU), so a single
consumer GPU sees effectively infinite data without a terabyte-scale preprocessing step.
A small frozen validation/test set is pre-rendered for reproducible metrics.

### 3.3 Generic-stereo pairs

For the music/film profile: real 5.1 content downmixed to stereo with the standard ITU
coefficients gives (stereo, 5.1) pairs; MUSDB stems panned to synthetic 5.1 mixes extend this.

---

## 4. Evaluation

- **Objective, synthetic test set:** per-channel SI-SDR, multi-res STFT distance,
  energy-vector angular error (degrees), ILD/ITD error. Frozen test set, versioned.
- **Downmix consistency on real recordings:** fold the generated 5.1 back to stereo and
  measure distance to the original — the only ground-truth-free objective check we have
  for real ASMR content.
- **Baselines to beat:** passive matrix upmix (Pro-Logic-II-style, implemented in Phase 1)
  and a mid/side ambience-extraction upmixer.
- **Listening tests:** small MUSHRA-style comparisons on a real 5.1 setup; ASMR clips scored
  for localization accuracy, timbre preservation, and absence of artifacts. Even informal,
  scheduled at every phase gate — spatial audio quality does not show up fully in metrics.

---

## 5. Repository layout & tooling ("easy to work with")

```
AI-Audio-Upsample-Training/
├── pyproject.toml          # single source of deps; `pip install -e .[dev]` or uv
├── README.md               # quickstart: install, convert a file, train
├── PLAN.md                 # this document
├── configs/                # YAML experiment configs (model, data, training)
│   ├── music_5_1.yaml
│   └── binaural_5_1.yaml
├── src/upmix/
│   ├── cli.py              # `upmix convert|train|evaluate|listen`
│   ├── io/                 # multichannel WAV/FLAC read-write, channel-order handling
│   ├── dsp/                # STFT, VBAP, crossover/LFE, matrix-upmix baseline, downmix
│   ├── spatial/            # HRTF loading, binaural renderer, room sim, trajectory sampling
│   ├── data/               # on-the-fly synthetic dataset, MUSDB/real-pair datasets
│   ├── models/             # BSRNN backbone, heads, losses (incl. spatial losses)
│   ├── train/              # Lightning module, checkpointing, resume
│   └── eval/               # metrics, test-set runner, report generation
├── scripts/                # dataset download/prep, test-set freezing
├── tests/                  # pytest; DSP correctness (downmix/VBAP/ITD) is heavily unit-tested
└── notebooks/              # exploration only, never load-bearing
```

Conventions that keep it friendly:

- **One CLI for everything** (`upmix …`), including training — no memorizing script paths.
- **Configs are plain YAML + dataclasses** (no framework magic); every training run logs its
  resolved config next to the checkpoint.
- **Everything resumable:** training checkpoints + `--resume`, dataset generation idempotent.
- **CI (GitHub Actions):** ruff + pytest on CPU, plus a 2-minute smoke run (tiny model, tiny
  synthetic batch) so training-loop breakage is caught in PRs, not after an overnight run.
- **Model zoo:** released checkpoints on Hugging Face Hub; `upmix convert` auto-downloads.
- Experiment tracking: TensorBoard by default, Weights & Biases optional via config flag.

---

## 6. Roadmap

### Phase 0 — Scaffolding (~1 week)
- [ ] Repo skeleton, `pyproject.toml`, ruff, pytest, CI
- [ ] Multichannel audio I/O with explicit channel-order handling (SMPTE/WAV vs FFmpeg orders
      are a classic silent-bug source — unit-test this first)
- **Gate:** `pip install -e .` + `upmix --help` works; CI green.

### Phase 1 — DSP foundation + usable CLI (~2 weeks)
- [ ] STFT utilities, ITU downmix, VBAP panner, LFE crossover
- [ ] Classical baseline upmixer (passive matrix + mid/side ambience extraction)
- [ ] `upmix convert in.wav --layout 5.1 --profile baseline -o out.wav`
- **Gate:** the tool already converts files end-to-end (DSP only). Useful from week 3, and
  every later model slots into an already-working CLI.

### Phase 2 — Synthetic data engine (~3 weeks)
- [ ] HRTF loader (SOFA files) + binaural renderer, multi-HRTF augmentation
- [ ] Room simulation + trajectory sampler (ASMR-style movement patterns)
- [ ] Paired rendering (binaural in / VBAP 5.1 target), on-the-fly training dataset
- [ ] Frozen validation + test sets, dataset download scripts
- **Gate:** listen-check notebook: rendered binaural clips localize convincingly on
  headphones and targets sound correct on speakers.

### Phase 3 — Model v1: generic stereo → 5.1 (~4 weeks)
- [ ] BSRNN backbone + losses (incl. spatial + downmix-consistency)
- [ ] Training loop (mixed precision, grad accumulation, resume), eval harness
- [ ] Train on MUSDB-derived + synthetic pairs; beat both DSP baselines on metrics
- **Gate:** objective metrics beat baselines; informal listening confirms no obvious artifacts.

### Phase 4 — Binaural specialization: the ASMR model (~4 weeks)
- [ ] Interaural-cue input features (IPD/ILD)
- [ ] Train binaural → 5.1 on the synthetic engine with voice/foley sources
- [ ] 7.1 output head (retrain or adapt; VBAP targets already support it)
- [ ] Fine-tune center/LFE/ambience post-processing defaults for ASMR
- **Gate:** on the synthetic test set, energy-vector error ≤ ~10°; real ASMR clips place
  voices convincingly behind/beside the listener on a physical 5.1 rig.

### Phase 5 — Evaluation, tuning, listening tests (~2 weeks)
- [ ] MUSHRA-style listening sessions vs. Pro-Logic-II baseline
- [ ] Failure-case catalog (reverberant rooms, music under voice, HRTF mismatch)
- [ ] Tune post-processing defaults per profile
- **Gate:** binaural model preferred over baseline in listening tests.

### Phase 6 — Packaging & release (~1–2 weeks)
- [ ] Checkpoints to Hugging Face Hub, auto-download in CLI
- [ ] Docs: README quickstart, conversion guide, training-reproduction guide
- [ ] Tagged v0.1.0 release, example clips (before/after)

### Future (explicitly out of scope for v0.1)
- Realtime/streaming inference (causal model variant, ONNX export)
- Dolby Atmos / object-based output (ADM BWF)
- Height channels (5.1.2 / 7.1.4)
- Automatic binaural-vs-stereo classifier feeding profile selection

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **HRTF mismatch** — real ASMR uses many different dummy heads/mics | Randomize HRTFs during training; evaluate cross-HRTF (train on N-1 sets, test on held-out) |
| **Reverb entanglement** — room tail confuses localization | Train with and without rooms; diffuse-vs-direct split is explicitly in the synthetic targets |
| **Generic upmix is subjective** | Ship it as a "profile" with tunable post-processing, not a single truth; downmix-consistency loss keeps it honest |
| **Single-GPU limits** | On-the-fly synthesis (no giant dataset), 10–40 M param models, AMP, short crops, gradient accumulation; each phase's model trains in ≤ ~3 days |
| **Metrics ≠ perception** | Listening gate at every phase; frozen test clips re-used across phases for A/B continuity |
| **Channel-order/format bugs** (silent, catastrophic) | Phase-0 unit tests, round-trip tests, known-signal fixtures (e.g., tone panned hard-left must land in FL) |

---

## 8. Immediate next steps

1. Phase 0 scaffolding PR: package skeleton, CLI stub, I/O module with channel-order tests, CI.
2. Phase 1 DSP PR: baseline upmixer — a working converter within the first two weeks.
