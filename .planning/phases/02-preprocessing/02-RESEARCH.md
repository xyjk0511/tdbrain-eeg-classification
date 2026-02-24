# Phase 2: Preprocessing - Research

**Researched:** 2026-02-23
**Domain:** MNE-Python EEG preprocessing (filter, re-reference, epoch, artifact rejection)
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PREP-01 | 基础预处理：带通滤波（1-40 Hz）、重参考（平均参考） | `raw.filter(1.0, 40.0)` + `raw.set_eeg_reference('average')` — 实测验证 |
| PREP-02 | 分段（epoch）：静息态切成固定长度片段（如 2s） | `mne.make_fixed_length_epochs(raw, duration=2.0)` — 实测 60 epochs/subject |
| PREP-03 | 简单伪迹剔除：振幅阈值法（±100 µV） | `epochs.drop_bad(reject=dict(eeg=100e-6))` — 实测 1.7% 剔除率 |
</phase_requirements>

## Summary

Phase 1 loads Raw objects with `preload=False`. Phase 2 must call `raw.load_data()` first, then apply: (1) pick 26 scalp EEG channels, (2) bandpass filter 1-40 Hz, (3) average re-reference, (4) segment into 2s epochs, (5) drop epochs exceeding ±100 µV peak-to-peak.

All 33 channels in TDBRAIN are typed as `eeg` in both MNE and the BIDS channels.tsv — the 7 non-scalp channels (VPVA, VNVB, HPHL, HNHR, Erbs, OrbOcc, Mass) must be excluded by explicit name list before re-referencing. Including them in the average reference would corrupt the reference.

The full pipeline was verified on real data: 26 ch, 60 epochs per subject, ~16 MB per subject in memory, ~1.7% epoch rejection rate.

**Primary recommendation:** `raw.pick(EEG_CHANNELS)` → `raw.filter(1.0, 40.0)` → `raw.set_eeg_reference('average')` → `make_fixed_length_epochs(duration=2.0)` → `epochs.drop_bad(reject=dict(eeg=100e-6))`.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mne | 1.11.0 | filter, re-reference, epoch, artifact rejection | Already installed; all required APIs verified |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | Path construction | Already used in Phase 1 |

**Installation:** No new dependencies needed.

## Architecture Patterns

### Recommended Project Structure

```
tdbrain/
├── config.py          # EEG_CHANNELS list, filter params, epoch duration, reject threshold
├── data_loader.py     # load_subjects(), load_raw()  [Phase 1]
├── preprocessor.py    # preprocess(raw) -> (Epochs, stats)  [Phase 2]
└── main.py            # orchestration
```

### Pattern 1: preprocess() function

**What:** Takes a Raw object (preload=False from Phase 1), returns clean Epochs + rejection stats.

```python
# Source: verified against MNE 1.11.0 on real TDBRAIN data (sub-19681349 EC)
import mne

EEG_CHANNELS = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC3','FCz','FC4','T7','C3','Cz','C4','T8',
    'CP3','CPz','CP4','P7','P3','Pz','P4','P8',
    'O1','Oz','O2'
]

def preprocess(raw: mne.io.Raw) -> tuple:
    raw.load_data(verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)
    n_after = len(epochs)
    return epochs, {'n_before': n_before, 'n_after': n_after}
```

### Anti-Patterns to Avoid

- **Re-reference before picking:** Including VPVA/VNVB/etc. in the average reference corrupts it — always `pick()` first.
- **Filter after epoching:** Filter the continuous Raw, not the Epochs — avoids edge artifacts at epoch boundaries.
- **`projection=True` for average reference:** Adds a projection vector instead of applying directly; use `projection=False`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FIR bandpass filter | Custom filter coefficients | `raw.filter(1.0, 40.0)` | MNE handles transition bands, zero-phase, edge padding |
| Fixed-length segmentation | Manual array slicing | `make_fixed_length_epochs()` | Handles boundary conditions cleanly |
| Peak-to-peak artifact rejection | Manual threshold loop | `epochs.drop_bad(reject=dict(eeg=...))` | Handles per-channel PTP, logs drop reasons |

## Common Pitfalls

### Pitfall 1: Non-scalp channels included in average reference

**What goes wrong:** Average reference computed over 33 channels including EOG/EMG channels — reference is biased.
**Why it happens:** All 33 channels are typed `eeg` in both MNE and BIDS channels.tsv.
**How to avoid:** Always call `raw.pick(EEG_CHANNELS)` with the explicit 26-channel list before `set_eeg_reference`.
**Warning signs:** Channel count is still 33 after re-referencing.

### Pitfall 2: Calling filter() on preload=False Raw

**What goes wrong:** `ValueError: Raw data needs to be preloaded to filter.`
**Why it happens:** Phase 1 loads with `preload=False` to save memory.
**How to avoid:** Call `raw.load_data()` at the start of `preprocess()`.
**Warning signs:** ValueError on `raw.filter()`.

### Pitfall 3: ±100 µV threshold is peak-to-peak, not absolute

**What goes wrong:** MNE `reject=dict(eeg=100e-6)` uses peak-to-peak (max - min), not absolute amplitude.
**Why it happens:** MNE PTP definition differs from "absolute ±100 µV" intuition.
**How to avoid:** `dict(eeg=100e-6)` rejects epochs where PTP > 100 µV (conservative). If requirement means absolute ±100 µV swing, use `dict(eeg=200e-6)`. Document the choice in config.
**Warning signs:** Unexpected rejection rates.

### Pitfall 4: Memory accumulation across 520 subjects

**What goes wrong:** Storing all Epochs objects exhausts RAM (520 × 60 × 26 × 1000 × 8 bytes ≈ 6.5 GB).
**Why it happens:** Preprocessing all subjects before feature extraction.
**How to avoid:** Process one subject at a time: preprocess → extract features → discard Epochs.
**Warning signs:** Memory grows linearly with subject count.

## Code Examples

### Epoch shape

```
epochs.get_data().shape == (n_epochs, 26, 1000)
# n_epochs: up to 60 (120s / 2s)
# 26: scalp EEG channels
# 1000: samples (2s × 500 Hz)
```

## Open Questions

1. **±100 µV threshold interpretation**
   - What we know: MNE `reject=dict(eeg=100e-6)` uses peak-to-peak amplitude
   - What's unclear: Requirements say "振幅超过 ±100 µV" — absolute or PTP?
   - Recommendation: Use `dict(eeg=100e-6)` as conservative default; document in config.

## Sources

### Primary (HIGH confidence)

- MNE 1.11.0 installed — `raw.filter()`, `raw.set_eeg_reference()`, `make_fixed_length_epochs()`, `epochs.drop_bad()` verified via `help()` and live execution
- Real data test on `sub-19681349` EC — full pipeline verified end-to-end
- TDBRAIN channels.tsv — confirmed all 33 channels typed as EEG

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — MNE 1.11.0 installed, all APIs verified
- Architecture: HIGH — full pipeline tested on real data
- Pitfalls: HIGH — non-scalp channel issue confirmed by actual data inspection

**Research date:** 2026-02-23
**Valid until:** 2026-03-23
