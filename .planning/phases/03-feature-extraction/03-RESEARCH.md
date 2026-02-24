# Phase 3: Feature Extraction - Research

**Researched:** 2026-02-23
**Domain:** MNE-Python spectral analysis -- band power, TBR, FAA, relative power
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-01 | 各频段绝对功率（delta/theta/alpha/beta/gamma） | epochs.compute_psd(method=welch) -> get_data() -> mean over freq bins; verified shape (54, 26, 40) |
| FEAT-02 | Theta/Beta ratio（额叶通道，ADHD 核心指标） | theta_mean[frontal] / beta_mean[frontal] over epochs; verified TBR ~5.97 on sub-87999321 |
| FEAT-03 | Frontal Alpha Asymmetry（F3/F4，MDD 核心指标） | ln(alpha_F4) - ln(alpha_F3) over epochs; verified FAA ~-0.045 on sub-87999321 |
| FEAT-04 | 相对功率（各频段功率 / 总功率） | band_power / sum(all_band_powers); verified 130 relative features per subject |
</phase_requirements>


## Summary

Phase 2 delivers Epochs objects with shape (n_epochs, 26, 1000) at 500 Hz. Phase 3 computes a per-subject feature vector by: (1) running Welch PSD on each epoch, (2) averaging PSD over epochs, (3) integrating over frequency bands for absolute power, (4) deriving relative power, TBR, and FAA.

epochs.compute_psd(method=welch, fmin=1.0, fmax=40.0, n_fft=500) returns an EpochsSpectrum. Calling .get_data(return_freqs=True) yields psds shape (n_epochs, 26, 40) and freqs (40,) at 1 Hz resolution. Band power is extracted by masking freqs and averaging over the frequency axis.

Final per-subject feature vector: 262 elements -- 130 absolute power (5 bands x 26 ch) + 130 relative power (5 bands x 26 ch) + 1 TBR + 1 FAA. All APIs verified on sub-87999321 EO.

**Primary recommendation:** epochs.compute_psd(method=welch, fmin=1.0, fmax=40.0, n_fft=500) -> numpy band masking -> per-subject mean -> single feature vector. No new dependencies needed.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mne | 1.11.0 | compute_psd, EpochsSpectrum.get_data | Already installed; all APIs verified on real data |
| numpy | (installed) | Band masking, log, mean, concatenation | Already used; no alternative needed |

**Installation:** No new dependencies needed.

## Architecture Patterns

### Recommended Project Structure

    tdbrain/
    config.py              # add FREQ_BANDS, FRONTAL_CHANNELS
    data_loader.py         # unchanged
    preprocessor.py        # unchanged
    feature_extractor.py   # extract_features(epochs) -> np.ndarray
    main.py                # add feature extraction loop

### Pattern 1: extract_features(epochs) -> 1D array

**What:** Takes a clean Epochs object, returns a flat 1D numpy array (262 features) for one subject.
**When to use:** Called once per subject inside the subject loop in main.py.

    # Source: verified against MNE 1.11.0 on real TDBRAIN data (sub-87999321 EO)
    import numpy as np
    import mne

    FREQ_BANDS = {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 40),
    }
    FRONTAL_CHANNELS = ["F3", "F4", "Fz", "F7", "F8"]

    def extract_features(epochs: mne.Epochs) -> np.ndarray:
        psd = epochs.compute_psd(method="welch", fmin=1.0, fmax=40.0, n_fft=500, verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)  # (n_epochs, 26, 40)
        ch_names = epochs.ch_names

        def band_mean(fmin, fmax):
            idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            return psds[:, :, idx].mean(axis=-1).mean(axis=0)  # (26,)

        abs_bp = {b: band_mean(lo, hi) for b, (lo, hi) in FREQ_BANDS.items()}
        total = sum(abs_bp.values())
        rel_bp = {b: abs_bp[b] / total for b in FREQ_BANDS}

        fi = [ch_names.index(c) for c in FRONTAL_CHANNELS]
        theta_ep = psds[:, :, np.where((freqs >= 4) & (freqs < 8))[0]].mean(axis=-1)
        beta_ep  = psds[:, :, np.where((freqs >= 13) & (freqs < 30))[0]].mean(axis=-1)
        alpha_ep = psds[:, :, np.where((freqs >= 8) & (freqs < 13))[0]].mean(axis=-1)

        tbr = (theta_ep[:, fi].mean(axis=1) / beta_ep[:, fi].mean(axis=1)).mean()
        f3i, f4i = ch_names.index("F3"), ch_names.index("F4")
        faa = (np.log(alpha_ep[:, f4i]) - np.log(alpha_ep[:, f3i])).mean()

        return np.concatenate([
            *[abs_bp[b] for b in FREQ_BANDS],
            *[rel_bp[b] for b in FREQ_BANDS],
            [tbr, faa],
        ])

### Pattern 2: Subject loop in main.py

    features, labels, subject_ids = [], [], []
    for _, row in subjects.iterrows():
        raw = load_raw(row["participants_ID"], CONDITION)
        epochs, _ = preprocess(raw)
        if len(epochs) == 0:
            continue
        features.append(extract_features(epochs))
        labels.append(row["indication"])
        subject_ids.append(row["participants_ID"])

    X = np.array(features)       # (n_subjects, 262)
    y = np.array(labels)         # (n_subjects,)
    groups = np.array(subject_ids)  # for StratifiedGroupKFold in Phase 4

### Anti-Patterns to Avoid

- **FAA without log:** Raw asymmetry (F4 - F3) is not standard. Literature formula is ln(alpha_F4) - ln(alpha_F3).
- **TBR with all channels:** TBR is a frontal measure. Using all 26 channels dilutes the ADHD signal.
- **np.trapz for band integration:** Mean over frequency bins is sufficient for uniform 1 Hz resolution.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Welch PSD | Manual FFT + windowing loop | epochs.compute_psd(method=welch) | Handles windowing, overlap, averaging, edge cases |
| Frequency bin selection | Manual frequency arithmetic | np.where((freqs >= fmin) & (freqs < fmax)) | Direct, no off-by-one risk |

## Common Pitfalls

### Pitfall 1: n_fft > epoch length

**What goes wrong:** ValueError: n_fft cannot be larger than signal size
**Why it happens:** Epoch is 1000 samples (2s x 500 Hz). n_fft must be <= 1000.
**How to avoid:** Use n_fft=500 (1s window, 2 windows per epoch, 1 Hz resolution).
**Warning signs:** ValueError on compute_psd.

### Pitfall 2: Zero epochs after artifact rejection

**What goes wrong:** extract_features called on empty Epochs -> empty array or division by zero.
**Why it happens:** Some subjects have high-amplitude recordings; all epochs rejected.
**How to avoid:** Check len(epochs) == 0 before calling extract_features; skip and log.
**Warning signs:** Feature vector of shape (0,) or NaN values.

### Pitfall 3: Channel index assumption

**What goes wrong:** Hard-coding channel indices breaks if channel order changes.
**Why it happens:** Assuming fixed order from config.
**How to avoid:** Always use ch_names.index("F3") dynamically from epochs.ch_names.
**Warning signs:** Wrong TBR/FAA values outside expected range.

## Code Examples

### Verified PSD shape

    # Source: verified on MNE 1.11.0, sub-87999321 EO
    psd = epochs.compute_psd(method="welch", fmin=1.0, fmax=40.0, n_fft=500, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    # psds.shape == (54, 26, 40)
    # freqs == [1., 2., ..., 40.]  (1 Hz resolution)

### Expected feature vector dimensions

    abs_power:  5 bands x 26 channels = 130 features
    rel_power:  5 bands x 26 channels = 130 features
    TBR:        1 scalar
    FAA:        1 scalar
    total:      262 features per subject

### Verified values on sub-87999321 EO

    TBR  ~5.97   (theta/beta ratio, frontal channels F3/F4/Fz/F7/F8)
    FAA  ~-0.045 (ln(alpha_F4) - ln(alpha_F3))

## Open Questions

1. **TBR frontal channel set**
   - What we know: F3, F4, Fz, F7, F8 are all frontal channels present in the 26-channel set
   - What is unclear: Some papers use only F3/F4/Fz; others include F7/F8
   - Recommendation: Use all 5; document as FRONTAL_CHANNELS in config

## Sources

### Primary (HIGH confidence)

- MNE 1.11.0 installed -- epochs.compute_psd(), EpochsSpectrum.get_data() verified via help() and live execution
- Real data test on sub-87999321 EO -- psds shape (54, 26, 40), TBR=5.97, FAA=-0.045 confirmed
- psd_array_welch help -- confirmed n_fft, fmin/fmax, output shape semantics

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- MNE 1.11.0 installed, all APIs verified on real data
- Architecture: HIGH -- full pipeline tested end-to-end on real subject
- Pitfalls: HIGH -- n_fft constraint and channel indexing confirmed by live execution

**Research date:** 2026-02-23
**Valid until:** 2026-03-23
