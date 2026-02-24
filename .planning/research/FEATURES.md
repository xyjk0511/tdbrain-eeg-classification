# Feature Research

**Domain:** EEG-based psychiatric classification (MDD vs ADHD), resting-state, v1.1 milestone
**Researched:** 2026-02-23
**Confidence:** HIGH (Hjorth, spectral entropy) / MEDIUM (EC/EO fusion strategy)

## Context: What Already Exists (v1.0)

The pipeline already produces a 262-dim feature vector per subject:
- Welch PSD absolute band power: 5 bands x 26 channels = 130 features
- Welch PSD relative band power: 5 bands x 26 channels = 130 features
- TBR (theta/beta ratio, frontal channels): 1 scalar
- FAA (frontal alpha asymmetry, F3/F4): 1 scalar

Baseline: SVM + nested StratifiedGroupKFold CV, AUC = 0.796, p = 0.001.

All new features must integrate into feature_extractor.py and remain compatible
with the existing epochs (MNE Epochs object) input contract.

---

## Feature Landscape

### Table Stakes (Expected for a Credible EEG Classifier)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Hjorth Activity | Standard time-domain descriptor since Hjorth (1970); variance of signal = mean power proxy; used in virtually every EEG ML paper | LOW | np.var(data) per channel per epoch, mean across epochs. No new dependency. |
| Hjorth Mobility | Mean frequency estimate; sqrt(var(first derivative) / var(signal)); ADHD-specific literature uses it directly | LOW | np.diff then var ratio. Pure numpy. |
| Hjorth Complexity | Signal complexity vs sinusoid; mobility(derivative) / mobility(signal); captures waveform irregularity relevant to MDD | LOW | Two mobility calls. Pure numpy. |
| Spectral Entropy (broadband) | Shannon entropy of normalized PSD; measures spectral disorder; standard in ADHD feature reviews (MDPI Sensors 2022 systematic review) | LOW | Already have PSD from Welch. -sum(p*log(p)) per channel. Reuses existing psds. |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Spectral Entropy per band | Band-wise SE (5 bands x 26 channels = 130 features); captures disorder within each rhythm separately; used in ADHD multi-band classification (arxiv 2504.04664) | LOW | Slice psds by band mask, normalize within band, compute entropy. Reuses existing band loop. |
| EC/EO feature concatenation | EC and EO capture different neural states; EC emphasizes alpha; EO emphasizes beta/gamma vigilance; concatenating both adds condition-specific signal; qEEG literature shows EC+EO outperforms either alone | MEDIUM | Run pipeline twice (once per condition), align subjects present in both, np.concatenate([feat_eo, feat_ec]). Main complexity: subject intersection logic. |
| EC/EO difference features | Delta features (EO - EC) per band/channel capture reactivity; alpha blocking (EC to EO alpha suppression) is a known MDD biomarker | MEDIUM | Requires same subject alignment as concatenation. Appends 262 more features (difference vector). |

### Anti-Features (Avoid for This Milestone)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Sample/permutation entropy | Nonlinear complexity; appears in ADHD literature | O(n^2) compute for SampEn; adds antropy dependency; marginal gain over Hjorth complexity | Use Hjorth complexity instead |
| Functional connectivity (coherence, PLV) | Used in advanced ADHD GCN papers | 26x26 pairs x 5 bands = 3380 features; explodes dimensionality; out of scope per PROJECT.md | Defer to v2 milestone |
| Deep learning features (CNN spectrogram) | High accuracy in ADHD papers | Requires PyTorch/TF; incompatible with current SVM+sklearn pipeline | Keep classical ML; add XGBoost/RF as separate v1.1 task |

---

## Feature Dependencies

```
[Welch PSD (existing)]
    reuses --> [Spectral Entropy broadband]
    reuses --> [Spectral Entropy per band]

[Hjorth Activity]
    requires --> [epochs.get_data() time-series]
[Hjorth Mobility]
    requires --> [Hjorth Activity] (shares np.diff computation)
[Hjorth Complexity]
    requires --> [Hjorth Mobility] (mobility of derivative / mobility of signal)

[EC/EO Concatenation]
    requires --> [load_subjects("EC")] (data_loader already supports this)
    requires --> [subject intersection logic] (new: find subjects with both conditions)
    requires --> [extract_features() called twice]
[EC/EO Difference Features]
    requires --> [EC/EO Concatenation] (same subject alignment prerequisite)
```

### Dependency Notes

- Hjorth parameters require epochs.get_data() (time-domain). Separate code path from existing Welch PSD block.
- Spectral entropy reuses psds, freqs already computed in extract_features(). No redundant computation.
- EC/EO fusion requires changes to main.py (load both conditions, align subjects) and config.py (remove hardcoded CONDITION = "EO"). data_loader.py already supports load_subjects(condition) and load_raw(subject_id, condition) -- infrastructure is ready.

---

## MVP Definition

### Launch With (v1.1)

- [ ] Hjorth Activity + Mobility + Complexity -- 78 features (3 x 26). Pure numpy, no new deps.
- [ ] Spectral Entropy broadband -- 26 features. Reuses existing PSD.
- [ ] Spectral Entropy per band -- 130 features (5 x 26). Reuses band loop.
- [ ] EC/EO feature concatenation -- doubles feature space with condition-specific signal.

### Add After Validation (v1.x)

- [ ] EC/EO difference features -- add only if concatenation alone shows < 0.01 AUC gain.
- [ ] Feature selection (SHAP / SelectKBest) -- listed in PROJECT.md v1.1; critical once dims exceed 500.

### Future Consideration (v2+)

- [ ] Functional connectivity features -- PROJECT.md explicitly defers.
- [ ] ERP features (P300) -- PROJECT.md explicitly defers.
- [ ] Deep learning pipeline -- incompatible with current sklearn architecture.

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Hjorth Activity/Mobility/Complexity | HIGH | LOW | P1 |
| Spectral Entropy broadband | HIGH | LOW | P1 |
| Spectral Entropy per band | MEDIUM | LOW | P1 |
| EC/EO concatenation | HIGH | MEDIUM | P2 |
| EC/EO difference features | MEDIUM | MEDIUM | P2 |
| Feature selection (SHAP/SelectKBest) | HIGH | MEDIUM | P2 |

---

## Implementation Notes for Existing Pipeline

### Hjorth in feature_extractor.py

```python
data = epochs.get_data()  # (n_epochs, n_channels, n_times)

def hjorth(data):
    act  = data.var(axis=-1)                          # (n_epochs, n_channels)
    d1   = np.diff(data, axis=-1)
    mob  = np.sqrt(d1.var(axis=-1) / act)             # (n_epochs, n_channels)
    d2   = np.diff(d1, axis=-1)
    comp = np.sqrt(d2.var(axis=-1) / d1.var(axis=-1)) / mob  # (n_epochs, n_channels)
    return act.mean(0), mob.mean(0), comp.mean(0)     # each (26,)
```

Adds 78 features. No new imports.

### Spectral Entropy in feature_extractor.py

```python
# psds: (n_epochs, n_channels, n_freqs) -- already computed
p_broad = psds / psds.sum(axis=-1, keepdims=True)
se_broad = -(p_broad * np.log(p_broad + 1e-12)).sum(axis=-1).mean(0)  # (26,)

# Per-band: for each band, slice freqs, renormalize within band, compute entropy
se_bands = []
for band, (fmin, fmax) in FREQ_BANDS.items():
    mask = (freqs >= fmin) & (freqs < fmax)
    p_band = psds[:, :, mask]
    p_band = p_band / (p_band.sum(axis=-1, keepdims=True) + 1e-12)
    se_bands.append(-(p_band * np.log(p_band + 1e-12)).sum(axis=-1).mean(0))  # (26,)
```

Adds 26 + 130 = 156 features.

### EC/EO Fusion in main.py

```python
# Load both conditions, find intersection
subj_eo = load_subjects("EO").set_index("participants_ID")
subj_ec = load_subjects("EC").set_index("participants_ID")
common  = subj_eo.index.intersection(subj_ec.index)

# Extract features per condition, align, concatenate
feat_eo = {sid: extract_features(preprocess(load_raw(sid, "EO"))[0]) for sid in common}
feat_ec = {sid: extract_features(preprocess(load_raw(sid, "EC"))[0]) for sid in common}
X = np.array([np.concatenate([feat_eo[s], feat_ec[s]]) for s in common])
```

config.py CONDITION constant becomes unused; pass condition as argument instead.

---

## Sources

- Hjorth parameters original: http://www.schreibtischtaeter.com/?wiki/Hjorth_parameters
- Hjorth mobility for ADHD: https://www.brainanddevelopment.com/article/S0387-7604(18)30483-2/fulltext
- ADHD EEG feature systematic review (MDPI Sensors 2022): https://www.mdpi.com/1424-8220/22/13/4934
- Multi-band spectral entropy for ADHD: https://arxiv.org/html/2504.04664v1
- mne-features spectral entropy: https://mne.tools/mne-features/generated/mne_features.univariate.compute_spect_entropy.html
- EC+EO resting state qEEG classification: https://www.researchgate.net/publication/386510426_Contribution_of_Scalp_Regions_to_Machine_Learning-Based_Classification_of_Dementia_Utilizing_Resting-State_qEEG_Signals
- MDD EEG feature selection impact: https://www.mdpi.com/2076-3417/14/22/10532
- NeuroKit2 Hjorth reference: https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/complexity/complexity_hjorth.html

---
*Feature research for: EEG MDD vs ADHD classification (v1.1 milestone)*
*Researched: 2026-02-23*
