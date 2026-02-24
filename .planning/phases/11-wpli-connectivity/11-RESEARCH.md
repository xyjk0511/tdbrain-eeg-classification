# Phase 11: wPLI Connectivity Extraction - Research

**Researched:** 2026-02-24
**Domain:** EEG functional connectivity (wPLI), mne-connectivity
**Confidence:** HIGH

## Summary

Phase 11 adds weighted Phase Lag Index (wPLI) connectivity features to the existing 496-dim spectral/Hjorth feature vector. wPLI measures phase synchronization between channel pairs while being robust to volume conduction — preferred over coherence/PLV for scalp EEG. Uses `mne-connectivity` 0.7.0 `spectral_connectivity_epochs` with `method='wpli'`, 3 bands (theta/alpha/beta), 325 channel pairs, ROI-averaged to 45 features per condition.

ROI averaging: 26 channels -> 5 regions (Frontal/Central/Temporal/Parietal/Occipital). "15 pairs" = C(5,2)+5 = 10 inter + 5 intra. 15 pairs x 3 bands = 45/condition. "~90 dim" = 2 conditions x 45 after fusion (Phase 12).

**Primary recommendation:** `spectral_connectivity_epochs(epochs, method='wpli', fmin=(4,8,13), fmax=(8,13,30), faverage=True)` — one call, ROI-average to 45 features, cache to `.npz`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CON-01 | Install mne-connectivity, extract wPLI, 3 bands x 325 pairs | `spectral_connectivity_epochs(method='wpli', fmin=(4,8,13), fmax=(8,13,30), faverage=True)` |
| CON-02 | ROI averaging (5 ROI -> 15 pairs), ~90 dim | 15 pairs x 3 bands = 45/condition; 90 after EO+EC fusion |
| CON-03 | Cache connectivity to disk (.npz) | `cache_connectivity/{cond}/{sid}.npz` with `conn` key |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mne-connectivity | 0.7.0 | wPLI via `spectral_connectivity_epochs` | Official MNE ecosystem; accepts Epochs directly |
| mne | 1.11.0 (existing) | Epochs object | Already installed |
| numpy | 2.4.2 (existing) | ROI averaging | Already installed |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| mne-connectivity wPLI | Manual cross-spectra | 50+ lines error-prone DSP vs 1 call |
| `spectral_connectivity_epochs` | `spectral_connectivity_time` | `_time` is for time-resolved; overkill here |
| `wpli` | `wpli2_debiased` | Debiased reduces bias for small N; standard `wpli` sufficient |

**Installation:** `pip install mne-connectivity`

## Architecture Patterns

### Project Structure

```
tdbrain/
├── connectivity_extractor.py   # NEW — extract_connectivity(epochs) -> (45,)
├── config.py                   # MODIFIED — add ROI_GROUPS, CONN_BANDS, CONN_CACHE_DIR
├── feature_extractor.py        # UNCHANGED
├── main.py                     # MODIFIED (Phase 12, not this phase)
└── cache_connectivity/         # NEW — .npz cache
    ├── EO/{sid}.npz
    └── EC/{sid}.npz
```

### Pattern 1: Separate Connectivity Extractor

New `connectivity_extractor.py` with `extract_connectivity(epochs) -> np.ndarray` returning 45-dim vector. Separate from `feature_extractor.py` because connectivity uses cross-spectral density across channel pairs (expensive, all-pairs) vs per-channel PSD, and needs different caching strategy.

### Pattern 2: Multi-Band Single Call

Pass `fmin=(4,8,13)` and `fmax=(8,13,30)` as tuples with `faverage=True`. Result shape: `(325, 3)`.

```python
from mne_connectivity import spectral_connectivity_epochs

con = spectral_connectivity_epochs(
    epochs, method='wpli',
    fmin=(4, 8, 13), fmax=(8, 13, 30),
    faverage=True, verbose=False,
)
wpli = con.get_data()  # (325, 3)
```

### Pattern 3: ROI Averaging via Index Mapping

Pre-compute mapping from 325 channel pairs to ROI pairs. Average wPLI within each ROI pair.

```python
from itertools import combinations_with_replacement

ROI_GROUPS = {
    'Frontal':  ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4'],
    'Central':  ['C3','Cz','C4'],
    'Temporal': ['T7','T8'],
    'Parietal': ['CP3','CPz','CP4','P7','P3','Pz','P4','P8'],
    'Occipital':['O1','Oz','O2'],
}

roi_names = list(ROI_GROUPS.keys())
roi_pairs = list(combinations_with_replacement(roi_names, 2))  # 15 pairs
```

### Pattern 4: Disk Cache

```python
cache_path = CONN_CACHE_DIR / cond / f"{sid}.npz"
if cache_path.exists():
    return np.load(cache_path)['conn']
conn = extract_connectivity(epochs)
cache_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(cache_path, conn=conn)
```

### Anti-Patterns to Avoid

- **Computing wPLI per band in separate calls:** Recomputes CSD 3x. Use tuple `fmin/fmax`.
- **Storing full 325-pair connectivity as features:** 975 dims = dimensionality explosion. ROI averaging mandatory.
- **Putting connectivity inside feature_extractor.py:** Violates single-responsibility; caching needs subject ID.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| wPLI computation | Manual cross-spectral + phase weighting | `spectral_connectivity_epochs(method='wpli')` | Handles windowing, tapering, averaging |
| Channel pair indexing | Nested loop over 26x26 | `np.triu_indices(26, k=1)` | Off-by-one errors common |
| Multi-band connectivity | 3 separate calls | Single call with tuple `fmin/fmax` | CSD computed once |

## Common Pitfalls

### Pitfall 1: Volume Conduction Inflating Connectivity

Using coherence/PLV instead of wPLI. Nearby channels share volume-conducted signal.

**How to avoid:** Use `method='wpli'` exclusively. wPLI suppresses zero-lag components.

**Warning signs:** Connectivity near 1.0 for adjacent channels.

### Pitfall 2: Wrong Channel Pair Indexing

`con.get_data()` returns flat `(325, n_freqs)`. Ordering follows `np.triu_indices(n_ch, k=1)` upper triangle row-major.

**How to avoid:** Use `np.triu_indices(26, k=1)`. Channel order matches `EEG_CHANNELS` in config.py.

**Warning signs:** Frontal-Occipital higher than Frontal-Central.

### Pitfall 3: Epoch Count Too Low for Stable wPLI

wPLI averages imaginary cross-spectrum across epochs. With <10 epochs, estimate is noisy.

**How to avoid:** Assert `len(epochs) >= 10` before computing.

### Pitfall 4: Forgetting faverage=True

Without `faverage=True`, result is per-frequency-bin, not band-averaged. Shape `(325, n_freqs)` instead of `(325, 3)`.

**How to avoid:** Always pass `faverage=True` with tuple `fmin/fmax`.

### Pitfall 5: Intra-ROI wPLI for Small ROIs

Temporal ROI has only 2 channels (T7, T8) — single pair, higher variance than Frontal (45 pairs).

**How to avoid:** Accept as known limitation. SelectKBest will down-weight if noisy.

## Code Examples

### Complete wPLI Extraction with ROI Averaging

```python
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
from itertools import combinations_with_replacement

def extract_connectivity(epochs) -> np.ndarray:
    """Return (45,): 15 ROI pairs x 3 bands wPLI."""
    con = spectral_connectivity_epochs(
        epochs, method='wpli',
        fmin=(4, 8, 13), fmax=(8, 13, 30),
        faverage=True, verbose=False,
    )
    wpli = con.get_data()  # (325, 3)

    ch_names = epochs.ch_names
    row_idx, col_idx = np.triu_indices(len(ch_names), k=1)

    # Map each pair to ROI pair, average within
    # (see Pattern 3 for ROI_GROUPS definition)
    return np.concatenate(features)  # (45,)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `mne.connectivity.spectral_connectivity` | `mne_connectivity.spectral_connectivity_epochs` | 2021 | Separate install; cleaner API |
| PLI (unweighted) | wPLI (weighted) | Vinck 2011 | Volume conduction robustness |
| All-pairs features | ROI-averaged | Common practice | Reduces dimensionality |

**Deprecated:** `mne.connectivity.spectral_connectivity` removed in MNE >= 1.0.

## Open Questions

1. **Intra-ROI pairs: include or exclude?**
   - Requirement says "15 pairs" = 10 inter + 5 intra, implying included
   - Recommendation: Include. wPLI suppresses zero-lag; SelectKBest handles noise.

2. **Feature dimension: 45 vs 90**
   - 15 pairs x 3 bands = 45 per condition. "~90 dim" = 2 conditions after fusion.
   - Recommendation: 45 per condition. Phase 12 (CON-04) fuses to 90.

## Sources

### Primary (HIGH confidence)
- mne_connectivity.spectral_connectivity_epochs 0.7.0 docs
- Comparing PLI/wPLI/dPLI — MNE-Connectivity 0.7.0 examples
- mne-connectivity PyPI — v0.7.0, Python >=3.9
- Direct codebase inspection: config.py, feature_extractor.py, preprocessor.py, main.py

### Secondary (MEDIUM confidence)
- wPLI and wSMI — Nature Sci Rep 2019
- PLI/wPLI reproducibility — PLOS ONE 2014

### Tertiary (LOW confidence)
- ROI grouping: derived from 10-20 anatomy, not validated for this exact 26-ch montage

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — mne-connectivity 0.7.0 API verified via official docs
- Architecture: HIGH — follows existing project patterns
- Pitfalls: HIGH — volume conduction, pair indexing well-documented
- ROI grouping: MEDIUM — standard anatomy, not validated for this exact montage

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
