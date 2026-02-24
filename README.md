# TDBRAIN: EEG-Based MDD vs ADHD Classification

Resting-state EEG classifier distinguishing Major Depressive Disorder (MDD) from Attention Deficit Hyperactivity Disorder (ADHD) using the [TDBRAIN dataset](https://brainclinics.com/resources/).

## Results

| Model | AUC | Accuracy | Sensitivity | Specificity | p-value |
|-------|-----|----------|-------------|-------------|---------|
| SVM | 0.770 | 0.720 | 0.797 | 0.596 | < 0.001 |
| Random Forest | 0.796 | 0.765 | 0.862 | 0.606 | < 0.001 |
| XGBoost | 0.796 | 0.753 | 0.823 | 0.638 | < 0.001 |
| **Ensemble (soft-vote)** | **0.798** | — | — | — | — |

- 493 subjects (305 MDD, 188 ADHD), dual-condition (eyes-open + eyes-closed)
- 5-fold StratifiedGroupKFold with strict subject isolation
- Permutation test: 1000 permutations, all models p < 0.001

## Features (992 dimensions)

Per condition (EO/EC, 496 each), concatenated:

| Feature | Dimensions | Source |
|---------|-----------|--------|
| Absolute band power | 130 | 5 bands x 26 channels |
| Relative band power | 130 | 5 bands x 26 channels |
| TBR + FAA | 2 | Theta/Beta ratio, Frontal Alpha Asymmetry |
| Hjorth parameters | 78 | Activity/Mobility/Complexity x 26 channels |
| Broadband spectral entropy | 26 | 26 channels |
| Per-band spectral entropy | 130 | 5 bands x 26 channels |

### Top SHAP Features (MDD vs ADHD)

1. `EO_hjorth_comp_T3` — Hjorth Complexity at left temporal
2. `EO_abs_delta_T3` — Delta power at left temporal
3. `EC_abs_delta_T4` — Delta power at right temporal (eyes-closed)
4. `EO_abs_delta_C3` — Delta power at left central
5. `EC_hjorth_mob_T5` — Hjorth Mobility at left posterior temporal

Temporal and delta-band features dominate, consistent with literature on MDD/ADHD EEG differences.

## Pipeline

```
Raw EEG (.bdf)
  → Average reference + 1-45 Hz bandpass
  → 2s epochs + 150µV artifact rejection
  → Feature extraction (496 dims per condition)
  → EO + EC concatenation (992 dims)
  → ImbPipeline: StandardScaler → SMOTE → SelectKBest(k=50) → Classifier
  → 5-fold nested CV (outer: evaluation, inner: hyperparameter tuning)
  → Youden-index optimal threshold
  → Permutation test (1000 iterations)
```

## Project Structure

```
config.py                  # Dataset paths, EEG channels, frequency bands, constants
data_loader.py             # Load TDBRAIN participants and raw EEG files
preprocessor.py            # Re-reference, filter, epoch, artifact rejection
feature_extractor.py       # PSD, Hjorth, spectral entropy extraction
classifier.py              # Model registry, nested CV, ensemble, permutation test
connectivity_extractor.py  # wPLI functional connectivity (Phase 11, for future use)
main.py                    # Full pipeline: load → preprocess → extract → classify
results.json               # Classification results
shap_summary.json          # SHAP feature importance (top 20)
```

## Setup

### Requirements

- Python 3.11+
- TDBRAIN dataset (BDF format)

### Install

```bash
pip install mne numpy scipy scikit-learn xgboost imbalanced-learn shap antropy mne-connectivity
```

### Configure

Edit `config.py` to set your dataset path:

```python
DATASET_ROOT = Path("/path/to/tdbrain_dataset")
```

### Run

```bash
python main.py
```

Output: `results.json` + `shap_summary.json`

## Methodology Notes

- **No data leakage**: Feature selection (SelectKBest) and oversampling (SMOTE) are inside the CV pipeline, never applied to the full dataset before splitting
- **Subject isolation**: StratifiedGroupKFold ensures no subject appears in both train and test folds
- **SMOTE placement**: Inside ImbPipeline, applied only to training folds
- **Permutation test**: Group-level label permutation preserving subject structure
- **Threshold optimization**: Youden index (max sensitivity + specificity - 1) on OOF predictions

## Dataset

[TDBRAIN](https://brainclinics.com/resources/) — Two Decades of BRAIN data. Resting-state EEG recordings from psychiatric patients. Access requires registration.

## License

MIT
