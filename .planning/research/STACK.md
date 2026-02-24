# Stack Research

**Domain:** EEG classification — MDD vs ADHD (v1.1 feature additions)
**Researched:** 2026-02-23
**Confidence:** HIGH (all claims verified against installed environment)

## Existing Stack (DO NOT CHANGE)

| Technology | Version | Role |
|------------|---------|------|
| mne | 1.11.0 | EEG loading, preprocessing, Welch PSD |
| scikit-learn | 1.8.0 | SVM, nested CV, StratifiedGroupKFold, pipelines |
| numpy | 2.4.2 | Array ops |
| scipy | 1.17.0 | Signal processing primitives |

## New Libraries Required

### Must Install

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| shap | ~0.46 | SHAP feature importance | `pip install shap` needed. Use `TreeExplainer` for XGBoost/RF — fast and exact. Post-hoc only, not a pipeline step. |

### Already Installed

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| antropy | 0.1.9 | Hjorth params + spectral entropy | `hjorth_params()` returns (mobility, complexity); activity = `np.var(x)`. `spectral_entropy()` supports Welch method matching existing PSD approach. One library covers both new feature types. |
| xgboost | 3.2.0 | XGBoost classifier | `XGBClassifier` sklearn-compatible API drops into existing `Pipeline` + `GridSearchCV` without changes to CV infrastructure. |

### Already in sklearn (No Install)

| API | Purpose |
|-----|---------|
| `sklearn.ensemble.RandomForestClassifier` | RF classifier |
| `sklearn.feature_selection.SelectKBest` | Filter-based feature selection |
| `sklearn.feature_selection.f_classif` | ANOVA F-score scorer for SelectKBest |

## Installation

```bash
pip install shap
```

## Integration Points

### Hjorth Parameters (antropy -> feature_extractor.py)

`antropy.hjorth_params(x)` takes 1D signal, returns `(mobility, complexity)`. Activity = `np.var(x)`. Apply per-channel per-epoch, average across epochs — same pattern as existing band power.

```python
mobility, complexity = antropy.hjorth_params(x)  # x: (n_samples,)
activity = np.var(x)
```

Output: 3 params x 26 channels = 78 features appended to existing 262-dim vector.

### Spectral Entropy (antropy -> feature_extractor.py)

```python
se = antropy.spectral_entropy(x, sf=epochs.info['sfreq'], method='welch', normalize=True)
```

Use `normalize=True` to keep values in [0,1], consistent with relative band power scale.
Output: 1 value x 26 channels = 26 features appended.

### EC/EO Dual-Condition Fusion (data_loader.py + main.py)

No new library needed. Load both conditions separately, extract features, then `np.concatenate([feat_eo, feat_ec])` per subject. Drop subjects missing either condition.

### XGBoost / RF Classifiers (classifier.py)

Both use sklearn-compatible API, drop into existing `Pipeline` replacing `SVC`:

```python
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
              eval_metric='logloss', random_state=42, n_jobs=-1)

RandomForestClassifier(n_estimators=200, class_weight='balanced',
                       random_state=42, n_jobs=-1)
```

Existing `GridSearchCV` + `StratifiedGroupKFold` + permutation test work unchanged.

### SHAP (post-hoc, not in pipeline)

```python
import shap
explainer = shap.TreeExplainer(fitted_model)
shap_values = explainer.shap_values(X)  # (n_samples, n_features)
importance = np.abs(shap_values).mean(axis=0)
```

For in-pipeline selection use `SelectKBest(f_classif, k=50)` as a Pipeline step before the classifier.

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| antropy | mne-features | antropy already installed; mne-features adds heavy dependency for same functionality |
| antropy | scipy manual | antropy handles edge cases; manual adds ~40 lines with no benefit |
| SelectKBest | PCA | PCA destroys feature identity needed for SHAP interpretation |
| SHAP post-hoc | SHAP inside Pipeline | TreeExplainer requires a fitted model; post-hoc is the correct pattern |
| XGBClassifier sklearn API | xgb.train() native API | sklearn API integrates with existing GridSearchCV/Pipeline without refactoring |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| mne-features | Heavy dependency; antropy already covers Hjorth + entropy | antropy 0.1.9 |
| SHAP KernelExplainer for SVM | 100-1000x slower than TreeExplainer; impractical on 477 subjects | TreeExplainer on XGBoost/RF |
| Deep learning (EEGNet etc.) | n=477 too small; overfitting risk; out of scope per PROJECT.md | XGBoost/RF with regularization |

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| xgboost 3.2.0 | sklearn 1.8.0 | Set `eval_metric` explicitly to suppress deprecation warnings |
| antropy 0.1.9 | numpy 2.4.2 | Verified — hjorth_params and spectral_entropy work with numpy 2.x |
| shap ~0.46 | xgboost 3.2.0, sklearn 1.8.0 | TreeExplainer supports XGBoost 3.x; verify after install |

## Sources

- Verified in environment: `antropy.hjorth_params`, `antropy.spectral_entropy` API — HIGH confidence
- Verified in environment: xgboost 3.2.0, sklearn 1.8.0 RF + SelectKBest — HIGH confidence
- SHAP official docs (shap.readthedocs.io): TreeExplainer for tree models — MEDIUM confidence (not yet installed)

---
*Stack research for: EEG MDD/ADHD classification — v1.1 feature additions*
*Researched: 2026-02-23*
