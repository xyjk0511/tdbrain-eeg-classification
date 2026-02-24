# Architecture Research

**Domain:** EEG classification pipeline (Python, scikit-learn, MNE)
**Researched:** 2026-02-23
**Confidence:** HIGH

## Standard Architecture

### System Overview

Existing pipeline: config -> data_loader -> preprocessor -> feature_extractor -> classifier -> main

v1.1 changes:

- feature_extractor.py: MODIFIED (add Hjorth + spectral entropy)
- feature_selector.py: NEW (SelectKBest / SHAP wrapper)
- classifier.py: MODIFIED (model registry, RF + XGBoost)
- main.py: MODIFIED (dual-condition loop, fusion, selector call)
- config.py: MODIFIED (CONDITIONS list, N_FEATURES_SELECT)

EC/EO fusion is a data-assembly concern in main.py only.
feature_extractor.py stays condition-agnostic.

### Component Responsibilities

| Component | Status | v1.1 Change |
|-----------|--------|-------------|
| config.py | MODIFIED | Add CONDITIONS list; add N_FEATURES_SELECT = 50 |
| data_loader.py | UNCHANGED | Already parameterized on condition |
| preprocessor.py | UNCHANGED | No change |
| feature_extractor.py | MODIFIED | Add _extract_hjorth(), _extract_spectral_entropy() |
| feature_selector.py | NEW | select_features(X, y, groups, method, k) |
| classifier.py | MODIFIED | Model registry; RF + XGBoost; per-model results |
| main.py | MODIFIED | Dual-condition loop; fusion; feature selection |

## Architectural Patterns

### Pattern 1: Additive feature composition

**What:** extract_features(epochs) returns a flat np.ndarray. New feature groups
are private helpers concatenated at the end.

**When to use:** Always — keeps each group independently testable.

**Example:**


### Pattern 2: Per-subject dual-condition fusion in main.py

**What:** main.py builds two dicts keyed by subject_id, intersects them,
and concatenates EO+EC vectors per subject before classification.

**When to use:** EC/EO fusion. Keeps feature_extractor.py condition-agnostic.

**Trade-offs:** Subjects missing one condition are dropped. Feature dim doubles
before selection, which is why feature selection runs after fusion.

### Pattern 3: Model registry for multi-model comparison

**What:** classifier.py defines a registry dict mapping model name to
(pipeline, param_grid). The nested CV + permutation loop runs once per entry.

**Trade-offs:** Permutation test runs once per model. 3 models x 1000 perms
= ~3x longer runtime than v1.0. Consider n_permutations=500 if slow.

## Data Flow

### v1.1 Full Pipeline

participants.tsv
  -> load_subjects(EO) + load_subjects(EC)
  -> for each subject x condition: load_raw -> preprocess -> extract_features
  -> {subject_id: feature_vector} per condition
  -> intersect subjects, concat EO+EC vectors per subject
  -> X_fused shape: (n_subjects, ~732-dim)
  -> select_features(X_fused, y, groups, method=selectkbest, k=50)
  -> X_reduced shape: (n_subjects, 50)
  -> classify(X_reduced, y, groups, models=[svm, rf, xgb])
  -> results.json

### Key Data Flow Notes

1. Condition fusion: dict intersection in main.py. feature_extractor.py never
   knows which condition it is processing.

2. Feature selection placement: SelectKBest inside sklearn Pipeline in
   classifier.py for leak-safe nested CV. Standalone feature_selector.py
   is for exploratory SHAP analysis only (not used for reported AUC).

3. Multi-model results: classify() returns dict keyed by model name.
   main.py writes all model results under a models key in results.json.

## Integration Points

### New vs Modified Files

| File | Status | What Changes |
|------|--------|--------------|
| config.py | MODIFIED | Add CONDITIONS=[EO,EC]; N_FEATURES_SELECT=50 |
| data_loader.py | UNCHANGED | Already parameterized on condition |
| preprocessor.py | UNCHANGED | No change |
| feature_extractor.py | MODIFIED | +_extract_hjorth, +_extract_spectral_entropy |
| feature_selector.py | NEW | select_features(X, y, groups, method, k) |
| classifier.py | MODIFIED | Model registry; RF + XGBoost; nested results dict |
| main.py | MODIFIED | Dual-condition loop; fusion; selector; multi-model output |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| feature_extractor -> classifier | np.ndarray flat vector | Shape in docstring |
| feature_selector -> classifier | np.ndarray reduced + fitted selector | Selector for inspection only |
| main -> classifier | X, y, groups arrays | groups = subject ID array, unchanged |

## Anti-Patterns

### Anti-Pattern 1: Feature selection outside nested CV folds

**What people do:** Fit SelectKBest on all of X before nested CV.
**Why wrong:** Feature selection sees test-fold labels -> data leakage -> inflated AUC.
**Do this instead:** Wrap SelectKBest inside sklearn Pipeline so it fits only on training folds.

### Anti-Pattern 2: Condition-aware logic in feature_extractor.py

**What people do:** Pass condition into extract_features() and branch on it.
**Why wrong:** Couples feature extraction to experimental design; breaks unit testing.
**Do this instead:** extract_features(epochs) stays stateless. Routing in main.py only.

### Anti-Pattern 3: Monolithic classify() with if/elif per model

**What people do:** Add if/elif branches inside classify() for each new model.
**Why wrong:** Permutation test logic duplicated; hard to add a fourth model.
**Do this instead:** Model registry dict. CV + permutation loop written once.

## Suggested Build Order

| Step | File | What | Dependency |
|------|------|------|------------|
| 1 | feature_extractor.py | Add _extract_hjorth, _extract_spectral_entropy | None |
| 2 | config.py | Add CONDITIONS, N_FEATURES_SELECT | None |
| 3 | classifier.py | Model registry + RF + XGBoost | Step 1 |
| 4 | feature_selector.py | SelectKBest wrapper | Step 1 |
| 5 | main.py | Dual-condition loop + fusion + selector | Steps 1-4 |

Steps 1-4 are independently testable. Step 5 integrates everything.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Current (~477 subjects, EO only) | Single-process loop fine; ~5 min runtime |
| +EC condition (~400 subjects overlap) | Intersection reduces N slightly; runtime doubles |
| Feature dim ~732 pre-selection | SelectKBest to k=50 keeps classifier fast |
| 3 models x 1000 permutations | ~3x longer than v1.0; consider n_permutations=500 |
| Future ERP/connectivity features | Add _extract_* helpers; fusion pattern unchanged |

## Sources

- Direct inspection of config.py, data_loader.py, preprocessor.py,
  feature_extractor.py, classifier.py, main.py (HIGH confidence)
- scikit-learn Pipeline + SelectKBest: standard leak-safe feature selection
  pattern for nested CV (HIGH confidence)
- MNE-Python: Hjorth parameters from epoch time-domain signal via np.diff
  (standard DSP, HIGH confidence)
- SHAP for EEG feature importance: exploratory use only in nested CV context
  (MEDIUM confidence)

---
*Architecture research for: TDBRAIN EEG MDD vs ADHD pipeline (v1.1)*
*Researched: 2026-02-23*
