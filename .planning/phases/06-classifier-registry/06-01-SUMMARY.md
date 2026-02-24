---
phase: 06-classifier-registry
plan: 01
subsystem: classification
tags: [sklearn, xgboost, feature-selection, nested-cv, permutation-test]

requires:
  - phase: 05-feature-extraction
    provides: 496-dim feature vectors (spectral + Hjorth + entropy)

provides:
  - classify() returning per-model results dict {svm, rf, xgb}
  - SelectKBest(k=50) inside Pipeline CV (no leakage)
  - N_FEATURES_SELECT constant in config

affects: [07-reporting, any downstream analysis using classify()]

tech-stack:
  added: [xgboost]
  patterns: [model registry dict, _selector() factory for DRY Pipeline steps]

key-files:
  created: []
  modified: [config.py, classifier.py]

key-decisions:
  - "SelectKBest inside Pipeline (not outside CV) — prevents feature selection leakage"
  - "XGBoost max_depth=[2,3,4] — default 6 overfits on ~300 training samples"
  - "_selector() factory function — avoids repeating SelectKBest(...) across 6 Pipeline definitions"
  - "FIXED_PIPES dict mirrors MODEL_REGISTRY structure — permutation test uses same selector step"

patterns-established:
  - "Model registry: local dict of (pipe, param_grid) tuples, loop over .items()"
  - "Selector step always named 'selector' in Pipeline for consistent param access"

requirements-completed: [CLF-01, CLF-02, CLF-03, CLF-04, CLF-05]

duration: 2min
completed: 2026-02-23
---

# Phase 6 Plan 1: Classifier Registry Summary

**Multi-model nested CV (SVM/RF/XGBoost) with SelectKBest(k=50) inside Pipeline, returning per-model {auc, accuracy, sensitivity, specificity, permutation_pvalue}**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T10:40:28Z
- **Completed:** 2026-02-23T10:42:20Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Model registry with SVM, RF, XGBoost — each Pipeline includes SelectKBest(k=50) as named "selector" step
- classify() returns 3-key dict; each sub-dict has auc, accuracy, sensitivity, specificity, permutation_pvalue, n_permutations, n_splits
- Smoke test passes: 60 samples, 100 features, 3-fold, 5 perms

## Task Commits

1. **Task 1: Add N_FEATURES_SELECT to config.py** - `a525016` (feat)
2. **Task 2: Model registry + per-model results** - `5f05003` (feat)

## Files Created/Modified

- `config.py` - Added N_FEATURES_SELECT = 50
- `classifier.py` - Replaced with model registry, SelectKBest in all pipelines, per-model results loop

## Decisions Made

- SelectKBest inside Pipeline (not outside CV) — prevents feature selection leakage
- XGBoost max_depth=[2,3,4] — default 6 overfits on ~300 training samples
- `_selector()` factory function — avoids repeating SelectKBest(...) across 6 Pipeline definitions
- FIXED_PIPES dict mirrors MODEL_REGISTRY structure — permutation test uses same selector step

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- classify() ready for downstream reporting/analysis
- All three models produce auc + permutation_pvalue for comparison

---
*Phase: 06-classifier-registry*
*Completed: 2026-02-23*
