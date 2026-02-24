---
phase: 04-classification-output
plan: 01
subsystem: classification-output
tags: [svm-rbf, stratifiedgroupkfold, permutation-test, json-output]
dependency_graph:
  requires: [03-01]
  provides: [results-json]
  affects: [project-reporting]
tech_stack:
  added: [scikit-learn-pipeline, scikit-learn-svc, sklearn-permutation-test]
  patterns: [group-cv, label-encoding, metric-serialization]
key_files:
  created: [classifier.py, tests/test_classifier.py]
  modified: [main.py]
decisions:
  - "StandardScaler kept inside Pipeline to prevent CV leakage"
  - "MDD probability column selected via LabelEncoder class lookup"
metrics:
  duration: 20 min
  completed: 2026-02-23
---

# Phase 04 Plan 01: Classification and Output Summary

Implemented SVM(RBF) classification with StratifiedGroupKFold, 1000-permutation significance testing, and JSON result export.

## Tasks Completed

| Task | Name | Files |
|------|------|-------|
| 1 | Create classifier module (`classify`) | classifier.py |
| 2 | Extend pipeline entrypoint for metrics + JSON output | main.py |
| 3 | Add contract test for classifier outputs | tests/test_classifier.py |

## Verification Results

- `pytest -q tests/test_classifier.py` - passed (`1 passed`)
- `python -c "from classifier import classify; print('import ok')"` - passed
- `python -c "import ast, pathlib; ast.parse(pathlib.Path('main.py').read_text(encoding='utf-8')); print('syntax ok')"` - passed
- `python main.py` - passed
  - `Feature matrix: X.shape=(492, 262), y.shape=(492,)`
  - `AUC=0.768  ACC=0.762  SEN=0.984  SPE=0.395  p=1.0000`
  - `Saved results.json`
- `results.json` schema check - all required keys present:
  - `auc, accuracy, sensitivity, specificity, permutation_pvalue, n_permutations, n_splits`

## Decisions Made

- Kept stratification and subject leakage control via `StratifiedGroupKFold`.
- Preserved Windows-safe call pattern by running classification inside `if __name__ == "__main__":` path in `main.py`.

## Deviations from Plan

- Added `tests/test_classifier.py` for TDD/contract verification (plan only required runtime checks).

## Self-Check: PASSED
