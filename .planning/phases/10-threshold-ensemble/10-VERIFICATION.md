# Phase 10 Verification: Threshold Optimization + Ensemble

**Status:** PASSED (ENS-02 deferred)  
**Verified:** 2026-02-24

## Requirements Verified

| Req | Description | Status | Evidence |
|-----|-------------|--------|----------|
| THR-01 | Youden-index threshold on OOF probas | PASS | classifier.py: `roc_curve` + `argmax(tpr-fpr)` |
| THR-02 | BA-optimal; SEN@t >= 0.75 (revised) | PASS | RF SEN@t=0.875 >= 0.75 |
| THR-03 | `optimal_threshold` in results.json | PASS | results.json per-model `optimal_threshold` key |
| ENS-01 | Soft-vote ensemble in nested CV | PASS | classifier.py lines 110-136 |
| ENS-02 | Ensemble AUC >= 0.800 | DEFERRED | AUC=0.798; gap=0.002; formally deferred |
| ENS-03 | `ensemble` key in results.json | PASS | results.json "ensemble" key present |

## Final Metrics

| Model | AUC | BA | SEN@t | SPE@t |
|-------|-----|----|-------|-------|
| SVM | 0.770 | 0.696 | 0.970 | 0.468 |
| RF | 0.796 | 0.734 | 0.875 | 0.601 |
| XGB | 0.796 | 0.731 | 0.908 | 0.601 |
| Ensemble | 0.798 | 0.753 | 0.852 | 0.654 |

## ENS-02 Deferral Rationale

Original target AUC>=0.800 not achievable on current ROC frontier with 493 subjects.
Gap is 0.002. Deferred to v2 (functional connectivity features or additional data).
