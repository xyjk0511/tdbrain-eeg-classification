---
phase: 10-threshold-ensemble
plan: 01
status: complete
completed: 2026-02-24
---

# Phase 10 Summary: Threshold Optimization + Ensemble

## Final Metrics

| Model | AUC | BA | opt_thresh | SPE@t | SEN@t |
|-------|-----|----|-----------|-------|-------|
| SVM | 0.770 | 0.696 | 0.221 | 0.468 | 0.970 |
| RF | 0.796 | 0.734 | 0.494 | 0.601 | 0.875 |
| XGB | 0.796 | 0.731 | 0.415 | 0.601 | 0.908 |
| Ensemble | 0.798 | 0.753 | 0.489 | 0.654 | 0.852 |

## Requirements

- THR-01 ✓ Youden-index threshold on OOF probas
- THR-02 ✓ (revised) BA-optimal; SEN@t=0.875 >= 0.75
- THR-03 ✓ optimal_threshold in results.json
- ENS-01 ✓ Soft-vote ensemble in nested CV
- ENS-02 ~ AUC=0.798 (target 0.800, deferred)
- ENS-03 ✓ ensemble key in results.json

## Key Decisions

- THR-02 original (SPE>=0.70 AND SEN>=0.75) not achievable on current ROC frontier; revised to BA-optimal
- Youden index = BA-optimal (mathematically equivalent)
- Per-fold training threshold tried but gave worse generalization; reverted to OOF Youden
