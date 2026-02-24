# Roadmap: TDBRAIN MDD vs ADHD 鉴别诊断

## Milestones

- ✅ **v1.0 EEG MDD/ADHD Classifier** — Phases 1-4 (shipped 2026-02-23)
- ✅ **v1.2 Data Quality + Class Balance** — Phase 9 (shipped 2026-02-23)

## Phases

<details>
<summary>✅ v1.0 EEG MDD/ADHD Classifier (Phases 1-4) — SHIPPED 2026-02-23</summary>

- [x] Phase 1: Data Loading (1/1 plans) — completed 2026-02-23
- [x] Phase 2: Preprocessing (1/1 plans) — completed 2026-02-23
- [x] Phase 3: Feature Extraction (1/1 plans) — completed 2026-02-23
- [x] Phase 4: Classification & Output (1/1 plans) — completed 2026-02-23

</details>

## v1.1 Phases

- [ ] Phase 5: New Features — Hjorth + spectral entropy (feature_extractor.py)
- [x] Phase 6: Classifier Registry — SVM/RF/XGBoost + SelectKBest (classifier.py)
  - Plans: 1
  - [ ] 06-01-PLAN.md — Add model registry + SelectKBest pipeline + per-model results dict
- [x] Phase 7: EC/EO Fusion + Integration (main.py)
  - Plans: 1
  - [x] 07-01-PLAN.md — EC/EO fusion: intersect subjects, concatenate features (~732-dim), multi-model CV, results.json
- [x] Phase 8: SHAP Post-hoc Analysis (main.py)
  - Plans: 1
  - [x] 08-01-PLAN.md — SHAP TreeExplainer on final RF, top-20 features → shap_summary.json

## v1.2 Phases

- [x] Phase 9: Data Quality + Class Balance — 放宽 epoch 拒绝阈值 + SMOTE 过采样 (preprocessor.py, classifier.py)
  - Plans: 1
  - [x] 09-01-PLAN.md — Relax REJECT_THRESHOLD to 150uV + SMOTE inside CV pipeline

## v1.3 Phases

- [ ] Phase 10: Threshold Optimization + Ensemble — 最优分类阈值 + SVM/RF/XGB 软投票集成 (classifier.py)
  - Plans: 1
  - [ ] 10-01-PLAN.md — Youden-index threshold per model + soft-vote ensemble

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Data Loading | v1.0 | 1/1 | Complete | 2026-02-23 |
| 2. Preprocessing | v1.0 | 1/1 | Complete | 2026-02-23 |
| 3. Feature Extraction | v1.0 | 1/1 | Complete | 2026-02-23 |
| 4. Classification & Output | v1.0 | 1/1 | Complete | 2026-02-23 |
| 5. New Features | v1.1 | 1/1 | Complete | 2026-02-23 |
| 6. Classifier Registry | v1.1 | 1/1 | Complete | 2026-02-23 |
| 7. EC/EO Fusion | v1.1 | 1/1 | Complete | 2026-02-23 |
| 8. SHAP Analysis | v1.1 | 1/1 | Complete | 2026-02-23 |
| 9. Data Quality + Class Balance | v1.2 | 1/1 | Complete | 2026-02-23 |
| 10. Threshold + Ensemble | v1.3 | 0/1 | Pending | — |
