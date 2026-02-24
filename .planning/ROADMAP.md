# Roadmap: TDBRAIN MDD vs ADHD 鉴别诊断

## Milestones

- ✅ **v1.0 EEG MDD/ADHD Classifier** — Phases 1-4 (shipped 2026-02-23)
- ✅ **v1.2 Data Quality + Class Balance** — Phase 9 (shipped 2026-02-23)
- ✅ **v1.3 Threshold + Ensemble** — Phase 10 (shipped 2026-02-24)
- 🔄 **v2.0 Stacking + Functional Connectivity** — Phases 11-14

## Phases

<details>
<summary>✅ v1.0 EEG MDD/ADHD Classifier (Phases 1-4) — SHIPPED 2026-02-23</summary>

- [x] Phase 1: Data Loading (1/1 plans) — completed 2026-02-23
- [x] Phase 2: Preprocessing (1/1 plans) — completed 2026-02-23
- [x] Phase 3: Feature Extraction (1/1 plans) — completed 2026-02-23
- [x] Phase 4: Classification & Output (1/1 plans) — completed 2026-02-23

</details>

<details>
<summary>✅ v1.1 (Phases 5-8) — SHIPPED 2026-02-23</summary>

- [x] Phase 5: New Features — Hjorth + spectral entropy
- [x] Phase 6: Classifier Registry — SVM/RF/XGBoost + SelectKBest
- [x] Phase 7: EC/EO Fusion + Integration
- [x] Phase 8: SHAP Post-hoc Analysis

</details>

<details>
<summary>✅ v1.2–v1.3 (Phases 9-10) — SHIPPED 2026-02-24</summary>

- [x] Phase 9: Data Quality + Class Balance — SMOTE + 150µV threshold
- [x] Phase 10: Threshold Optimization + Ensemble — Youden + soft-vote

</details>

## v2.0 Phases

- [x] Phase 11: wPLI Connectivity Extraction — 新建 connectivity_extractor.py，wPLI + ROI 平均 + 磁盘缓存 (completed 2026-02-24)
  - Requirements: CON-01, CON-02, CON-03
  - Files: connectivity_extractor.py (NEW), config.py (MODIFIED)
  - Risk: MEDIUM — mne-connectivity API, ROI mapping correctness
  - Goal: 每条件输出 45 维 wPLI 连接特征向量（15 ROI 对 × 3 频段），缓存到 .npz（EC+EO 融合后 90 维，见 Phase 13）
  - **Plans:** 1 plan
  - Plans:
    - [x] 11-01-PLAN.md — wPLI extraction + ROI averaging + disk caching

- [ ] Phase 12: Manual Stacking Ensemble — 手动 stacking 替换 soft-vote，可先用现有 992 维验证
  - Requirements: STK-01, STK-02, STK-03, STK-04
  - Files: classifier.py (MODIFIED), main.py (MODIFIED)
  - Risk: MEDIUM — OOF meta-feature 生成、groups 泄露防护
  - Goal: stacking AUC ≥ soft-vote AUC (0.798)，结果写入 results.json
  - **Plans:** 1 plan
  - Plans:
    - [ ] 12-01-PLAN.md — Stacking OOF loop + meta-learner + metrics + diagnostics

- [ ] Phase 13: Connectivity Integration + k Sweep — 连接特征拼接 + SelectKBest k 优化
  - Requirements: CON-04, TUN-01
  - Files: main.py (MODIFIED), main_diagnostic.py (MODIFIED)
  - Risk: LOW — 拼接逻辑简单，k sweep 复用现有 GridSearchCV
  - Goal: 992+180 维融合，最优 k 下 stacking AUC > 0.798

- [ ] Phase 14: Permutation Test Update — 置换检验覆盖完整 stacking pipeline
  - Requirements: TUN-02
  - Files: classifier.py (MODIFIED)
  - Risk: LOW — 逻辑清晰，主要是运行时间
  - Goal: stacking p-value < 0.01

## Dependencies

```
Phase 11 (connectivity) ──┐
                          ├──> Phase 13 (integration + k sweep) ──> Phase 14 (perm test)
Phase 12 (stacking)  ─────┘
```

Phase 11 和 Phase 12 互相独立，可并行开发。Phase 13 依赖两者完成。Phase 14 最后。

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
| 10. Threshold + Ensemble | v1.3 | 1/1 | Complete | 2026-02-24 |
| 11. wPLI Connectivity | v2.0 | 1/1 | Complete | 2026-02-24 |
| 12. Manual Stacking | v2.0 | 0/1 | Pending | — |
| 13. Integration + k Sweep | v2.0 | 0/? | Pending | — |
| 14. Permutation Test | v2.0 | 0/? | Pending | — |
