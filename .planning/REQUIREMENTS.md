# Requirements: TDBRAIN MDD vs ADHD — v2.0 Stacking + Functional Connectivity

**Defined:** 2026-02-24
**Core Value:** 用 stacking + 功能连接特征突破 AUC=0.798 瓶颈

## v2.0 Requirements

### Connectivity

- [ ] **CON-01**: 安装 mne-connectivity，提取 wPLI（非 coherence/PLV），theta/alpha/beta 3频段 × 325 对
- [ ] **CON-02**: ROI 平均（5 ROI → 15 对），每条件 ~90 维连接特征
- [ ] **CON-03**: 连接特征缓存到磁盘（.npz），避免重复计算
- [ ] **CON-04**: 连接特征拼接到现有 992 维，EC/EO 融合后 ~1172 维

### Stacking

- [ ] **STK-01**: 手动 stacking 循环（非 StackingClassifier），StratifiedGroupKFold 生成 OOF meta-features
- [ ] **STK-02**: LogisticRegression(C=1.0) 作为 meta-learner，passthrough=False
- [ ] **STK-03**: SMOTE 仅在 base model ImbPipeline 内部，stacking 层不做过采样
- [ ] **STK-04**: Stacking 结果以 `stacking` key 写入 results.json

### Tuning

- [ ] **TUN-01**: SelectKBest k sweep（k=50,80,100,150），选最优 k
- [ ] **TUN-02**: 置换检验覆盖完整 stacking pipeline（非单模型）

## Archived: v1.1–v1.3 (All Complete)

<details>
<summary>v1.1 Requirements (14/14 complete)</summary>

- [x] **FEAT-01**: Hjorth 参数 78维
- [x] **FEAT-02**: 宽带 spectral entropy 26维
- [x] **FEAT-03**: 分频段 spectral entropy 130维
- [x] **FEAT-04**: Hjorth µV² 验证
- [x] **CLF-01**: SVM/RF/XGB model registry
- [x] **CLF-02**: SelectKBest(k=50) in Pipeline
- [x] **CLF-03**: XGBoost max_depth=[2,3,4]
- [x] **CLF-04**: permutation test fixed_pipe 一致
- [x] **CLF-05**: classify() 返回独立结果 dict
- [x] **FUS-01**: EC 条件加载
- [x] **FUS-02**: EO/EC 交集
- [x] **FUS-03**: EO+EC 拼接
- [x] **OUT-01**: results.json 三模型对比
- [x] **OUT-02**: SHAP 特征重要性

</details>

<details>
<summary>v1.2 Requirements (3/3 complete)</summary>

- [x] **DQ-01**: epoch 阈值放宽至 150µV
- [x] **BAL-01**: SMOTE 过采样
- [x] **BAL-02/03**: SPE 提升验证

</details>

<details>
<summary>v1.3 Requirements (5/6 complete, 1 deferred)</summary>

- [x] **THR-01**: Youden index 最优阈值
- [x] **THR-02**: BA 最大化（revised）
- [x] **THR-03**: optimal_threshold 输出
- [x] **ENS-01**: 软投票集成
- [ ] **ENS-02**: AUC ≥ 0.800 → deferred to v2.0
- [x] **ENS-03**: ensemble key in results.json

</details>

## Out of Scope

| Feature | Reason |
|---------|--------|
| EC/EO 差值连接特征 | 信噪比低，仅当拼接方案有效后考虑 |
| All-pairs 连接（3250维） | 维度灾难，必须 ROI 平均 |
| 深度学习 meta-learner | 仅 3 输入，过拟合风险 |
| ERP 特征（P300） | 留后续 milestone |
| 深度学习（EEGNet/CNN） | 与 sklearn 架构不兼容 |
| 实时推断/临床部署 | 纯研究验证 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FEAT-01~04 | Phase 5 | Complete |
| CLF-01~05 | Phase 6 | Complete |
| FUS-01~03, OUT-01 | Phase 7 | Complete |
| OUT-02 | Phase 8 | Complete |
| THR-01~03, ENS-01/03 | Phase 10 | Complete |
| ENS-02 | Phase 10 | Deferred |
| CON-01~04 | TBD | Pending |
| STK-01~04 | TBD | Pending |
| TUN-01~02 | TBD | Pending |

---
*Requirements defined: 2026-02-24*
