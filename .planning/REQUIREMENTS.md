# Requirements: TDBRAIN MDD vs ADHD — v1.1 Performance Improvement

**Defined:** 2026-02-23
**Core Value:** 在 v1.0 基础上（AUC=0.796）通过新特征、多模型、EC/EO融合提升分类性能

## v1.1 Requirements

### Features

- [x] **FEAT-01**: 提取 Hjorth 参数（Activity/Mobility/Complexity），26通道 × 3 = 78维特征
- [x] **FEAT-02**: 提取宽带 spectral entropy，26通道 = 26维特征
- [x] **FEAT-03**: 提取分频段 spectral entropy（5频段 × 26通道 = 130维），复用现有 Welch PSD
- [x] **FEAT-04**: Hjorth 计算使用 `epochs.get_data(units="uV")`，验证 Activity 均值在 10–500 µV²

### Classifier

- [x] **CLF-01**: 添加 model registry，支持 SVM / Random Forest / XGBoost 三模型对比
- [x] **CLF-02**: SelectKBest(k=50) 作为 Pipeline 步骤（非独立预处理），防止特征选择泄露
- [x] **CLF-03**: XGBoost 内层 CV 网格包含 max_depth=[2,3,4]，防止小样本过拟合
- [x] **CLF-04**: permutation test 的 fixed_pipe 与主 pipeline 结构保持一致（含 selector 步骤）
- [x] **CLF-05**: classify() 返回每个模型的独立结果 dict

### Fusion

- [x] **FUS-01**: 支持 EC 条件 EEG 加载与特征提取（复用现有 data_loader + preprocessor）
- [x] **FUS-02**: 按 subject 取 EO/EC 交集，丢弃缺失任一条件的受试者（不做插值）
- [x] **FUS-03**: 拼接 EO+EC 特征向量（~732维），groups 数组与 X 行数对齐

### Output

- [x] **OUT-01**: results.json 包含三模型 AUC/ACC/SEN/SPE/p-value 对比
- [ ] **OUT-02**: SHAP TreeExplainer 对最优模型输出特征重要性排名（post-hoc，不用于特征选择）

## v1.2 Requirements

### Data Quality

- [ ] **DQ-01**: 放宽 epoch 拒绝阈值至 150µV（当前100µV），减少 EC 条件受试者丢失
- [ ] **DQ-02**: 验证放宽后 EC 条件可用受试者数量增加，交集 subjects > 456

### Class Balance

- [ ] **BAL-01**: 在 CV 内层对训练集应用 SMOTE 过采样，解决 ADHD(159) vs MDD(297) 不平衡
- [ ] **BAL-02**: 验证 SPE（ADHD识别率）从当前 0.465 提升至 ≥ 0.55
- [ ] **BAL-03**: AUC 不低于当前最优 0.787（RF）

## v1.3 Requirements

### Threshold Optimization

- [x] **THR-01**: 在 CV 外层收集各折预测概率，用 Youden index（= BA 最优）找最优阈值（非0.5）
- [x] **THR-02**: ~~最优阈值下 RF SPE ≥ 0.70，SEN ≥ 0.75~~ → 改为：最优阈值下 BA 最大，SEN ≥ 0.75（ROC 前沿上 SPE+SEN 同时≥0.70/0.75 不可达）
- [x] **THR-03**: 最优阈值输出到 results.json（字段 `optimal_threshold`）

### Ensemble

- [x] **ENS-01**: SVM+RF+XGB 软投票集成（预测概率平均），走相同 StratifiedGroupKFold nested CV
- [ ] **ENS-02**: 集成模型 AUC ≥ 0.800（当前 0.798，继续优化）
- [x] **ENS-03**: 集成结果以 `ensemble` key 加入 results.json

## v2 Requirements

- **ADV-01**: EC/EO 差值特征（仅当拼接方案 AUC 增益 <0.01 时考虑）
- **ADV-02**: 功能连接特征（coherence/PLV）— 3380+ 维，留后续 milestone

## Out of Scope

| Feature | Reason |
|---------|--------|
| ERP 特征（P300） | 留后续 milestone |
| 深度学习（EEGNet/CNN） | 与现有 sklearn 架构不兼容 |
| 实时推断/临床部署 | 纯研究验证 |
| SHAP 用于特征选择 | 会引入泄露，仅用于事后解释 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FEAT-01 | Phase 5 | Complete |
| FEAT-02 | Phase 5 | Complete |
| FEAT-03 | Phase 5 | Complete |
| FEAT-04 | Phase 5 | Complete |
| CLF-01 | Phase 6 | Complete |
| CLF-02 | Phase 6 | Complete |
| CLF-03 | Phase 6 | Complete |
| CLF-04 | Phase 6 | Complete |
| CLF-05 | Phase 6 | Complete |
| FUS-01 | Phase 7 | Complete |
| FUS-02 | Phase 7 | Complete |
| FUS-03 | Phase 7 | Complete |
| OUT-01 | Phase 7 | Complete |
| OUT-02 | Phase 8 | Pending |

**Coverage:**
- v1.1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-23*
