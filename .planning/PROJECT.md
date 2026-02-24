# TDBRAIN MDD vs ADHD 鉴别诊断

## What This Is

基于 TDBRAIN 公开数据集，用静息态 EEG 频谱特征验证能否区分 MDD（320例）和 ADHD（200例）。核心问题：EEG 频谱特征是否能在置换检验显著的前提下达到 AUC > 0.70。

## Core Value

用频谱特征（theta/beta ratio、alpha 不对称）在 TDBRAIN 上跑通 MDD vs ADHD 分类，置换检验显著即为成功。

## Current Milestone: v2.0 Stacking + 功能连接

**Goal:** 用 stacking meta-learner 替代软投票集成，加入功能连接特征（coherence/PLV），冲击 AUC≥0.800

**Target features:**
- Stacking ensemble（meta-learner 替代 soft-vote）
- 功能连接特征（coherence、PLV）
- EC/EO 差值特征（如拼接方案增益不足）

## Requirements

### Validated

- ✓ 从 participants.tsv 筛出 MDD/ADHD 并加载对应 EEG 文件（.vhdr） — v1.0
- ✓ 提取频谱特征：theta/beta ratio、frontal alpha asymmetry、band power — v1.0
- ✓ StratifiedGroupKFold 交叉验证（按 subject 分组，防数据泄露） — v1.0
- ✓ 1000 次置换检验，报告 AUC 和 p 值 — v1.0
- ✓ 结果达到 AUC > 0.70 且 p < 0.05 — v1.0 (AUC=0.796, p=0.001)
- ✓ Hjorth 参数 + spectral entropy 特征 — v1.1
- ✓ SVM/RF/XGBoost 多模型对比 + SelectKBest — v1.1
- ✓ EC/EO 双条件特征融合（992维） — v1.1
- ✓ SHAP post-hoc 特征重要性 — v1.1
- ✓ SMOTE 类别平衡 + 阈值放宽至150µV — v1.2
- ✓ Youden 阈值优化 + 软投票集成 — v1.3 (ENS AUC=0.798, BA=0.753)

### Active (v2.0)

- [ ] Stacking ensemble（meta-learner 替代 soft-vote）
- [ ] 功能连接特征（coherence/PLV）
- [ ] AUC≥0.800 目标

### Out of Scope

- ERP 特征（P300）— 留后续 milestone
- 深度学习（EEGNet/CNN）— 与 sklearn 架构不兼容
- 实时推断 / 临床部署 — 纯研究验证

## Context

- 数据：/d/eeg/TDBRAIN-dataset/，BIDS 格式，.vhdr EEG 文件
- 样本量：MDD=320，ADHD=200（来自 participants.tsv）
- 参考实现：/d/eeg/modma_mdd_real_experiment.py（MODMA pipeline）
- EEG：64 通道，静息态（EC/EO），BrainVision 格式
- 样本数据：/d/eeg/TD-BRAIN-SAMPLE/

## Constraints

- 数据：仅用 TDBRAIN-dataset
- 特征：v1 只做频谱特征
- 验证：必须有置换检验
- 代码：可参考 MODMA pipeline 结构，独立实现

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 先只做频谱特征 | 最小可行验证 | ✓ Good — 262维特征足以达到 AUC=0.796 |
| StratifiedGroupKFold | 防 subject 泄露 | ✓ Good — 组级分组验证通过 |
| 1000 次置换检验 | 统计严谨性 | ✓ Good — p=0.001，显著 |
| 手动置换检验替代 permutation_test_score | StratifiedGroupKFold 兼容性问题 | ✓ Good — 修复了 p=1.0 bug |
| class_weight='balanced' 纳入 GridSearchCV 调参 | 消除参数选择偏差 | ✓ Good — Nested CV 结果与 flat CV 一致 |
| drop_duplicates 去重受试者 | participants.tsv 存在重复行 | ✓ Good — 477 唯一受试者 |

---
*Last updated: 2026-02-24 after v2.0 milestone start*
