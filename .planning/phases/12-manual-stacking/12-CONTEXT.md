# Phase 12: Manual Stacking Ensemble - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

手动 stacking 循环替换现有 soft-vote 集成。训练 LogisticRegression meta-learner 学习 SVM/RF/XGB 最优权重。可先用现有 992 维特征验证，不依赖 Phase 11 连接特征。

</domain>

<decisions>
## Implementation Decisions

### Stacking CV 结构
- 内层 3-fold StratifiedGroupKFold（严格 subject 隔离）
- Base model 在 stacking 层使用固定超参数（不做 GridSearchCV）
- 超参数来源：预跑一次完整 pipeline 取各模型最优值，硬编码到 stacking
- 避免 3 层嵌套 CV 的运行时间爆炸

### Fallback 策略
- 保留 stacking 和 soft-vote 两者结果对比
- Stacking 结果新增 `stacking` key，现有 `ensemble` key 保持不变
- 当 stacking AUC < soft-vote AUC 时：console warning + JSON 对比字段 `stacking_vs_ensemble`
- 记录诊断信息：meta-learner 3 个系数（SVM/RF/XGB 权重）+ OOF 单模型 AUC

### 结果报告格式
- 完整指标集：AUC, BA, SEN, SPE, optimal_threshold（与现有模型一致）
- Stacking 的 optimal_threshold 用 Youden index（与现有模型一致）
- p-value 不在 Phase 12 做，留给 Phase 14 统一处理置换检验

### Claude's Discretion
- meta-learner 正则化参数 C 的具体值
- OOF 预测概率的存储格式
- console warning 的具体措辞

</decisions>

<specifics>
## Specific Ideas

- 预跑取最优超参数的方式：可以从现有 results.json 或 cache 中提取，不需要额外运行
- stacking_vs_ensemble 对比字段应包含 delta AUC 和 delta BA
- meta-learner 系数可以帮助判断哪个 base model 贡献最大

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-manual-stacking*
*Context gathered: 2026-02-24*
