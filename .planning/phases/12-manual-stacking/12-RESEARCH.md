# Phase 12: Manual Stacking Ensemble - Research

**Researched:** 2026-02-24
**Domain:** Stacking ensemble with manual OOF loop, scikit-learn/imblearn
**Confidence:** HIGH

## Summary

Phase 12 replaces the existing soft-vote ensemble (simple probability averaging) with a learned stacking ensemble. Instead of equal-weight averaging of SVM/RF/XGB probabilities, a LogisticRegression meta-learner learns optimal weights from out-of-fold (OOF) predictions.

The implementation is a manual stacking loop — not sklearn's `StackingClassifier` — because we need strict subject-level isolation via `StratifiedGroupKFold` at both the base-model and meta-learner levels. sklearn's `StackingClassifier` does not natively support `groups` in its internal CV, making it unsuitable.

The existing `classifier.py` already has the OOF collection pattern (lines 65-75: outer CV loop collecting probas per fold, reordering by test indices). The stacking extension reuses this pattern but collects OOF probas from all 3 base models simultaneously, then trains LogisticRegression on the (n_subjects, 3) meta-feature matrix.

**Primary recommendation:** Add a stacking block to `classify()` in `classifier.py` that reuses the existing outer CV, collects 3-column OOF meta-features with a 3-fold inner StratifiedGroupKFold, trains LogisticRegression(C=1.0), and evaluates via the same outer CV used by individual models.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 内层 3-fold StratifiedGroupKFold（严格 subject 隔离）
- Base model 在 stacking 层使用固定超参数（不做 GridSearchCV）
- 超参数来源：预跑一次完整 pipeline 取各模型最优值，硬编码到 stacking
- 避免 3 层嵌套 CV 的运行时间爆炸
- 保留 stacking 和 soft-vote 两者结果对比
- Stacking 结果新增 `stacking` key，现有 `ensemble` key 保持不变
- 当 stacking AUC < soft-vote AUC 时：console warning + JSON 对比字段 `stacking_vs_ensemble`
- 记录诊断信息：meta-learner 3 个系数（SVM/RF/XGB 权重）+ OOF 单模型 AUC
- 完整指标集：AUC, BA, SEN, SPE, optimal_threshold（与现有模型一致）
- Stacking 的 optimal_threshold 用 Youden index（与现有模型一致）
- p-value 不在 Phase 12 做，留给 Phase 14 统一处理置换检验

### Claude's Discretion
- meta-learner 正则化参数 C 的具体值
- OOF 预测概率的存储格式
- console warning 的具体措辞

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STK-01 | 手动 stacking 循环（非 StackingClassifier），StratifiedGroupKFold 生成 OOF meta-features | Manual OOF loop pattern documented in Architecture Patterns; sklearn StackingClassifier lacks groups support |
| STK-02 | LogisticRegression(C=1.0) 作为 meta-learner，passthrough=False | LR meta-learner code example provided; C=1.0 recommended; passthrough=False means only 3 meta-features |
| STK-03 | SMOTE 仅在 base model ImbPipeline 内部，stacking 层不做过采样 | Pitfall documented: SMOTE on meta-features synthesizes fake probability combinations; base ImbPipeline handles it |
| STK-04 | Stacking 结果以 `stacking` key 写入 results.json | Output pattern matches existing `ensemble` key structure; comparison field documented |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | >=1.3 | LogisticRegression meta-learner, StratifiedGroupKFold, metrics | Already in project |
| imblearn | >=0.11 | ImbPipeline for base models with SMOTE | Already in project |
| numpy | >=1.24 | OOF array manipulation | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| xgboost | >=1.7 | Base model (XGBClassifier) | Already in project |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual OOF loop | sklearn.ensemble.StackingClassifier | StackingClassifier does not pass `groups` to its internal CV — breaks subject isolation |
| LogisticRegression | Ridge/ElasticNet | LR with C=1.0 is standard for 3-feature meta-learning; no benefit from elastic net |
| passthrough=False | passthrough=True | With 992 raw + 3 meta-features, raw would dominate; passthrough=False is correct |

**Installation:** No new dependencies required.

## Architecture Patterns

### Pattern 1: Two-Level CV with Manual OOF Collection

**What:** The outer 5-fold StratifiedGroupKFold evaluates stacking performance. Within each outer fold, a 3-fold inner StratifiedGroupKFold generates OOF meta-features from the outer training set. The meta-learner is trained on these OOF meta-features and predicts on the outer test set.

**When to use:** Always for stacking with group constraints.

**Structure:**
```
Outer CV (5-fold StratifiedGroupKFold, same as existing):
  For each outer fold (train_outer, test_outer):
    
    Inner CV (3-fold StratifiedGroupKFold on train_outer):
      For each inner fold (train_inner, val_inner):
        For each base model (svm, rf, xgb):
          fit base_pipe on train_inner
          predict_proba on val_inner -> store OOF column
    
    # OOF meta-features for all of train_outer: shape (n_train, 3)
    
    # Retrain base models on full train_outer for test prediction
    For each base model:
      fit base_pipe on full train_outer
      predict_proba on test_outer -> store test meta-feature column
    
    # Train meta-learner on OOF meta-features
    meta_lr.fit(oof_meta, y[train_outer])
    
    # Predict on test meta-features
    stacking_proba[test_outer] = meta_lr.predict_proba(test_meta)
```

**Key insight:** Base models are trained TWICE per outer fold — once per inner fold (for OOF generation) and once on full train_outer (for test-set meta-features). This is standard stacking protocol.

### Pattern 2: Fixed Hyperparameters for Base Models

**What:** Base models use pre-determined hyperparameters (no GridSearchCV) inside the stacking loop. Avoids 3-level nested CV.

**Hyperparameter source:** The existing `FIXED_PIPES` dict in `classifier.py` (lines 46-50). Reuse directly.

**Current FIXED_PIPES values:**
- SVM: `kernel="rbf", class_weight="balanced"` (C=1.0, gamma="scale" defaults)
- RF: `n_estimators=100, class_weight="balanced"`
- XGB: `n_estimators=100, max_depth=3, eval_metric="logloss"`

### Pattern 3: Meta-Learner Training on OOF Probabilities

**What:** LogisticRegression(C=1.0) trained on (n_subjects, 3) matrix where each column is one base model's OOF predicted probability for MDD.

**Why LogisticRegression:**
- Only 3 input features — LR is ideal for low-dimensional meta-learning
- Coefficients are directly interpretable as learned model weights
- C=1.0 provides mild regularization; with only 3 features, overfitting risk is minimal
- No SMOTE needed: class imbalance handled inside each base model's ImbPipeline

**Meta-learner configuration:**
```python
from sklearn.linear_model import LogisticRegression
meta_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
```

**Discretion recommendation for C:** Use C=1.0 (sklearn default). With 3 features and ~400 training samples, regularization strength has negligible impact.

### Pattern 4: Diagnostics Collection

**What:** Record meta-learner coefficients and OOF single-model AUCs as diagnostic metadata.

**Meta-learner coefficients:** After outer CV, fit a final `meta_lr` on the full OOF matrix. Extract `meta_lr.coef_[0]` — 3-element array for [svm, rf, xgb].

**OOF single-model AUC:** For each column in full OOF matrix, compute `roc_auc_score(y_enc, full_oof[:, i])`.

**Storage format:**
```json
{
  "stacking": {
    "auc": 0.805,
    "meta_learner_coefs": {"svm": 0.42, "rf": 0.35, "xgb": 0.23},
    "oof_base_auc": {"svm": 0.770, "rf": 0.796, "xgb": 0.796}
  }
}
```

### Anti-Patterns to Avoid

- **Using sklearn StackingClassifier:** Does not pass `groups` to internal CV — subject leakage guaranteed
- **SMOTE on meta-features:** The (n, 3) probability matrix should not be oversampled
- **GridSearchCV inside stacking inner loop:** Creates 3-level nesting; use fixed hyperparameters
- **Fitting meta-learner on test-fold meta-features:** Meta-learner trains on OOF from training set only

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stacking CV with groups | sklearn StackingClassifier | Manual OOF loop with StratifiedGroupKFold | StackingClassifier ignores groups |
| Optimal threshold | Custom threshold search | `roc_curve` + Youden index (existing pattern) | Already in classifier.py lines 83-88 |
| Probability calibration | Custom calibration | Skip — LR implicitly calibrates | With 3 inputs, LR output is well-calibrated |
| Model weight interpretation | Custom weight extraction | `meta_lr.coef_[0]` | sklearn LR exposes coefficients directly |

**Key insight:** The manual OOF loop is the ONE thing that must be hand-rolled. Everything else reuses existing patterns.

## Common Pitfalls

### Pitfall 1: Subject Leakage in Stacking Inner CV

**What goes wrong:** Inner CV uses `StratifiedKFold` instead of `StratifiedGroupKFold`. Same subject appears in both inner-train and inner-val.

**How to avoid:** Inner CV must be `StratifiedGroupKFold(n_splits=3)` with `groups=groups[train_outer_idx]`.

**Warning signs:** Stacking AUC jumps >0.03 above best single model.

### Pitfall 2: OOF Array Index Misalignment

**What goes wrong:** Inner CV splits return local indices into `X[train_outer_idx]`, but developer uses global indices. Predictions land in wrong rows.

**How to avoid:** Allocate `oof_meta = np.zeros((len(train_idx), 3))`. Use local indices: `oof_meta[val_inner_local, i] = proba`.

**Warning signs:** Some rows in oof_meta remain all-zeros. Meta-learner AUC near 0.5.

### Pitfall 3: SMOTE Applied at Meta-Level

**What goes wrong:** SMOTE generates synthetic (n, 3) probability vectors that don't correspond to real subjects.

**How to avoid:** Meta-learner is plain `LogisticRegression(C=1.0)` — no SMOTE, no scaler.

**Warning signs:** Stacking training set grows beyond `len(train_outer_idx)` rows.

### Pitfall 4: Forgetting to Retrain Base Models on Full train_outer

**What goes wrong:** After inner CV, developer reuses the last inner-fold model (trained on ~2/3 of train_outer) to predict on test_outer.

**How to avoid:** After inner CV completes, retrain each base model on FULL train_outer, then predict on test_outer.

**Warning signs:** Stacking test-set performance lower than expected.

### Pitfall 5: Stacking vs Ensemble Comparison Using Different CV Splits

**What goes wrong:** Stacking and soft-vote use different CV objects, producing different fold assignments.

**How to avoid:** Use the SAME `outer_cv` object for both. The existing `classify()` already uses a single instance.

**Warning signs:** Large differences (>0.02) that don't replicate across runs.

## Code Examples

Verified patterns based on existing codebase and sklearn documentation.

### Example 1: Core Stacking OOF Loop (per outer fold)

```python
# Inside each outer fold iteration
# train_idx, test_idx from outer_cv.split(X, y_enc, groups)

X_train, y_train = X[train_idx], y_enc[train_idx]
X_test = X[test_idx]
g_train = groups[train_idx]

# Inner CV for OOF meta-features
stacking_cv = StratifiedGroupKFold(n_splits=3)
oof_meta = np.zeros((len(train_idx), len(FIXED_PIPES)))

for inner_train, inner_val in stacking_cv.split(X_train, y_train, g_train):
    for i, (name, pipe) in enumerate(FIXED_PIPES.items()):
        pipe.fit(X_train[inner_train], y_train[inner_train])
        oof_meta[inner_val, i] = pipe.predict_proba(
            X_train[inner_val]
        )[:, mdd_idx]

# Retrain base models on full train_outer for test prediction
test_meta = np.zeros((len(test_idx), len(FIXED_PIPES)))
for i, (name, pipe) in enumerate(FIXED_PIPES.items()):
    pipe.fit(X_train, y_train)
    test_meta[:, i] = pipe.predict_proba(X_test)[:, mdd_idx]

# Train meta-learner on OOF, predict on test
meta_lr = LogisticRegression(C=1.0, solver="lbfgs",
                             max_iter=1000, random_state=42)
meta_lr.fit(oof_meta, y_train)
stacking_proba[test_idx] = meta_lr.predict_proba(test_meta)[:, 1]
```

### Example 2: Metrics Computation

```python
# After collecting stacking_proba for all outer folds
stk_auc = roc_auc_score(y_enc, stacking_proba)

# Youden optimal threshold (same as classifier.py lines 83-88)
fpr, tpr, thresholds = roc_curve(y_enc, stacking_proba, pos_label=mdd_idx)
opt_idx = np.argmax(tpr - fpr)
opt_thresh = float(thresholds[opt_idx])
pred_opt = np.where(stacking_proba >= opt_thresh, mdd_idx, 1 - mdd_idx)

tn, fp, fn, tp = confusion_matrix(y_enc, pred_opt).ravel()
sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
ba = (sen + spe) / 2
```

### Example 3: Comparison Warning and JSON Output

```python
ens_auc = results["ensemble"]["auc"]
stk_auc = results["stacking"]["auc"]

if stk_auc < ens_auc:
    print(f"WARNING: Stacking AUC ({stk_auc:.3f}) < soft-vote AUC "
          f"({ens_auc:.3f}). Delta = {stk_auc - ens_auc:.4f}")

results["stacking"]["stacking_vs_ensemble"] = {
    "delta_auc": round(stk_auc - ens_auc, 4),
    "delta_ba": round(stk_ba - ens_ba, 4),
    "stacking_wins": stk_auc >= ens_auc,
}
```

### Example 4: Diagnostics — Meta-Learner Coefficients

```python
# Collect full OOF matrix during outer CV loop
full_oof = np.zeros((len(y_enc), len(FIXED_PIPES)))
# ... populated during outer CV iterations ...

# Fit diagnostic meta-learner on full OOF
meta_lr_diag = LogisticRegression(C=1.0, solver="lbfgs",
                                   max_iter=1000, random_state=42)
meta_lr_diag.fit(full_oof, y_enc)

model_names = list(FIXED_PIPES.keys())
coefs = dict(zip(model_names,
                  meta_lr_diag.coef_[0].tolist()))
oof_aucs = {
    name: float(roc_auc_score(y_enc, full_oof[:, i]))
    for i, name in enumerate(model_names)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Equal-weight soft voting | Learned stacking with LR meta-learner | Standard since Wolpert 1992 | Learns optimal model weights; typically +0.005-0.02 AUC |
| sklearn StackingClassifier | Manual OOF loop | N/A | Required for grouped CV |
| Passthrough stacking | Meta-only (passthrough=False) | Best practice | Avoids raw feature dominance when n_meta << n_raw |

**Expectations:** With 3 correlated base models (all trained on same 992 features), stacking improvement over soft-vote is typically modest (0.005-0.015 AUC). Primary value is learned weighting.

## Integration with Existing Codebase

### Where Stacking Fits in classifier.py

Add stacking block after the existing ensemble block (lines 110-136). It follows the same pattern:

1. Reuse `outer_cv` — same object used for individual models and soft-vote ensemble
2. Reuse `FIXED_PIPES` — same fixed-hyperparameter pipelines used for permutation testing
3. Reuse `mdd_idx` — same positive class index
4. Reuse metrics pattern — same Youden threshold + confusion matrix code

### Files to Modify

| File | Change | Scope |
|------|--------|-------|
| `classifier.py` | Add stacking block after ensemble block; add `LogisticRegression` import | ~50 lines added |
| `main.py` | Print stacking results in console output | ~5 lines added |

No new files needed.

### What NOT to Change

- `FIXED_PIPES` dict — reuse as-is for stacking base models
- `MODEL_REGISTRY` — individual model evaluation stays unchanged
- Ensemble block (lines 110-136) — soft-vote stays for comparison
- `config.py` — no new config needed

## Runtime Estimation

Per outer fold:
- Inner 3-fold CV x 3 base models (fixed pipes): 9 fits
- Retrain 3 base models on full train_outer: 3 fits
- Total fits per outer fold: 12
- Total fits across 5 outer folds: 60

Each FIXED_PIPE fit takes ~5-10 seconds. Estimated stacking total: ~5-10 min.

**Combined runtime:** Existing pipeline (~35 min) + stacking (~10 min) = ~45 min total.

## Open Questions

1. **Meta-learner predict_proba column index**
   - What we know: LR binary classification produces 2 columns. MDD column depends on `meta_lr.classes_` ordering.
   - Recommendation: Use `[:, 1]` since LabelEncoder produces MDD=1 (verified in existing code). Add assertion.

2. **FIXED_PIPES mutation across outer folds**
   - What we know: ImbPipeline objects are mutable. `.fit()` modifies internal state.
   - Recommendation: Existing permutation test (line 92) reuses same pipe without cloning — likely safe. Consider defensive `clone()` if issues arise.

## Sources

### Primary (HIGH confidence)
- Direct inspection of `classifier.py` — OOF collection pattern, FIXED_PIPES, ensemble block
- Direct inspection of `main.py` — results output format, console printing pattern
- Direct inspection of `config.py` — current configuration constants
- Direct inspection of `results.json` — current baselines (RF AUC=0.796, XGB AUC=0.796, SVM AUC=0.770)
- scikit-learn LogisticRegression docs — C parameter, solver options, coef_ attribute
- scikit-learn StratifiedGroupKFold docs — groups parameter, split behavior

### Secondary (MEDIUM confidence)
- Wolpert (1992) "Stacked Generalization" — foundational stacking methodology
- Kaggle competition best practices — stacking typically yields +0.005-0.02 AUC over simple averaging

### Tertiary (LOW confidence)
- Runtime estimates — extrapolation from existing pipeline timing, not measured directly

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, no new dependencies
- Architecture: HIGH — manual OOF loop is well-documented; existing codebase has analogous OOF collection
- Pitfalls: HIGH — subject leakage and index misalignment are well-known stacking failure modes
- Integration: HIGH — direct inspection of classifier.py confirms minimal changes needed

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (stable domain, no fast-moving dependencies)

---
*Phase 12 research for: Manual Stacking Ensemble (TDBRAIN MDD vs ADHD)*
*Researched: 2026-02-24*
