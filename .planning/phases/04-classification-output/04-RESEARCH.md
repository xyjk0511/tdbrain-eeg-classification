# Phase 4: Classification & Output - Research

**Researched:** 2026-02-23
**Domain:** scikit-learn cross-validation, SVM, permutation test, JSON output
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLF-01 | StratifiedGroupKFold 交叉验证，按 subject 分组，防数据泄露 | StratifiedGroupKFold(n_splits=5) verified in sklearn 1.8.0; groups=subject_ids from main.py |
| CLF-02 | SVM(RBF) + 标准化 pipeline | Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', probability=True))]) verified |
| CLF-03 | AUC、准确率、敏感性、特异性 | cross_val_predict(method='predict_proba'/'predict') + roc_auc_score + confusion_matrix verified |
| CLF-04 | 1000 次置换检验，报告 p 值 | permutation_test_score(n_permutations=1000) returns (score, perm_scores, pvalue) verified |
| OUT-01 | 控制台打印结果摘要 | print() with AUC, accuracy, sensitivity, specificity, p-value |
| OUT-02 | 保存结果到 JSON 文件 | json.dump() with float() conversion; all metrics + config serializable |
</phase_requirements>

## Summary

Phase 3 delivers X(n_subjects, 262), y(n_subjects,), groups(n_subjects,) in main.py. Phase 4 adds classifier.py and extends main.py to call it. sklearn 1.8.0 is already installed — no new dependencies needed.

The critical correctness constraint is group-based CV: each subject's epochs must stay entirely in train or test. StratifiedGroupKFold enforces this. The label encoding order (ADHD=0, MDD=1 alphabetically) must be consistent between roc_auc_score and confusion_matrix calls.

permutation_test_score handles the 1000-permutation test in one call and returns the p-value directly. It reuses the same Pipeline and CV object.

**Primary recommendation:** Add classifier.py with classify(X, y, groups) -> dict; extend main.py to call it and save JSON.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | StratifiedGroupKFold, Pipeline, SVC, permutation_test_score | Already installed; all APIs verified |
| numpy | (installed) | Array ops | Already used throughout |
| json | stdlib | JSON serialization | No extra dependency |

**Installation:** No new dependencies needed.

## Architecture Patterns

### Recommended Project Structure

```
tdbrain/
  config.py              # unchanged
  data_loader.py         # unchanged
  preprocessor.py        # unchanged
  feature_extractor.py   # unchanged
  classifier.py          # NEW: classify(X, y, groups) -> dict
  main.py                # extend: call classify(), print, save JSON
```

### Pattern 1: classify(X, y, groups) -> dict

```python
# Source: verified against sklearn 1.8.0
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, permutation_test_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np

def classify(X, y, groups, n_splits=5, n_permutations=1000, random_state=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)          # ADHD=0, MDD=1 (alphabetical)
    mdd_idx = list(le.classes_).index("MDD")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, random_state=random_state)),
    ])
    cv = StratifiedGroupKFold(n_splits=n_splits)

    proba = cross_val_predict(pipe, X, y_enc, groups=groups, cv=cv, method="predict_proba")
    pred  = cross_val_predict(pipe, X, y_enc, groups=groups, cv=cv, method="predict")

    auc = roc_auc_score(y_enc, proba[:, mdd_idx])
    acc = accuracy_score(y_enc, pred)
    tn, fp, fn, tp = confusion_matrix(y_enc, pred).ravel()

    _, _, pvalue = permutation_test_score(
        pipe, X, y_enc, groups=groups, cv=cv,
        n_permutations=n_permutations, scoring="roc_auc",
        n_jobs=-1, random_state=random_state,
    )

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "permutation_pvalue": float(pvalue),
        "n_permutations": n_permutations,
        "n_splits": n_splits,
    }
```

### Pattern 2: main.py extension

```python
# After the subject loop that builds X, y, groups:
from classifier import classify
import json, datetime

results = classify(X, y, groups)
print(f"AUC={results['auc']:.3f}  ACC={results['accuracy']:.3f}  "
      f"SEN={results['sensitivity']:.3f}  SPE={results['specificity']:.3f}  "
      f"p={results['permutation_pvalue']:.4f}")

output = {
    "timestamp": datetime.datetime.now().isoformat(),
    "condition": CONDITION,
    "n_subjects": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "label_counts": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    **results,
}
with open("results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved results.json")
```

### Anti-Patterns to Avoid

- **StandardScaler outside Pipeline:** Fitting scaler on all X before CV leaks test statistics into training.
- **String labels to roc_auc_score:** Requires numeric labels; use LabelEncoder first.
- **numpy float64 in json.dump:** Not JSON-serializable; wrap all metrics with float().
- **SVC without probability=True:** cross_val_predict(method='predict_proba') requires it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Permutation test loop | Manual shuffle-and-score loop | permutation_test_score | Handles parallel execution, p-value formula, edge cases |
| Stratified group splitting | Manual fold assignment | StratifiedGroupKFold | Guarantees class balance AND no subject leakage per fold |
| Feature scaling in CV | Fit scaler on all X then CV | Pipeline with StandardScaler | Pipeline fits scaler only on train fold |

## Common Pitfalls

### Pitfall 1: Data leakage via StandardScaler outside Pipeline

**What goes wrong:** AUC is inflated; results not reproducible on new data.
**Why it happens:** Fitting StandardScaler on all X before CV lets test-fold statistics influence training.
**How to avoid:** Always put StandardScaler inside Pipeline.
**Warning signs:** AUC suspiciously high (>0.95).

### Pitfall 2: LabelEncoder class order assumption

**What goes wrong:** AUC computed for wrong class; sensitivity/specificity swapped.
**Why it happens:** LabelEncoder sorts alphabetically: ADHD=0, MDD=1. Assuming MDD=0 gives wrong proba column.
**How to avoid:** Use `list(le.classes_).index("MDD")` to get the correct proba column index.
**Warning signs:** AUC < 0.5 (often means 1 - correct_AUC).

### Pitfall 3: permutation_test_score with n_jobs=-1 on Windows

**What goes wrong:** Process hangs or spawns infinite subprocesses.
**Why it happens:** Windows uses spawn (not fork); joblib requires `if __name__ == "__main__"` guard.
**How to avoid:** classify() must be called inside `if __name__ == "__main__":` in main.py (already the case).
**Warning signs:** Process hangs without output.

### Pitfall 4: Too few subjects per class for n_splits=5

**What goes wrong:** ValueError from StratifiedGroupKFold.split().
**Why it happens:** If subjects are skipped during loading, class count may drop below n_splits.
**How to avoid:** Print label counts before calling classify(); n_splits=5 requires >=5 per class.
**Warning signs:** ValueError mentioning "least populated class".

## Code Examples

### permutation_test_score return tuple

```python
# Source: verified on sklearn 1.8.0
# Returns (observed_score, permutation_scores, pvalue)
score, perm_scores, pvalue = permutation_test_score(
    pipe, X, y_enc, groups=groups, cv=cv,
    n_permutations=1000, scoring="roc_auc",
    n_jobs=-1, random_state=42,
)
# pvalue = (sum(perm_scores >= score) + 1) / (n_permutations + 1)
```

### Sensitivity/specificity from confusion_matrix

```python
# Source: verified on sklearn 1.8.0
# Binary confusion_matrix ravel(): (TN, FP, FN, TP)
tn, fp, fn, tp = confusion_matrix(y_enc, pred).ravel()
sensitivity = tp / (tp + fn)   # recall for MDD (positive class)
specificity = tn / (tn + fp)   # recall for ADHD (negative class)
```

## Open Questions

1. **n_splits value**
   - What we know: Expected ~320 MDD, ~172 ADHD after preprocessing skips
   - What is unclear: Exact subject count after skips is unknown until runtime
   - Recommendation: Use n_splits=5; print class counts before CV as a guard

## Sources

### Primary (HIGH confidence)

- sklearn 1.8.0 installed — StratifiedGroupKFold, permutation_test_score, cross_val_predict, Pipeline, SVC, confusion_matrix all verified via live Python execution
- permutation_test_score signature verified: params=['estimator','X','y','groups','cv','n_permutations','n_jobs','random_state','verbose','scoring','params']
- permutation_test_score return verified: (score, perm_scores, pvalue) with perm_scores.shape=(n_permutations,)
- LabelEncoder alphabetical ordering verified: ADHD=0, MDD=1
- confusion_matrix binary ravel() pattern verified: (tn, fp, fn, tp)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — sklearn 1.8.0 installed, all APIs verified by live execution
- Architecture: HIGH — full pattern tested on synthetic data, return shapes confirmed
- Pitfalls: HIGH — label encoding order and Pipeline scaler leakage verified empirically

**Research date:** 2026-02-23
**Valid until:** 2026-03-23
