---
phase: 04-classification-output
verified: 2026-02-23T08:42:13Z
status: passed
score: 4/4 must-haves verified
---

# Phase 4: Classification and Output Verification Report

**Phase Goal:** 用交叉验证和置换检验验证频谱特征对 MDD vs ADHD 的分类能力，并输出可复现的结果
**Verified:** 2026-02-23T08:42:13Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 控制台打印 AUC、准确率、敏感性、特异性和置换检验 p 值 | VERIFIED | main.py:38-42 prints all five metrics |
| 2 | results.json 被创建，包含所有指标和配置信息 | VERIFIED | File exists; auc=0.781, accuracy=0.748, sensitivity=0.857, specificity=0.568, permutation_pvalue=0.000999, n_permutations=1000, n_splits=5 |
| 3 | 交叉验证按 subject 分组，无数据泄露（StratifiedGroupKFold） | VERIFIED | classifier.py:3,24 imports and instantiates StratifiedGroupKFold; test_no_group_leakage() asserts train/test group disjointness |
| 4 | 1000 次置换检验完成并报告 p 值 | VERIFIED | classifier.py:37-41 runs 1000-iteration permutation loop; results.json n_permutations=1000, p=0.000999 |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `classifier.py` | `classify(X, y, groups) -> dict` | VERIFIED | 52 lines, substantive; exports `classify`; imported by main.py |
| `main.py` | Calls classify(), prints results, saves results.json | VERIFIED | Lines 8,37-54 wire classify() call, print, and json.dump |
| `results.json` | JSON with all metric fields | VERIFIED | All 7 required keys present, AUC=0.781 in range 0.5-1.0 |
| `tests/test_classifier.py` | Contract test | VERIFIED | 4 tests: metrics contract, no-leakage, MDD proba index, result stability |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `main.py` | `classifier.classify` | `from classifier import classify` | WIRED | main.py:8 import + main.py:37 `classify(X, y, groups)` |
| `classifier.py` | `StratifiedGroupKFold` | `sklearn.model_selection` | WIRED | classifier.py:3 import + classifier.py:24 instantiation |
| `main.py` | `results.json` | `json.dump` | WIRED | main.py:53 `json.dump(output, f, indent=2)` |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| CLF-01 | StratifiedGroupKFold 交叉验证，按 subject 分组 | SATISFIED | classifier.py:24 `cv = StratifiedGroupKFold(n_splits=n_splits)`; groups passed to all CV calls |
| CLF-02 | SVM（RBF kernel）+ 标准化 | SATISFIED | classifier.py:18-23 Pipeline with StandardScaler + SVC(kernel="rbf", probability=True) |
| CLF-03 | 报告指标：AUC、准确率、敏感性、特异性 | SATISFIED | classifier.py:29-34 computes all four; main.py:38-42 prints all four |
| CLF-04 | 1000 次置换检验，报告 p 值 | SATISFIED | classifier.py:36-41 manual 1000-iteration loop; p=0.000999 in results.json |
| OUT-01 | 控制台打印结果摘要 | SATISFIED | main.py:38-42 prints AUC/ACC/SEN/SPE/p in one line |
| OUT-02 | 保存结果到 JSON 文件 | SATISFIED | main.py:44-54 builds output dict with timestamp+condition+metrics, writes results.json |

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, no empty implementations in classifier.py or main.py.

### Human Verification Required

None. All truths are programmatically verifiable via file inspection and results.json content.

### Implementation Note

The permutation test uses a manual loop (cross_val_score on permuted labels) rather than sklearn's permutation_test_score. This is functionally equivalent and produces a valid p-value via (count >= observed + 1) / (n_permutations + 1). The p=0.000999 result confirms statistically significant discriminative power.

---

_Verified: 2026-02-23T08:42:13Z_
_Verifier: Claude (gsd-verifier)_
