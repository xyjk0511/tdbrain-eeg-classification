# Project Research Summary

**Project:** TDBRAIN EEG MDD vs ADHD Classifier - v1.1 Performance Improvement
**Domain:** EEG-based psychiatric classification (resting-state, classical ML)
**Researched:** 2026-02-23
**Confidence:** HIGH

## Executive Summary

The v1.1 milestone adds three independent improvements to the existing SVM pipeline (AUC 0.796): new time-domain and entropy features, EC/EO dual-condition fusion, and multi-model comparison (RF + XGBoost). All three integrate cleanly into the existing sklearn Pipeline + nested StratifiedGroupKFold architecture. The recommended build order is: extend feature_extractor.py with Hjorth parameters and spectral entropy first (pure numpy, no new deps), then add RF/XGBoost to a model registry in classifier.py, then wire EC/EO fusion in main.py last with SelectKBest inside the pipeline for leak-safe feature selection.

The single highest risk is data leakage. Feature selection outside the CV pipeline, imputation on the full dataset before splitting, and SHAP used for selection rather than post-hoc explanation are all invisible bugs that inflate AUC by 0.05-0.15 without any error. Every new component must be wrapped inside the sklearn Pipeline so it re-fits only on training folds. The permutation test fixed_pipe must also be updated to match the real pipeline after any structural change.

The only external dependency to add is shap (pip install shap ~0.46, post-hoc only). antropy 0.1.9 and xgboost 3.2.0 are already installed and verified.

## Key Findings

### Recommended Stack

The existing stack (mne 1.11, sklearn 1.8, numpy 2.4, scipy 1.17) requires no changes. antropy 0.1.9 covers both Hjorth parameters and spectral entropy. xgboost 3.2.0 XGBClassifier drops into the existing Pipeline without refactoring. shap is used post-hoc only, never as a pipeline step.

**Core technologies:**
- antropy 0.1.9: Hjorth params + spectral entropy -- already installed, verified against numpy 2.x
- xgboost 3.2.0: XGBClassifier -- sklearn-compatible, drops into existing Pipeline/GridSearchCV
- sklearn SelectKBest: filter-based feature selection -- must be inside Pipeline, not standalone
- shap ~0.46: SHAP feature importance -- post-hoc explanation only, requires pip install shap

### Expected Features

**Must have (table stakes):**
- Hjorth Activity, Mobility, Complexity -- 78 features (3 x 26 channels), pure numpy, standard in every EEG ML paper
- Spectral Entropy broadband -- 26 features, reuses existing Welch PSD
- Spectral Entropy per band -- 130 features (5 bands x 26 channels), reuses existing band loop

**Should have (differentiators):**
- EC/EO feature concatenation -- doubles feature space; qEEG literature shows EC+EO outperforms either alone
- SelectKBest feature selection (k=50) -- critical once feature dim reaches ~732 post-fusion; must be inside Pipeline
- RF + XGBoost classifiers -- model registry pattern; compare against SVM baseline

**Defer (v2+):**
- EC/EO difference features -- add only if concatenation shows <0.01 AUC gain
- Functional connectivity (coherence, PLV) -- 3380+ features, out of scope per PROJECT.md
- Deep learning (EEGNet, CNN) -- incompatible with current sklearn architecture

### Architecture Approach

The pipeline is strictly additive. feature_extractor.py gains two private helpers (_extract_hjorth, _extract_spectral_entropy) concatenated to the existing 262-dim vector. EC/EO fusion is handled entirely in main.py via subject intersection -- feature_extractor.py stays condition-agnostic. classifier.py gains a model registry dict so the nested CV + permutation loop runs once per model entry. A new feature_selector.py is for exploratory SHAP analysis only; the actual leak-safe selection lives inside the Pipeline in classifier.py.

**Major components:**
1. feature_extractor.py (MODIFIED) -- add _extract_hjorth(), _extract_spectral_entropy(); output grows 262 -> ~496 per condition
2. classifier.py (MODIFIED) -- model registry {svm, rf, xgb}; SelectKBest inside each pipeline; returns per-model results dict
3. main.py (MODIFIED) -- dual-condition loop; subject intersection; ~732-dim fused X; calls classify() with all models
4. config.py (MODIFIED) -- add CONDITIONS=[EO,EC], N_FEATURES_SELECT=50
5. feature_selector.py (NEW) -- standalone SHAP/SelectKBest wrapper for exploratory analysis only

### Critical Pitfalls

1. **Feature selection outside CV pipeline** -- SelectKBest must be a Pipeline step, never fit on full X before CV. Leakage inflates AUC by 0.05-0.15 invisibly. Wrap as Pipeline([scaler, selector, clf]) and tune selector__k in param_grid.

2. **EC/EO imputation leakage** -- subjects missing one condition must be dropped (intersection), not imputed. Never call df.fillna(df.mean()) or SimpleImputer.fit_transform(X) on full X before CV split.

3. **Permutation test pipeline mismatch** -- fixed_pipe in the permutation loop must include the selector step after it is added to the main pipeline. Mismatched steps produce anti-conservative p-values.

4. **Hjorth unit error** -- epochs.get_data() returns Volts by default. Must use epochs.get_data(units="uV"). Mean Hjorth Activity should be 10-500 uV2; values <1e-6 indicate the unit error.

5. **XGBoost overfitting on small N** -- with ~300 training subjects, default XGBoost parameters overfit. Inner CV grid must include max_depth [2,3,4]. Outer-fold AUC std >0.08 is the warning sign.

## Implications for Roadmap

### Phase 1: New Features (feature_extractor.py)
**Rationale:** No dependencies on other phases; independently testable; unblocks everything downstream.
**Delivers:** 78 Hjorth + 156 entropy features; feature vector grows to ~496-dim per condition.
**Addresses:** Hjorth Activity/Mobility/Complexity, spectral entropy broadband + per-band.
**Avoids:** Hjorth unit error (verify mean Activity 10-500 uV2); spectral entropy normalization (assert values in [0, log(n_freqs)]).

### Phase 2: Config + Classifier Registry (config.py + classifier.py)
**Rationale:** Depends on Phase 1 feature shape being stable; must be done before main.py integration.
**Delivers:** CONDITIONS list in config; model registry with SVM, RF, XGBoost; SelectKBest inside each pipeline; per-model results dict.
**Uses:** xgboost 3.2.0 XGBClassifier, sklearn RF, SelectKBest(f_classif, k=50).
**Avoids:** XGBoost overfitting (max_depth grid [2,3,4]); feature selection leakage (selector inside Pipeline); permutation test mismatch (update fixed_pipe to include selector).

### Phase 3: EC/EO Fusion + Integration (main.py)
**Rationale:** Depends on Phases 1 and 2; integrates all components; highest complexity and leakage risk.
**Delivers:** Dual-condition loop; subject intersection; ~732-dim fused X; full multi-model nested CV run; updated results.json.
**Avoids:** EC/EO imputation leakage (drop missing subjects, not impute); groups array length mismatch after fusion.

### Phase 4: SHAP Post-hoc Analysis (feature_selector.py)
**Rationale:** Post-hoc only; runs after final CV loop completes; no leakage risk if kept outside CV.
**Delivers:** Feature importance ranking via shap.TreeExplainer on best-performing model.
**Uses:** shap ~0.46 TreeExplainer (requires pip install shap).
**Avoids:** SHAP leakage (never use SHAP output to select features for reported AUC).

### Phase Ordering Rationale

- Phase 1 first: feature_extractor.py has zero dependencies and its output shape must be stable before classifier or main.py can be written.
- Phase 2 before Phase 3: main.py calls classify() -- the model registry interface must exist first.
- Phase 4 last: TreeExplainer requires a fitted model from the completed CV run.
- EC/EO fusion in Phase 3 (not Phase 1): requires both the new feature shape (Phase 1) and the classifier interface (Phase 2) to be finalized.

### Research Flags

Phases with standard patterns (skip research-phase):
- **Phase 1:** Hjorth and spectral entropy are textbook DSP; implementation is pure numpy with verified formulas.
- **Phase 2:** sklearn Pipeline + GridSearchCV + model registry is a standard pattern; well-documented.

Phases needing attention during planning:
- **Phase 3:** EC/EO subject intersection needs validation against actual TDBRAIN participant counts -- number of subjects with both conditions is unknown until runtime. Plan for N reduction.
- **Phase 3:** groups array alignment after fusion needs explicit verification -- after intersection, len(groups) must equal X.shape[0].

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All libraries verified in environment except shap (not yet installed) |
| Features | HIGH (Hjorth, entropy) / MEDIUM (EC/EO) | Hjorth and entropy are standard; EC/EO gain unverified on this dataset |
| Architecture | HIGH | Based on direct inspection of existing codebase |
| Pitfalls | HIGH (leakage) / MEDIUM (EC/EO specifics) | Leakage pitfalls well-documented; EC/EO subject overlap unknown |

**Overall confidence:** HIGH

### Gaps to Address

- **EC/EO subject overlap:** Unknown how many subjects have both EO and EC recordings. If overlap <300, statistical power may be reduced. Validate before committing to fusion as primary approach.
- **shap compatibility:** shap ~0.46 with xgboost 3.2.0 not yet verified in this environment. Run pip install shap before Phase 4.
- **AUC target:** No explicit AUC target for v1.1 stated in research. Baseline is 0.796. Confirm whether goal is a specific threshold or best-effort improvement.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection (config.py, data_loader.py, feature_extractor.py, classifier.py, main.py)
- antropy 0.1.9 API verified in environment
- xgboost 3.2.0 + sklearn 1.8.0 verified in environment
- scikit-learn official docs -- Pipeline, SelectKBest, StratifiedGroupKFold

### Secondary (MEDIUM confidence)
- Frontiers in Neuroscience 2024 -- data leakage in EEG deep learning studies
- MDPI Sensors 2022 -- ADHD EEG feature systematic review (spectral entropy, Hjorth)
- Journal of Big Data 2024 -- SHAP-value feature selection strategies
- arxiv 2504.04664 -- multi-band spectral entropy for ADHD classification

### Tertiary (LOW confidence)
- shap ~0.46 + xgboost 3.2.0 compatibility -- documented but not yet verified in this environment
- EC/EO AUC gain estimate -- literature-supported but not validated on TDBRAIN specifically

---
*Research completed: 2026-02-23*
*Ready for roadmap: yes*
