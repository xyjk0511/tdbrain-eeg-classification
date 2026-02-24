# Pitfalls Research

**Domain:** EEG psychiatric classification -- adding features and classifiers to existing pipeline
**Researched:** 2026-02-23
**Confidence:** HIGH (leakage pitfalls), MEDIUM (feature-specific), MEDIUM (EC/EO fusion)

---

## Critical Pitfalls

### Pitfall 1: Feature Selection Outside the CV Pipeline (Data Leakage)

**What goes wrong:**
SelectKBest or SHAP-guided feature selection is applied to the full dataset before cross-validation, or applied once to the outer training fold without being re-fit per inner fold. Test-fold samples influence which features are kept.

**Why it happens:**
It feels natural to "first select features, then cross-validate." The bug is invisible -- the code runs, produces a number, and the number looks plausible. With 262 features and 477 subjects, leakage from SelectKBest can inflate AUC by 0.05-0.15.

**How to avoid:**
Wrap SelectKBest inside sklearn Pipeline so it is re-fit only on X[train_idx] in each fold. The existing Pipeline([scaler, svc]) pattern in classifier.py must be extended to Pipeline([scaler, selector, clf]) -- the selector step must be inside the pipeline passed to GridSearchCV.

**Warning signs:**
- AUC improves dramatically (>0.05) when feature selection is added
- Selected feature set is identical across all outer folds
- selector.fit() is called before the CV loop

**Phase to address:** Feature selection implementation (v1.1)

---

### Pitfall 2: SHAP Computed on Full Dataset or Outer Test Fold

**What goes wrong:**
SHAP values are computed after fitting a model on all data, then used to select features, then the same data is used to evaluate the model. In a nested CV context this leaks test-fold information into feature selection.

**Why it happens:**
SHAP is typically used for post-hoc explanation, not selection. Developers compute it once on the full trained model and use the ranking globally.

**How to avoid:**
Use SHAP only for post-hoc interpretation after the final nested CV loop completes. For actual feature selection inside CV, use SelectKBest (f_classif or mutual_info_classif) inside the pipeline.

**Warning signs:**
- SHAP is computed once outside any CV loop
- Feature mask derived from SHAP is applied before outer_cv.split()

**Phase to address:** Feature selection implementation (v1.1)

---

### Pitfall 3: EC/EO Fusion Imputation Leaks Test-Fold Statistics

**What goes wrong:**
Some TDBRAIN subjects have only EC or only EO recordings. Missing condition features are filled using statistics computed across all subjects including those in the test fold. df.fillna(df.mean()) or SimpleImputer called on full X before the CV loop is the exact failure mode.

**Why it happens:**
The current pipeline loads one condition at a time (CONDITION in config.py). The merge itself is safe. The imputation for missing conditions is where leakage enters.

**How to avoid:**
Move imputation inside the CV loop: compute fill values from X[train_idx] only, then apply to X[test_idx]. Use SimpleImputer inside the pipeline. Alternatively, exclude subjects missing either condition.

**Warning signs:**
- df.fillna(df.mean()) or SimpleImputer.fit_transform(X) called on full X before CV
- AUC for fused model is much higher than either condition alone without a plausible explanation

**Phase to address:** EC/EO fusion implementation (v1.1)

---

### Pitfall 4: Permutation Test Pipeline Does Not Match Real Pipeline

**What goes wrong:**
The existing classifier.py permutation test uses a fixed_pipe (no inner GridSearchCV) for speed. When adding feature selection, the permutation test must also include the selector step. If fixed_pipe is [scaler, clf] but the real pipeline is [scaler, selector, clf], the null distribution is generated without feature selection while the observed AUC was computed with it. The test becomes anti-conservative (p-value too small).

**Why it happens:**
The permutation test was deliberately simplified. Adding a new pipeline step without updating the permutation test is easy to miss.

**How to avoid:**
After any pipeline change, verify fixed_pipe step count == real pipeline step count. The selector must be present in fixed_pipe and will re-fit on permuted labels in each fold, which is correct behavior under H0.

**Warning signs:**
- fixed_pipe in the permutation loop has fewer steps than the main pipeline
- p-value drops to 0.001 immediately after adding feature selection (suspiciously low)

**Phase to address:** Any phase that modifies the pipeline structure

---

### Pitfall 5: Hjorth Activity Unit Error (V2 vs uV2)

**What goes wrong:**
Hjorth Activity = var(signal). MNE stores EEG data internally in Volts. epochs.get_data() returns Volts unless units are specified. Activity will be ~1e-12 instead of 10-500. When concatenated with spectral features, StandardScaler normalizes both -- but the scale error makes Hjorth Activity meaningless as a feature.

**Why it happens:**
The Hjorth formula has no unit awareness. The existing feature_extractor.py uses epochs.compute_psd() which handles units internally, but a new Hjorth extractor calling epochs.get_data() directly will get Volts by default.

**How to avoid:**
Extract data with explicit unit conversion: epochs.get_data(units="uV") or multiply by 1e6. Verify: mean Hjorth Activity across channels should be 10-500 uV2 for resting EEG.

**Warning signs:**
- Hjorth Activity values are < 1e-6 (Volts2) or > 1e6 (wrong scaling)
- Feature importance shows Hjorth Activity near-zero or dominating all other features by 10x after scaling

**Phase to address:** Hjorth feature extraction (v1.1)

---

### Pitfall 6: Spectral Entropy Not Normalized to Probability Distribution

**What goes wrong:**
Spectral entropy requires PSD normalized to sum=1 before computing H = -sum(p*log(p)). Raw PSD values (uV2/Hz) passed directly produce unbounded results dominated by absolute power, making spectral entropy redundant with total band power.

**Why it happens:**
The formula looks like Shannon entropy but the normalization step is easy to omit in manual implementations.

**How to avoid:**
Normalize per epoch per channel: p = psd / psd.sum(axis=-1, keepdims=True). If using antropy.spectral_entropy, pass normalize=True. Verify: values should be in [0, log(n_freqs)] and higher for broadband than narrowband signals.

**Warning signs:**
- Spectral entropy values are negative or greater than log(n_freqs)
- Spectral entropy correlates >0.95 with total band power across subjects

**Phase to address:** Entropy feature extraction (v1.1)

---

### Pitfall 7: XGBoost/RF Overfitting on Small N with Default Parameters

**What goes wrong:**
With ~300 training subjects per outer fold and 262+ features, XGBoost with max_depth=6 and n_estimators=100 (defaults) memorizes the training set. GridSearchCV selects an overfit configuration that looks good on inner folds but degrades on outer folds.

**Why it happens:**
Default XGBoost/RF parameters are tuned for large datasets. On 300 samples with hundreds of features, these defaults overfit.

**How to avoid:**
Inner CV grid for XGBoost must include: max_depth [2,3,4], n_estimators [50,100,200], subsample [0.7,1.0], colsample_bytree [0.7,1.0]. For RF: max_depth [None,5,10], max_features [sqrt, 0.3]. Compare outer-fold AUC variance -- std > 0.08 indicates overfitting.

**Warning signs:**
- Inner CV AUC >> outer CV AUC by more than 0.05
- Best max_depth from inner CV is always the maximum value in the grid
- Outer fold AUC std > 0.08

**Phase to address:** Classifier comparison (v1.1)

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|------------------|
| Fit SelectKBest once on full data | Faster, simpler code | Leakage inflates AUC; results not reproducible | Never |
| Compute SHAP globally for feature ranking | Easy to implement | Leakage if used for selection | Only for post-hoc explanation after CV, never for selection |
| Skip permutation test update when adding selector | Saves ~30 min runtime | Null distribution no longer matches observed pipeline; p-values invalid | Never |
| df.fillna(df.mean()) on full X before CV | One line of code | Leaks test-fold statistics | Never |
| Use same epoch length for Hjorth without checking | Reuses existing epochs | Hjorth Mobility/Complexity sensitive to epoch length | Only if epoch length is already >=2s |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| SelectKBest + GridSearchCV | Pass k as hyperparameter without wrapping selector in pipeline | Pipeline([sel, clf]) and tune sel__k in param_grid |
| XGBoost + StratifiedGroupKFold | Using xgb.cv which does not support group-based splits | Use XGBClassifier inside GridSearchCV with StratifiedGroupKFold |
| EC/EO merge + groups array | After merging into one row per subject, groups array length no longer matches X rows | After fusion each row is one subject; groups = subject_ids directly |
| Hjorth via antropy + MNE epochs | Calling antropy.hjorth_params on MNE epoch object directly | Extract first: data = epochs.get_data(units="uV"), apply per channel |
| SHAP + Pipeline with selector | SHAP feature indices refer to post-selection features, not original names | Map indices back via selector.get_support() |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Hjorth computed in pure Python loop per epoch per channel | Feature extraction >10 min for 477 subjects | Vectorize with np.diff and np.var over time axis | At >200 subjects |
| SHAP on full model with 262+ features per fold | SHAP computation >5 min per fold | Use shap.TreeExplainer (fast for trees); limit to post-CV explanation only | At >100 features |
| Full nested CV inside 1000-permutation loop | Runtime >8 hours | Keep permutation test using fixed_pipe without inner tuning (current approach is correct) | Always |

## "Looks Done But Isn't" Checklist

- [ ] **SelectKBest inside pipeline:** selector.fit() is never called outside a CV fold
- [ ] **Permutation test pipeline matches real pipeline:** fixed_pipe contains selector step after adding it to main pipeline
- [ ] **EC/EO imputation is fold-local:** No fillna or SimpleImputer on full X before CV split
- [ ] **Hjorth units verified:** Mean Hjorth Activity per channel is 10-500 uV2; if >10000 the signal is not in uV
- [ ] **Spectral entropy range valid:** All values satisfy 0 <= entropy <= log(n_freqs)
- [ ] **XGBoost grid includes shallow trees:** max_depth grid starts at 2 or 3, not 6
- [ ] **groups array length matches X rows:** After EC/EO fusion, len(groups) == X.shape[0]

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Feature selection leakage discovered after results reported | HIGH | Re-run full pipeline with selector inside pipeline; expect AUC to drop; re-run permutation test |
| Hjorth unit error | LOW | Fix unit conversion in extract_features, re-run feature extraction only |
| Spectral entropy not normalized | LOW | Fix normalization, re-run feature extraction |
| EC/EO imputation leakage | MEDIUM | Move imputation inside CV loop, re-run full pipeline |
| Permutation test pipeline mismatch | MEDIUM | Update fixed_pipe to match real pipeline, re-run 1000 permutations (~30 min) |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Feature selection leakage | Feature selection implementation | Assert selected feature indices differ across outer folds |
| SHAP leakage | Feature selection implementation | SHAP used only after final CV loop, never inside |
| EC/EO imputation leakage | EC/EO fusion implementation | No fillna/SimpleImputer on full X before CV split |
| Hjorth unit error | Hjorth feature extraction | Print mean Activity per channel; assert 10-500 uV2 |
| Spectral entropy not normalized | Entropy feature extraction | Assert entropy in [0, log(n_freqs)] for all samples |
| Permutation test pipeline mismatch | Any phase modifying pipeline | fixed_pipe step count == real pipeline step count |
| XGBoost overfitting on small N | Classifier comparison | Outer-fold AUC std < 0.08; inner AUC - outer AUC < 0.05 |

## Sources

- [Data leakage in deep learning studies of translational EEG -- Frontiers in Neuroscience 2024](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1373515/full)
- [scikit-learn Common Pitfalls -- official docs](https://scikit-learn.org/stable/common_pitfalls.html)
- [Explanations of ML Models in Repeated Nested Cross-Validation -- MDPI 2022](https://www.mdpi.com/2076-3417/12/13/6681)
- [antropy.hjorth_params documentation](https://raphaelvallat.com/entropy/build/html/generated/entropy.hjorth_params.html)
- [antropy.spectral_entropy documentation](https://raphaelvallat.com/entropy/build/html/generated/entropy.spectral_entropy.html)
- [The Impact of Parameter Choices on EEG Entropy Measures -- Sapien Labs](http://sapienlabs.org/the-impact-of-parameters-choices-on-eeg-entropy-measures/)
- [Advancing biomedical engineering: Leveraging Hjorth features for EEG -- PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10750318/)
- [Feature selection strategies: SHAP-value and importance-based methods -- Journal of Big Data 2024](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00905-w)

---
*Pitfalls research for: EEG MDD/ADHD classification -- v1.1 feature and classifier additions*
*Researched: 2026-02-23*
