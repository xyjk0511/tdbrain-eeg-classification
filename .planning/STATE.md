# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-24)

**Core value:** 用 stacking + 功能连接特征突破 AUC=0.798 瓶颈
**Current focus:** v2.0 — Stacking + 功能连接

## Current Position

Phase: 12 — Manual Stacking Ensemble
Plan: 0 of 1
Status: Ready to execute
Last activity: 2026-02-24 — Phase 11 complete, Phase 12 planned and ready

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 11 min
- Total execution time: 1.47 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-loading | 1 | 10 min | 10 min |
| 02-preprocessing | 1 | 5 min | 5 min |
| 03-feature-extraction | 1 | 8 min | 8 min |
| 04-classification-output | 1 | 20 min | 20 min |
| 06-classifier-registry | 1 | 2 min | 2 min |
| 09-data-quality-class-balance | 1 | 35 min | 35 min |
| 11-wpli-connectivity | 1 | 7 min | 7 min |

## Accumulated Context

### Decisions

- Use `indication` column (not `formal_status`) for MDD/ADHD filtering — formal_status has 816 UNKNOWN entries
- `preload=False` in load_raw() — avoids memory exhaustion across 520 subjects
- Pass `str(path)` to MNE — handles Windows path compatibility
- pick() before set_eeg_reference() — re-referencing before channel selection corrupts the average reference
- REJECT_THRESHOLD=100e-6 (100 uV PTP) — verified ~8.5% rejection on real data, acceptable
- Mean over uniform 1 Hz bins (no np.trapz) — Welch bins are uniform so mean is equivalent
- Dynamic ch_names.index() for frontal channels — no hardcoded indices
- StandardScaler inside Pipeline during CV — prevents leakage from global scaling
- Use StratifiedGroupKFold with subject IDs as groups — keeps subject isolation per fold
- Use epochs.get_data(units='uV') for Hjorth — default Volts gives Activity ~1e-12, meaningless
- Pass raw psds (V2/Hz) to _extract_spectral_entropy — normalization cancels scale, no unit conversion needed
- Feature layout: [0:130] abs | [130:260] rel | [260:262] TBR+FAA | [262:340] Hjorth | [340:366] broadband SE | [366:496] per-band SE
- SelectKBest inside Pipeline (not outside CV) — prevents feature selection leakage
- XGBoost max_depth=[2,3,4] — default 6 overfits on ~300 training samples
- _selector() factory function — avoids repeating SelectKBest(...) across 6 Pipeline definitions
- Intersection-only subject selection for EC/EO fusion — no imputation, drop subjects missing either condition
- CONDITIONS list in config.py replaces single CONDITION constant
- REJECT_THRESHOLD raised to 150e-6 (150uV) — recovers 37 subjects vs 100uV, n_subjects=493
- SMOTE inside ImbPipeline between scaler and selector — oversampling on training folds only, no leakage
- k_neighbors=5 for SMOTE — default, safe for ~160 ADHD training samples per fold
- mne_connectivity 0.7+ returns full 26x26 matrix (676), not upper-tri (325) — use np.tril_indices for lower-triangle extraction
- np.abs(wPLI) for magnitude-only connectivity values in [0,1]
- combinations_with_replacement for 15 ROI pairs (10 inter + 5 intra-ROI)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-24
Stopped at: Completed 11-01-PLAN.md (wPLI connectivity extraction)
Resume file: None
