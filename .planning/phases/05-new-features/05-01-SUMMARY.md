---
phase: 05-new-features
plan: 01
subsystem: feature-extraction
tags: [eeg, hjorth, spectral-entropy, numpy, mne]

requires:
  - phase: 04-evaluation
    provides: extract_features() returning 262-dim vector with abs/rel band power, TBR, FAA
provides:
  - _extract_hjorth helper returning (78,) Activity/Mobility/Complexity per channel
  - _extract_spectral_entropy helper returning (156,) broadband + per-band SE
  - extract_features() returning (496,) feature vector
affects: [06-classifier, 07-fusion]

tech-stack:
  added: []
  patterns:
    - "Hjorth parameters computed from epochs.get_data(units='uV') — mandatory unit conversion"
    - "Spectral entropy normalized per-epoch so absolute PSD scale cancels"

key-files:
  created: []
  modified: [feature_extractor.py]

key-decisions:
  - "Use epochs.get_data(units='uV') for Hjorth — default Volts gives Activity ~1e-12, meaningless"
  - "Pass raw psds (V2/Hz) to _extract_spectral_entropy — normalization cancels scale, no unit conversion needed"
  - "Feature layout: [0:130] abs | [130:260] rel | [260:262] TBR+FAA | [262:340] Hjorth | [340:366] broadband SE | [366:496] per-band SE"

patterns-established:
  - "Private helpers prefixed with _ for sub-computations in feature_extractor.py"

requirements-completed: [FEAT-01, FEAT-02, FEAT-03, FEAT-04]

duration: 1min
completed: 2026-02-23
---

# Phase 5 Plan 01: New Features Summary

**Hjorth parameters (78 dims) and spectral entropy (156 dims) added via pure numpy, growing extract_features() from 262 to 496 dimensions**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-23T10:25:02Z
- **Completed:** 2026-02-23T10:25:50Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added `_extract_hjorth`: Activity/Mobility/Complexity per channel averaged over epochs, using mandatory `get_data(units="uV")`
- Added `_extract_spectral_entropy`: broadband (26) + per-band (5x26=130) Shannon entropy, reusing existing Welch psds
- Updated `extract_features()` to concatenate both helpers, returning shape (496,)

## Task Commits

1. **Task 1+2: Add Hjorth and spectral entropy, update extract_features** - `7fae5f4` (feat)

## Files Created/Modified
- `feature_extractor.py` - Added two private helpers and updated return to 496-dim vector

## Decisions Made
- Used `epochs.get_data(units="uV")` for Hjorth — without this, Activity is ~1e-12 (Volts), numerically useless
- Passed raw psds (V2/Hz) to `_extract_spectral_entropy` without unit conversion — normalization cancels absolute scale

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 496-dim feature vector ready for Phase 6 classifier work (SVM/RF/XGBoost with SelectKBest)
- No blockers

---
*Phase: 05-new-features*
*Completed: 2026-02-23*
