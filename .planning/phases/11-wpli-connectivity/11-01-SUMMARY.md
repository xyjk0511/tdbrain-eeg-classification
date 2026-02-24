---
phase: 11-wpli-connectivity
plan: 01
subsystem: feature-extraction
tags: [wpli, connectivity, mne-connectivity, roi-averaging, caching]

requires:
  - phase: 03-feature-extraction
    provides: "feature_extractor.py pattern, config.py constants"
provides:
  - "extract_connectivity(epochs, sid, cond) -> 45-dim wPLI feature vector"
  - "ROI_GROUPS, CONN_BANDS, CONN_CACHE_DIR constants in config.py"
  - "Disk cache at cache_connectivity/{cond}/{sid}.npz"
affects: [13-connectivity-integration]

tech-stack:
  added: [mne-connectivity 0.7.0]
  patterns: [roi-averaging, disk-caching-npz]

key-files:
  created: [connectivity_extractor.py]
  modified: [config.py]

key-decisions:
  - "Use lower-triangle extraction from mne_connectivity 0.7+ full-matrix output (676 entries, not 325)"
  - "np.abs(wPLI) for magnitude-only values in [0,1]"
  - "combinations_with_replacement for 15 ROI pairs (10 inter + 5 intra)"

patterns-established:
  - "ROI averaging: channel-pair wPLI -> ROI-pair mean via ch_to_roi mapping"
  - "Connectivity caching: cache_connectivity/{cond}/{sid}.npz with conn key"

requirements-completed: [CON-01, CON-02, CON-03]

duration: 7min
completed: 2026-02-24
---

# Phase 11 Plan 01: wPLI Connectivity Extraction Summary

**wPLI connectivity extractor with 5-region ROI averaging (45 features) and npz disk caching via mne-connectivity 0.7**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-24T09:13:26Z
- **Completed:** 2026-02-24T09:20:09Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Config constants: ROI_GROUPS (5 regions, 26 channels), CONN_BANDS (3 bands), CONN_CACHE_DIR
- connectivity_extractor.py with extract_connectivity() returning (45,) vector
- Disk caching with cache-hit verification (identical results)
- Low-epoch warning for subjects with < 10 epochs

## Task Commits

Each task was committed atomically:

1. **Task 1: Add connectivity constants to config.py** - `1ce0bc4` (feat)
2. **Task 2: Create connectivity_extractor.py with wPLI + ROI averaging + caching** - `de7583b` (feat)

## Files Created/Modified
- `config.py` - Added ROI_GROUPS, CONN_BANDS, CONN_CACHE_DIR constants
- `connectivity_extractor.py` - wPLI extraction with ROI averaging and disk caching

## Decisions Made
- mne_connectivity 0.7+ returns full 26x26 matrix (676 entries) instead of upper-triangle (325); used lower-triangle extraction with np.tril_indices
- np.abs applied to wPLI values to get magnitude in [0,1] range
- combinations_with_replacement produces 15 ROI pairs including 5 intra-ROI self-pairs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mne_connectivity 0.7+ full-matrix output handling**
- **Found during:** Task 2 (connectivity extractor creation)
- **Issue:** Plan assumed get_data() returns (325, n_bands) upper-triangle pairs, but mne_connectivity 0.7.0 returns (676, n_bands) full matrix
- **Fix:** Reshape to (n_ch, n_ch, n_bands), use np.tril_indices(n_ch, k=-1) for lower-triangle extraction
- **Files modified:** connectivity_extractor.py
- **Verification:** extract_connectivity returns correct (45,) shape with values in [0.14, 0.57]
- **Committed in:** de7583b (Task 2 commit)

**2. [Rule 3 - Blocking] Installed missing mne-connectivity dependency**
- **Found during:** Task 2 (before verification)
- **Issue:** mne-connectivity not installed in environment
- **Fix:** pip install mne-connectivity (installed 0.7.0)
- **Verification:** Import succeeds, spectral_connectivity_epochs works
- **Committed in:** de7583b (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- extract_connectivity() ready for integration in Phase 13
- Cache directory structure established for both EO and EC conditions
- ROI_GROUPS and CONN_BANDS constants available for downstream use

---
*Phase: 11-wpli-connectivity*
*Completed: 2026-02-24*

## Self-Check: PASSED

All files and commits verified.
