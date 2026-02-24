---
phase: 02-preprocessing
plan: 01
subsystem: signal-processing
tags: [mne, eeg, bandpass-filter, epoching, artifact-rejection]

requires:
  - phase: 01-data-loading
    provides: load_raw() returning mne.io.Raw with preload=False

provides:
  - preprocess(raw) -> (Epochs, stats) with shape (n_epochs, 26, 1000)
  - 26-channel scalp EEG selection, 1-40 Hz bandpass, average re-reference, 2s epochs, 100uV rejection

affects: [03-features, 04-classification]

tech-stack:
  added: []
  patterns: [pick->filter->reref->epoch->drop_bad ordering enforced]

key-files:
  created: [preprocessor.py]
  modified: [config.py, main.py]

key-decisions:
  - "pick() before set_eeg_reference() — re-referencing before channel selection corrupts the average reference"
  - "REJECT_THRESHOLD=100e-6 (100 uV peak-to-peak) — verified ~8.5% rejection on real data, acceptable"

patterns-established:
  - "Preprocessing order: pick -> filter -> set_eeg_reference -> make_fixed_length_epochs -> drop_bad"

requirements-completed: [PREP-01, PREP-02, PREP-03]

duration: 5min
completed: 2026-02-23
---

# Phase 2 Plan 01: Preprocessing Summary

**MNE bandpass (1-40 Hz) + average re-reference + 2s fixed-length epochs + 100uV artifact rejection yielding (n_epochs, 26, 1000) Epochs**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-23T07:07:07Z
- **Completed:** 2026-02-23T07:12:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- preprocessor.py with preprocess() returning clean Epochs and rejection stats
- config.py extended with EEG_CHANNELS (26), EPOCH_DURATION, REJECT_THRESHOLD
- main.py validated end-to-end: 54/59 epochs kept (8.5% rejected), shape (54, 26, 1000)

## Task Commits

1. **Task 1: preprocessor.py + config.py preprocessing constants** - `ddfb4dc` (feat)
2. **Task 2: Update main.py to validate preprocessing on one subject** - `052a181` (feat)

## Files Created/Modified
- `preprocessor.py` - preprocess(raw) -> (Epochs, stats); enforces pick->filter->reref->epoch->drop_bad
- `config.py` - added EEG_CHANNELS (26 scalp channels), EPOCH_DURATION=2.0, REJECT_THRESHOLD=100e-6
- `main.py` - Phase 2 validation block printing epoch shape and rejection stats

## Decisions Made
- pick() before set_eeg_reference(): re-referencing all 33 channels then dropping 7 would corrupt the average reference
- 100 uV PTP threshold: research-verified; actual rejection was 8.5% on first subject (within acceptable range)

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- preprocess() is ready for Phase 3 feature extraction
- Epoch shape (n_epochs, 26, 1000) confirmed; 500 Hz * 2s = 1000 samples per epoch

---
*Phase: 02-preprocessing*
*Completed: 2026-02-23*
