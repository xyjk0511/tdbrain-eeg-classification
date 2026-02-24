---
phase: 01-data-loading
plan: 01
subsystem: data
tags: [mne, pandas, bids, eeg, brainvision]

requires: []
provides:
  - load_subjects() filters MDD/ADHD subjects from participants.tsv by condition
  - load_raw() loads .vhdr EEG files via MNE using BIDS path pattern
  - config.py centralizes DATASET_ROOT and CONDITION
affects: [02-preprocessing, 03-feature-extraction, 04-classification]

tech-stack:
  added: []
  patterns:
    - "BIDS path construction via pathlib.Path, pass str() to MNE"
    - "Condition-aware subject filtering via participants.tsv EC/EO columns"

key-files:
  created:
    - config.py
    - data_loader.py
    - main.py
  modified: []

key-decisions:
  - "Use indication column (not formal_status) for MDD/ADHD filtering"
  - "preload=False to avoid loading 520 subjects into memory"
  - "Pass str(path) to MNE to avoid Windows path issues"

patterns-established:
  - "Pattern 1: from config import DATASET_ROOT, CONDITION in all data modules"
  - "Pattern 2: load_subjects(condition) + load_raw(subject_id, condition) as standard data API"

requirements-completed: [DATA-01, DATA-02, DATA-03]

duration: 10min
completed: 2026-02-23
---

# Phase 1 Plan 1: Data Loading Summary

**pandas-based MDD/ADHD subject filtering from participants.tsv + MNE BrainVision loader with BIDS path construction, verified 520 subjects / 33ch / 500Hz / 120s**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-23T05:50:16Z
- **Completed:** 2026-02-23T06:00:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- config.py centralizes DATASET_ROOT, PARTICIPANTS_TSV, CONDITION for all downstream modules
- load_subjects() correctly filters MDD=320, ADHD=200 using indication column with EC/EO availability check
- load_raw() builds BIDS path and loads .vhdr via MNE, verified 33ch/500Hz/120s

## Task Commits

1. **Task 1: config.py and data_loader.py** - `0acf17c` (feat)
2. **Task 2: main.py validation script** - `d7339b7` (feat)

## Files Created/Modified

- `config.py` - DATASET_ROOT, PARTICIPANTS_TSV, CONDITION constants
- `data_loader.py` - load_subjects() and load_raw() functions
- `main.py` - end-to-end validation entry point

## Decisions Made

- Used `indication` column instead of `formal_status` (formal_status has 816 UNKNOWN entries)
- `preload=False` to avoid memory exhaustion across 520 subjects
- `str(path)` passed to MNE to handle Windows path compatibility

## Deviations from Plan

None - plan executed exactly as written.

Note: Research doc stated EO condition yields ADHD=199, but actual data shows all 200 ADHD subjects have EO=1. Implementation correctly reads from real data.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Data loading API is stable: load_subjects(condition) + load_raw(subject_id, condition)
- Phase 2 preprocessing can iterate subjects via load_subjects() and load each via load_raw()
- 33 channels include 7 non-EEG channels (EOG/misc); Phase 2 should pick EEG channels

---
*Phase: 01-data-loading*
*Completed: 2026-02-23*
