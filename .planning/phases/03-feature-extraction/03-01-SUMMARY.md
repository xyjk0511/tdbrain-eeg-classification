---
phase: 03-feature-extraction
plan: 01
subsystem: feature-extraction
tags: [spectral, welch-psd, band-power, tbr, faa]
dependency_graph:
  requires: [02-01]
  provides: [X-matrix-262d]
  affects: [04-classification]
tech_stack:
  added: [mne-psd, numpy-concatenate]
  patterns: [welch-psd, band-mean, frontal-channel-index]
key_files:
  created: [feature_extractor.py]
  modified: [config.py, main.py]
decisions:
  - "mean over uniform 1 Hz bins instead of np.trapz"
  - "dynamic ch_names.index() for frontal channels"
metrics:
  duration: 8 min
  completed: 2026-02-23
---

# Phase 03 Plan 01: Spectral Feature Extraction Summary

Welch PSD feature extraction producing 262-element vectors (130 abs + 130 rel band power + TBR + FAA) for 492 of 500 subjects.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add FREQ_BANDS/FRONTAL_CHANNELS, create feature_extractor.py | 27d9ec8 | config.py, feature_extractor.py |
| 2 | Update main.py with full subject loop | a79f3d7 | main.py |

## Verification Results

- `python -c "import feature_extractor; print('import ok')"` — passed
- `python main.py` — `Feature matrix: X.shape=(492, 262), y.shape=(492,)`
- Labels: ADHD=185, MDD=307
- 8 subjects skipped (0 epochs after rejection) — no crash

## Decisions Made

- Mean over uniform 1 Hz bins (no np.trapz) — Welch bins are uniform so mean is equivalent
- Dynamic `ch_names.index()` for all channel lookups — no hardcoded indices

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED
