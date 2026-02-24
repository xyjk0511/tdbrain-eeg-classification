# Phase 05 Verification: New Features

**Status:** PASSED  
**Verified:** 2026-02-23

## Requirements Verified

| Req | Description | Status | Evidence |
|-----|-------------|--------|----------|
| FEAT-01 | Hjorth parameters (78-dim) | PASS | `_extract_hjorth` returns (78,): act/mob/comp × 26 ch |
| FEAT-02 | Broadband spectral entropy (26-dim) | PASS | `_extract_spectral_entropy` se_broad (26,) |
| FEAT-03 | Per-band spectral entropy (130-dim) | PASS | 5 bands × 26 ch = 130 |
| FEAT-04 | Hjorth uses `get_data(units="uV")` | PASS | feature_extractor.py:25 |

## Feature Vector Layout

`extract_features()` → shape **(496,)**

| Slice | Content | Dims |
|-------|---------|------|
| [0:130] | abs band power | 130 |
| [130:260] | rel band power | 130 |
| [260:262] | TBR, FAA | 2 |
| [262:340] | Hjorth | 78 |
| [340:366] | broadband SE | 26 |
| [366:496] | per-band SE | 130 |

All FEAT-01–04 satisfied.
