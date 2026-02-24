"""wPLI connectivity feature extractor with ROI averaging and disk caching."""

import hashlib
import warnings
import numpy as np
from itertools import combinations_with_replacement
from mne_connectivity import spectral_connectivity_epochs

from config import EEG_CHANNELS, ROI_GROUPS, CONN_BANDS, CONN_CACHE_DIR


def _config_hash() -> str:
    """Deterministic hash of connectivity config for cache invalidation."""
    key = repr((EEG_CHANNELS, dict(ROI_GROUPS), dict(CONN_BANDS)))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def extract_connectivity(epochs, sid: str, cond: str) -> np.ndarray:
    """Extract wPLI connectivity features averaged over ROI pairs.

    Returns a 45-dim vector: 15 ROI pairs x 3 frequency bands.
    Results are cached to disk at cache_connectivity/{cond}/{sid}.npz.
    """
    cache_path = CONN_CACHE_DIR / cond / f"{sid}.npz"

    # Cache hit — return if config hash matches
    cfg_hash = _config_hash()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if cached.get("cfg_hash", np.array("")).item() == cfg_hash:
            return cached["conn"]

    # Warn on low epoch count
    if len(epochs) < 10:
        warnings.warn(
            f"Subject {sid} ({cond}): only {len(epochs)} epochs, "
            "connectivity estimates may be unreliable",
            stacklevel=2,
        )

    # Enforce canonical channel order before computing connectivity
    epochs = epochs.reorder_channels(EEG_CHANNELS)

    n_ch = len(EEG_CHANNELS)
    fmin = tuple(f[0] for f in CONN_BANDS.values())
    fmax = tuple(f[1] for f in CONN_BANDS.values())

    # Compute wPLI
    con = spectral_connectivity_epochs(
        epochs,
        method="wpli",
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False,
    )
    # mne_connectivity 0.7+ returns raveled full matrix (n_ch^2, n_bands)
    # Reshape to (n_ch, n_ch, n_bands) and use lower-triangle (where values live)
    raw_data = np.abs(con.get_data())  # absolute value — magnitude only
    n_bands = len(CONN_BANDS)
    wpli_mat = raw_data.reshape(n_ch, n_ch, n_bands)

    # Build channel-to-ROI mapping
    ch_to_roi = {}
    for roi, channels in ROI_GROUPS.items():
        for ch in channels:
            ch_to_roi[ch] = roi

    # Lower-triangle indices (k=-1 excludes diagonal)
    row_idx, col_idx = np.tril_indices(n_ch, k=-1)

    # Canonical ROI pairs (15 = 10 inter + 5 intra)
    roi_pairs = list(combinations_with_replacement(ROI_GROUPS.keys(), 2))

    # Average wPLI per ROI pair per band
    result = np.zeros(len(roi_pairs) * n_bands)

    for rp_idx, (roi_a, roi_b) in enumerate(roi_pairs):
        # Find channel-pair indices belonging to this ROI pair
        mask = np.zeros(len(row_idx), dtype=bool)
        for i, (r, c) in enumerate(zip(row_idx, col_idx)):
            ch_r_roi = ch_to_roi[EEG_CHANNELS[r]]
            ch_c_roi = ch_to_roi[EEG_CHANNELS[c]]
            # Normalize to canonical order (sorted tuple)
            pair = tuple(sorted([ch_r_roi, ch_c_roi]))
            if pair == tuple(sorted([roi_a, roi_b])):
                mask[i] = True

        if mask.any():
            # Extract lower-tri values for matching pairs
            vals = wpli_mat[row_idx[mask], col_idx[mask], :]  # (n_match, n_bands)
            for b_idx in range(n_bands):
                result[rp_idx * n_bands + b_idx] = vals[:, b_idx].mean()

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, conn=result, cfg_hash=np.array(cfg_hash))

    return result
