import numpy as np
from config import FREQ_BANDS, FRONTAL_CHANNELS


def _extract_spectral_entropy(psds, freqs) -> np.ndarray:
    """Return (156,): 26 broadband SE + 5*26 per-band SE, averaged over epochs.

    psds: (n_epochs, 26, n_freqs) from compute_psd — units cancel after normalization.
    """
    p = psds / psds.sum(axis=-1, keepdims=True)
    se_broad = -(p * np.log(p + 1e-12)).sum(axis=-1).mean(0)  # (26,)

    se_bands = []
    for fmin, fmax in FREQ_BANDS.values():
        mask = (freqs >= fmin) & (freqs < fmax)
        pb = psds[:, :, mask]
        pb = pb / (pb.sum(axis=-1, keepdims=True) + 1e-12)
        se_bands.append(-(pb * np.log(pb + 1e-12)).sum(axis=-1).mean(0))  # (26,)

    return np.concatenate([se_broad, *se_bands])  # (156,)


def _extract_hjorth(epochs) -> np.ndarray:
    """Return (78,): Activity, Mobility, Complexity per channel, averaged over epochs."""
    data = epochs.get_data(units="uV")  # (n_epochs, 26, n_times) — must use uV not V
    act  = data.var(axis=-1)                                    # (n_epochs, 26)
    d1   = np.diff(data, axis=-1)
    mob  = np.sqrt(d1.var(axis=-1) / act)                       # (n_epochs, 26)
    d2   = np.diff(d1, axis=-1)
    comp = np.sqrt(d2.var(axis=-1) / d1.var(axis=-1)) / mob     # (n_epochs, 26)
    return np.concatenate([act.mean(0), mob.mean(0), comp.mean(0)])  # (78,)


def extract_features(epochs) -> np.ndarray:
    """Return 496-element feature vector: 130 abs + 130 rel band power + TBR + FAA + 78 Hjorth + 156 spectral entropy."""
    psd = epochs.compute_psd(method="welch", fmin=1.0, fmax=40.0, n_fft=500, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)  # (n_epochs, 26, n_freqs)

    ch_names = epochs.ch_names

    def band_mean(fmin, fmax):
        mask = (freqs >= fmin) & (freqs < fmax)
        return psds[:, :, mask].mean(axis=2).mean(axis=0)  # (26,)

    abs_bp = {b: band_mean(*FREQ_BANDS[b]) for b in FREQ_BANDS}
    total = sum(abs_bp.values())
    rel_bp = {b: abs_bp[b] / total for b in FREQ_BANDS}

    # TBR: theta/beta ratio over frontal channels only
    frontal_idx = [ch_names.index(c) for c in FRONTAL_CHANNELS]
    theta = psds[:, frontal_idx, :][:, :, (freqs >= 4) & (freqs < 8)].mean(axis=2)   # (n_epochs, 5)
    beta  = psds[:, frontal_idx, :][:, :, (freqs >= 13) & (freqs < 30)].mean(axis=2) # (n_epochs, 5)
    tbr = (theta.mean(axis=1) / beta.mean(axis=1)).mean()  # scalar

    # FAA: ln(alpha_F4) - ln(alpha_F3)
    f3_idx = ch_names.index("F3")
    f4_idx = ch_names.index("F4")
    alpha_mask = (freqs >= 8) & (freqs < 13)
    alpha_f3 = psds[:, f3_idx, :][:, alpha_mask].mean(axis=1)  # (n_epochs,)
    alpha_f4 = psds[:, f4_idx, :][:, alpha_mask].mean(axis=1)
    faa = (np.log(alpha_f4) - np.log(alpha_f3)).mean()  # scalar

    hjorth  = _extract_hjorth(epochs)
    entropy = _extract_spectral_entropy(psds, freqs)

    return np.concatenate(
        [*[abs_bp[b] for b in FREQ_BANDS],
         *[rel_bp[b] for b in FREQ_BANDS],
         [tbr, faa],
         hjorth,
         entropy]
    )  # shape (496,)
