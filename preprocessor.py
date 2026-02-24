import mne
from config import EEG_CHANNELS, EPOCH_DURATION, REJECT_THRESHOLD


def preprocess(raw: mne.io.Raw) -> tuple:
    """Filter, re-reference, epoch, and reject artifacts.
    Returns (Epochs, {'n_before': int, 'n_after': int}).
    Epoch shape: (n_epochs, 26, 1000)
    """
    raw.load_data(verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION, preload=True, verbose=False)
    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=REJECT_THRESHOLD), verbose=False)
    return epochs, {'n_before': n_before, 'n_after': len(epochs)}
