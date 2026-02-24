from pathlib import Path

DATASET_ROOT = Path(r"D:\eeg\data\tdbrain_dataset")
PARTICIPANTS_TSV = DATASET_ROOT / "participants.tsv"
CONDITIONS = ["EO", "EC"]

EEG_CHANNELS = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC3','FCz','FC4','T7','C3','Cz','C4','T8',
    'CP3','CPz','CP4','P7','P3','Pz','P4','P8',
    'O1','Oz','O2'
]
EPOCH_DURATION = 2.0       # seconds
REJECT_THRESHOLD = 150e-6  # peak-to-peak 150 uV

FREQ_BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 40),
}
FRONTAL_CHANNELS = ["F3", "F4", "Fz", "F7", "F8"]

N_FEATURES_SELECT = 50  # SelectKBest k inside Pipeline CV
