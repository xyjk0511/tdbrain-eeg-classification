import mne
import pandas as pd
from config import DATASET_ROOT, PARTICIPANTS_TSV


def load_subjects(condition: str) -> pd.DataFrame:
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    mask = df["indication"].isin(["MDD", "ADHD"]) & (df[condition] == 1)
    
    # Exclude know bad subjects with corrupted/short recordings for specific conditions
    if condition == "EO":
        mask = mask & ~(df["participants_ID"] == "sub-88049673")
        
    return (df[mask][["participants_ID", "indication"]]
            .drop_duplicates(subset=["participants_ID"])
            .reset_index(drop=True))


def load_raw(subject_id: str, condition: str) -> mne.io.Raw:
    vhdr = (DATASET_ROOT / subject_id / "ses-1" / "eeg"
            / f"{subject_id}_ses-1_task-rest{condition}_eeg.vhdr")
    if not vhdr.exists():
        raise FileNotFoundError(f"EEG file not found: {vhdr}")
    return mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
