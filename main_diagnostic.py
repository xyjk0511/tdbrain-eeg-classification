"""MDD vs nonMDD diagnostic classification on all DISCOVERY subjects.
Standalone script — does not modify existing pipeline files.
Features are cached to disk for fast re-runs.
"""
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATASET_ROOT, PARTICIPANTS_TSV, CONDITIONS
from data_loader import load_raw
from preprocessor import preprocess
from feature_extractor import extract_features
from classifier import classify

CACHE_DIR = Path("cache_diagnostic")


def load_discovery_subjects(condition: str) -> pd.DataFrame:
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    mask = (df["DISC/REP"] == "DISCOVERY") & (df[condition] == 1)
    if condition == "EO":
        mask = mask & ~(df["participants_ID"] == "sub-88049673")
    out = (df[mask][["participants_ID", "indication"]]
           .drop_duplicates(subset=["participants_ID"])
           .reset_index(drop=True))
    out["label"] = out["indication"].apply(lambda x: "MDD" if x == "MDD" else "nonMDD")
    return out


def extract_or_load(condition: str) -> dict:
    """Extract features for all DISCOVERY subjects, with per-subject caching."""
    cache = CACHE_DIR / condition
    cache.mkdir(parents=True, exist_ok=True)

    subjects = load_discovery_subjects(condition)
    total = len(subjects)
    feats = {}

    for i, (_, row) in enumerate(subjects.iterrows()):
        sid = row["participants_ID"]
        label = row["label"]
        npz = cache / f"{sid}.npz"

        if npz.exists():
            data = np.load(npz)
            feats[sid] = (data["feat"], str(data["label"]))
            if (i + 1) % 200 == 0:
                print(f"  [{condition}] {i+1}/{total} (cached)")
            continue

        try:
            raw = load_raw(sid, condition)
            epochs, _ = preprocess(raw)
        except Exception as e:
            print(f"[SKIP] {sid} {condition}: {e}")
            continue
        if len(epochs) == 0:
            print(f"[SKIP] {sid} {condition}: 0 epochs")
            continue

        feat = extract_features(epochs)
        np.savez_compressed(npz, feat=feat, label=np.array(label))
        feats[sid] = (feat, label)

        if (i + 1) % 50 == 0:
            print(f"  [{condition}] {i+1}/{total} extracted")

    print(f"  [{condition}] done: {len(feats)}/{total}")
    return feats


if __name__ == "__main__":
    print("=== MDD vs nonMDD Diagnostic Classification ===")

    # 1. Extract features
    feats = {}
    for cond in CONDITIONS:
        print(f"Loading {cond}...")
        feats[cond] = extract_or_load(cond)

    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    print(f"\nSubjects with both EO+EC: {len(common)}")

    X = np.array([np.concatenate([feats[c][sid][0] for c in CONDITIONS]) for sid in common])
    y = np.array([feats[CONDITIONS[0]][sid][1] for sid in common])
    groups = np.array(common)

    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"X: {X.shape}, labels: {counts}")

    # 2. Classify (reduced permutations for quick first pass)
    print("\nRunning classification (10 permutations)...")
    results = classify(X, y, groups, n_permutations=10)

    # 3. Print results
    print("\n=== Results ===")
    for name, m in results.items():
        ba = (m['sensitivity'] + m['specificity']) / 2
        pval = m['permutation_pvalue']
        pval_str = f"{pval:.4f}" if pval is not None else "N/A"
        print(f"{name}: AUC={m['auc']:.3f}  BA={ba:.3f}  "
              f"SEN={m['sensitivity']:.3f}  SPE={m['specificity']:.3f}  p={pval_str}")

    # 4. Save
    output = {
        "task": "MDD_vs_nonMDD_diagnostic",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "models": results,
    }
    with open("results_diagnostic.json", "w", newline="\n", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\nSaved results_diagnostic.json")
