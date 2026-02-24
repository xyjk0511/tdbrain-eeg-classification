"""Feature engineering sweep: SelectKBest k scan + PCA comparison.
Uses SVM (best baseline model) with 5-fold StratifiedGroupKFold.
Reports BA/SEN/SPE for each config.
"""
import json, datetime, numpy as np
from pathlib import Path
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CONDITIONS

CACHE_DIR = Path("cache_diagnostic")
RS = 42
N_JOBS = 2
K_VALUES = [10, 20, 30, 50, 75, 100, 150, 200]
PCA_DIMS = [10, 20, 30, 50, 75, 100]


def load_cached_features():
    feats = {}
    for cond in CONDITIONS:
        feats[cond] = {}
        for npz in (CACHE_DIR / cond).glob("*.npz"):
            data = np.load(npz)
            feats[cond][npz.stem] = (data["feat"], str(data["label"]))
    return feats


def eval_pipeline(pipe, X, y, groups, cv):
    """Run OOF evaluation, return dict with AUC, BA, SEN, SPE."""
    auc = cross_val_score(pipe, X, y, groups=groups, cv=cv,
                          scoring="roc_auc", n_jobs=N_JOBS).mean()
    preds, idxs = [], []
    for tr, te in cv.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        preds.append(pipe.predict(X[te]))
        idxs.append(te)
    order = np.argsort(np.concatenate(idxs))
    pred = np.concatenate(preds)[order]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"auc": float(auc), "ba": float((sen + spe) / 2),
            "sen": float(sen), "spe": float(spe)}


if __name__ == "__main__":
    print("=== Feature Engineering Sweep ===")
    feats = load_cached_features()
    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    X = np.array([np.concatenate([feats[c][s][0] for c in CONDITIONS]) for s in common])
    y = np.array([1 if feats[CONDITIONS[0]][s][1] == "MDD" else 0 for s in common])
    groups = np.array(common)
    print(f"Subjects: {len(common)}, Features: {X.shape[1]}")

    cv = StratifiedGroupKFold(n_splits=5)
    results = []

    # SelectKBest sweep
    print("\n--- SelectKBest k sweep (SVM) ---")
    print(f"{'k':>5} {'AUC':>6} {'BA':>6} {'SEN':>6} {'SPE':>6}")
    for k in K_VALUES:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RS, k_neighbors=5)),
            ("selector", SelectKBest(f_classif, k=k)),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RS)),
        ])
        m = eval_pipeline(pipe, X, y, groups, cv)
        m["method"] = "SelectKBest"
        m["dim"] = k
        results.append(m)
        print(f"{k:>5} {m['auc']:>6.3f} {m['ba']:>6.3f} {m['sen']:>6.3f} {m['spe']:>6.3f}")

    # PCA sweep
    print("\n--- PCA sweep (SVM) ---")
    print(f"{'n':>5} {'AUC':>6} {'BA':>6} {'SEN':>6} {'SPE':>6}")
    for n in PCA_DIMS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RS, k_neighbors=5)),
            ("pca", PCA(n_components=n, random_state=RS)),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RS)),
        ])
        m = eval_pipeline(pipe, X, y, groups, cv)
        m["method"] = "PCA"
        m["dim"] = n
        results.append(m)
        print(f"{n:>5} {m['auc']:>6.3f} {m['ba']:>6.3f} {m['sen']:>6.3f} {m['spe']:>6.3f}")

    # Best config
    best = max(results, key=lambda r: r["ba"])
    print(f"\nBest: {best['method']} dim={best['dim']} "
          f"BA={best['ba']:.3f} AUC={best['auc']:.3f}")

    output = {
        "task": "feature_engineering_sweep",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": int(X.shape[0]),
        "baseline_ba": 0.613,
        "results": results,
    }
    with open("results_feat_sweep.json", "w", newline="\n", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("Saved results_feat_sweep.json")
