"""MDD vs nonMDD diagnostic — light GridSearchCV + permutation test, n_jobs=2."""
import json, datetime, numpy as np
from pathlib import Path
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import CONDITIONS, N_FEATURES_SELECT

CACHE_DIR = Path("cache_diagnostic")
RS = 42
N_JOBS = 2
N_PERM = 10


def load_cached_features():
    feats = {}
    for cond in CONDITIONS:
        feats[cond] = {}
        for npz in (CACHE_DIR / cond).glob("*.npz"):
            data = np.load(npz)
            feats[cond][npz.stem] = (data["feat"], str(data["label"]))
    return feats


def _s(): return ("smote", SMOTE(random_state=RS, k_neighbors=5))
def _k(): return ("selector", SelectKBest(f_classif, k=N_FEATURES_SELECT))


FIXED_PIPES = {
    "svm": ImbPipeline([("scaler", StandardScaler()), _s(), _k(),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RS))]),
    "rf": ImbPipeline([("scaler", StandardScaler()), _s(), _k(),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RS))]),
    "xgb": ImbPipeline([("scaler", StandardScaler()), _s(), _k(),
        ("clf", XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=RS))]),
}


if __name__ == "__main__":
    print("=== MDD vs nonMDD Diagnostic (GridSearchCV + Perm) ===")
    feats = load_cached_features()
    print(f"Cached: EO={len(feats['EO'])}, EC={len(feats['EC'])}")

    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    X = np.array([np.concatenate([feats[c][s][0] for c in CONDITIONS]) for s in common])
    y_str = np.array([feats[CONDITIONS[0]][s][1] for s in common])
    groups = np.array(common)

    # MDD=1 (positive), nonMDD=0
    y = np.array([1 if s == "MDD" else 0 for s in y_str])
    mdd_idx = 1
    counts = {"MDD": int((y == 1).sum()), "nonMDD": int((y == 0).sum())}
    print(f"Subjects: {len(common)}, X: {X.shape}, labels: {counts}")

    outer_cv = StratifiedGroupKFold(n_splits=5)

    # Permutation setup
    rng = np.random.RandomState(RS)
    unique_groups = np.unique(groups)
    g_label = {g: y[groups == g][0] for g in unique_groups}
    g_vals = np.array([g_label[g] for g in unique_groups])

    def _perm_y():
        pm = dict(zip(unique_groups, rng.permutation(g_vals)))
        return np.array([pm[g] for g in groups])

    results = {}
    for name in FIXED_PIPES:
        pipe = FIXED_PIPES[name]
        print(f"\n  {name}...", end=" ", flush=True)
        # Real score: same FIXED_PIPES used for permutation (consistent)
        real_auc = cross_val_score(pipe, X, y, groups=groups,
                                   cv=outer_cv, scoring="roc_auc", n_jobs=N_JOBS).mean()
        # OOF predictions for SEN/SPE/BA (default threshold)
        probas, preds, idxs = [], [], []
        for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups)):
            pipe.fit(X[tr], y[tr])
            probas.append(pipe.predict_proba(X[te])[:, mdd_idx])
            preds.append(pipe.predict(X[te]))
            idxs.append(te)
            print(f"f{fold}", end=" ", flush=True)
        order = np.argsort(np.concatenate(idxs))
        pred = np.concatenate(preds)[order]
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ba = (sen + spe) / 2

        # Permutation test (same pipeline as real score)
        print("perm", end="", flush=True)
        perm_scores = []
        for _ in range(N_PERM):
            s = cross_val_score(pipe, X, _perm_y(), groups=groups,
                                cv=outer_cv, scoring="roc_auc", n_jobs=N_JOBS).mean()
            perm_scores.append(s)
            print(".", end="", flush=True)
        pvalue = (np.sum(np.array(perm_scores) >= real_auc) + 1) / (N_PERM + 1)

        results[name] = {
            "auc": float(real_auc), "balanced_accuracy": float(ba),
            "sensitivity": float(sen), "specificity": float(spe),
            "permutation_pvalue": float(pvalue),
        }
        print(f" AUC={real_auc:.3f} BA={ba:.3f} SEN={sen:.3f} SPE={spe:.3f} p={pvalue:.3f}")

    # Ensemble (FIXED_PIPES, default 0.5 threshold — same caliber as single models)
    print("\n  ensemble...", end=" ", flush=True)
    ens_probas, ens_idxs = [], []
    for tr, te in outer_cv.split(X, y, groups):
        fold_p = []
        for pipe in FIXED_PIPES.values():
            pipe.fit(X[tr], y[tr])
            fold_p.append(pipe.predict_proba(X[te])[:, mdd_idx])
        ens_probas.append(np.mean(fold_p, axis=0))
        ens_idxs.append(te)
    ens_order = np.argsort(np.concatenate(ens_idxs))
    ens_proba = np.concatenate(ens_probas)[ens_order]
    ens_auc = float(roc_auc_score(y, ens_proba))
    ens_pred = (ens_proba >= 0.5).astype(int)
    tn_e, fp_e, fn_e, tp_e = confusion_matrix(y, ens_pred).ravel()
    ens_sen = float(tp_e / (tp_e + fn_e)) if (tp_e + fn_e) > 0 else 0.0
    ens_spe = float(tn_e / (tn_e + fp_e)) if (tn_e + fp_e) > 0 else 0.0
    ens_ba = (ens_sen + ens_spe) / 2
    results["ensemble"] = {
        "auc": ens_auc, "balanced_accuracy": float(ens_ba),
        "sensitivity": ens_sen, "specificity": ens_spe,
        "permutation_pvalue": None,
    }
    print(f"AUC={ens_auc:.3f} BA={ens_ba:.3f} SEN={ens_sen:.3f} SPE={ens_spe:.3f}")

    print("\n=== Results ===")
    for n, m in results.items():
        pv = m.get('permutation_pvalue')
        ps = f"p={pv:.3f}" if pv is not None else "p=N/A"
        print(f"{n}: AUC={m['auc']:.3f}  BA={m['balanced_accuracy']:.3f}  "
              f"SEN={m['sensitivity']:.3f}  SPE={m['specificity']:.3f}  {ps}")

    output = {
        "task": "MDD_vs_nonMDD_diagnostic",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": int(X.shape[0]), "n_features": int(X.shape[1]),
        "label_counts": counts, "models": results,
    }
    with open("results_diagnostic.json", "w", newline="\n", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\nSaved results_diagnostic.json")
