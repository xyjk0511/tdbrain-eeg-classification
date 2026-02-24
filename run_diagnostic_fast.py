"""Fast MDD vs nonMDD diagnostic — fixed hyperparams, no grid search, no permutation."""
import json, datetime, numpy as np, pandas as pd
from pathlib import Path
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import PARTICIPANTS_TSV, CONDITIONS, N_FEATURES_SELECT

CACHE_DIR = Path("cache_diagnostic")
RS = 42


def load_cached_features():
    """Load cached .npz features for all DISCOVERY subjects."""
    feats = {}
    for cond in CONDITIONS:
        feats[cond] = {}
        for npz in (CACHE_DIR / cond).glob("*.npz"):
            sid = npz.stem
            data = np.load(npz)
            feats[cond][sid] = (data["feat"], str(data["label"]))
    return feats


def build_pipes():
    def _s(): return ("smote", SMOTE(random_state=RS, k_neighbors=5))
    def _k(): return ("selector", SelectKBest(f_classif, k=N_FEATURES_SELECT))
    return {
        "svm": ImbPipeline([("scaler", StandardScaler()), _s(), _k(),
            ("clf", SVC(kernel="rbf", C=1, class_weight="balanced", probability=True, random_state=RS))]),
        "rf": ImbPipeline([("scaler", StandardScaler()), _s(), _k(),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS))]),
        "xgb": ImbPipeline([("scaler", StandardScaler()), _s(), _k(),
            ("clf", XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric="logloss", random_state=RS))]),
    }


if __name__ == "__main__":
    print("=== Fast MDD vs nonMDD Diagnostic ===")
    feats = load_cached_features()
    print(f"Cached: EO={len(feats['EO'])}, EC={len(feats['EC'])}")

    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    X = np.array([np.concatenate([feats[c][s][0] for c in CONDITIONS]) for s in common])
    y_str = np.array([feats[CONDITIONS[0]][s][1] for s in common])
    groups = np.array(common)

    # MDD=1 (positive), nonMDD=0 — avoid LabelEncoder alphabetical inversion
    y = np.array([1 if s == "MDD" else 0 for s in y_str])
    mdd_idx = 1
    counts = dict(zip(*np.unique(y_str, return_counts=True)))
    print(f"Subjects: {len(common)}, X: {X.shape}, labels: {counts}")

    cv = StratifiedGroupKFold(n_splits=5)
    pipes = build_pipes()
    results = {}

    for name, pipe in pipes.items():
        print(f"\n  {name}...", end=" ", flush=True)
        probas, preds, idxs = [], [], []
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            pipe.fit(X[tr], y[tr])
            probas.append(pipe.predict_proba(X[te])[:, mdd_idx])
            preds.append(pipe.predict(X[te]))
            idxs.append(te)
            print(f"f{fold}", end=" ", flush=True)

        order = np.argsort(np.concatenate(idxs))
        proba = np.concatenate(probas)[order]
        pred = np.concatenate(preds)[order]

        auc = roc_auc_score(y, proba)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ba = (sen + spe) / 2

        fpr, tpr, thr = roc_curve(y, proba, pos_label=mdd_idx)
        opt_t = float(thr[np.argmax(tpr - fpr)])
        pred_opt = np.where(proba >= opt_t, mdd_idx, 1 - mdd_idx)
        tn2, fp2, fn2, tp2 = confusion_matrix(y, pred_opt).ravel()
        sen2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
        spe2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0

        results[name] = {
            "auc": float(auc), "balanced_accuracy": float(ba),
            "sensitivity": float(sen), "specificity": float(spe),
            "optimal_threshold": opt_t,
            "sensitivity_at_threshold": float(sen2),
            "specificity_at_threshold": float(spe2),
        }
        print(f"AUC={auc:.3f} BA={ba:.3f} SEN={sen:.3f} SPE={spe:.3f}")

    # Ensemble
    print("\n  ensemble...", end=" ", flush=True)
    ens_probas, ens_idxs = [], []
    for tr, te in cv.split(X, y, groups):
        fold_p = []
        for pipe in build_pipes().values():
            pipe.fit(X[tr], y[tr])
            fold_p.append(pipe.predict_proba(X[te])[:, mdd_idx])
        ens_probas.append(np.mean(fold_p, axis=0))
        ens_idxs.append(te)
    ens_order = np.argsort(np.concatenate(ens_idxs))
    ens_proba = np.concatenate(ens_probas)[ens_order]
    ens_auc = float(roc_auc_score(y, ens_proba))
    efpr, etpr, ethr = roc_curve(y, ens_proba, pos_label=mdd_idx)
    ens_t = float(ethr[np.argmax(etpr - efpr)])
    ens_pred = np.where(ens_proba >= ens_t, mdd_idx, 1 - mdd_idx)
    tn_e, fp_e, fn_e, tp_e = confusion_matrix(y, ens_pred).ravel()
    ens_sen = float(tp_e / (tp_e + fn_e)) if (tp_e + fn_e) > 0 else 0.0
    ens_spe = float(tn_e / (tn_e + fp_e)) if (tn_e + fp_e) > 0 else 0.0
    ens_ba = (ens_sen + ens_spe) / 2
    results["ensemble"] = {
        "auc": ens_auc, "balanced_accuracy": float(ens_ba),
        "sensitivity": ens_sen, "specificity": ens_spe,
        "optimal_threshold": ens_t,
    }
    print(f"AUC={ens_auc:.3f} BA={ens_ba:.3f} SEN={ens_sen:.3f} SPE={ens_spe:.3f}")

    # Summary
    print("\n=== Results ===")
    for n, m in results.items():
        print(f"{n}: AUC={m['auc']:.3f}  BA={m['balanced_accuracy']:.3f}  "
              f"SEN={m['sensitivity']:.3f}  SPE={m['specificity']:.3f}")

    output = {
        "task": "MDD_vs_nonMDD_diagnostic_fast",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "models": results,
    }
    with open("results_diagnostic.json", "w", newline="\n", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\nSaved results_diagnostic.json")
