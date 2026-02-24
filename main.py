import numpy as np
import json
import datetime
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import CONDITIONS, N_FEATURES_SELECT
from data_loader import load_subjects, load_raw
from preprocessor import preprocess
from feature_extractor import extract_features
from classifier import classify

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6",
    "O1","O2","A1","A2","F9","F10","T9","T10","Iz"
]
BANDS = ["delta","theta","alpha","beta","gamma"]

def _feat_names_496():
    names = []
    for band in BANDS:
        for ch in CHANNELS:
            names.append(f"abs_{band}_{ch}")
    for band in BANDS:
        for ch in CHANNELS:
            names.append(f"rel_{band}_{ch}")
    names += ["TBR", "FAA"]
    for prefix in ("hjorth_act", "hjorth_mob", "hjorth_comp"):
        for ch in CHANNELS:
            names.append(f"{prefix}_{ch}")
    for ch in CHANNELS:
        names.append(f"SE_broad_{ch}")
    for band in BANDS:
        for ch in CHANNELS:
            names.append(f"SE_{band}_{ch}")
    return names

if __name__ == "__main__":
    feats = {}
    for cond in CONDITIONS:
        feats[cond] = {}
        subjects = load_subjects(cond)
        for _, row in subjects.iterrows():
            sid = row["participants_ID"]
            try:
                raw = load_raw(sid, cond)
                epochs, _ = preprocess(raw)
            except Exception as e:
                print(f"[SKIP] {sid} {cond}: {e}")
                continue
            if len(epochs) == 0:
                print(f"[SKIP] {sid} {cond}: 0 epochs")
                continue
            feats[cond][sid] = (extract_features(epochs), row["indication"])

    common = sorted(set(feats["EO"]) & set(feats["EC"]))
    print(f"Subjects with both EO and EC: {len(common)}")

    rows = [np.concatenate([feats[c][sid][0] for c in CONDITIONS]) for sid in common]
    X = np.array(rows)
    y = np.array([feats[CONDITIONS[0]][sid][1] for sid in common])
    groups = np.array(common)
    assert len(groups) == X.shape[0], "groups/X length mismatch after intersection"
    print(f"Fused X: {X.shape}, labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    results = classify(X, y, groups)
    for model_name, m in results.items():
        pval = m['permutation_pvalue']
        pval_str = f"{pval:.4f}" if pval is not None else "N/A"
        print(f"{model_name}: AUC={m['auc']:.3f}  ACC={m['accuracy']:.3f}  "
              f"SEN={m['sensitivity']:.3f}  SPE={m['specificity']:.3f}  "
              f"p={pval_str}")

    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "conditions": CONDITIONS,
        "n_subjects": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_counts": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "models": results,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("Saved results.json")

    # SHAP post-hoc analysis (OUT-02)
    feat_names = [f"EO_{n}" for n in _feat_names_496()] + [f"EC_{n}" for n in _feat_names_496()]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    mdd_idx = list(le.classes_).index("MDD")

    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=N_FEATURES_SELECT)),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ])
    final_pipe.fit(X, y_enc)

    X_sel = final_pipe[:-1].transform(X)
    sel_idx = final_pipe["selector"].get_support(indices=True)
    selected_names = [feat_names[i] for i in sel_idx]

    shap_values = shap.TreeExplainer(final_pipe["clf"]).shap_values(X_sel)
    # shap_values: list[ndarray] (old SHAP) or ndarray (n,f,c) (new SHAP)
    if isinstance(shap_values, list):
        shap_vals = shap_values[mdd_idx]
    else:
        shap_vals = shap_values[:, :, mdd_idx] if shap_values.ndim == 3 else shap_values
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    ranked = sorted(zip(selected_names, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)
    shap_top20 = [{"feature": n, "mean_abs_shap": v} for n, v in ranked[:20]]

    with open("shap_summary.json", "w", encoding="utf-8") as f:
        json.dump({"shap_top20": shap_top20}, f, indent=2)
    print("Top-5 SHAP features (MDD vs ADHD):")
    for e in shap_top20[:5]:
        print(f"  {e['feature']}: {e['mean_abs_shap']:.4f}")
    print("Saved shap_summary.json")
