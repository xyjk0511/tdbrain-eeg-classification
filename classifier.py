import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import N_FEATURES_SELECT


def classify(X, y, groups, n_splits=5, n_permutations=1000, random_state=42) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if "MDD" not in le.classes_:
        raise ValueError("Label 'MDD' not found in y.")

    mdd_idx = list(le.classes_).index("MDD")
    rs = random_state

    def _selector():
        return ("selector", SelectKBest(f_classif, k=N_FEATURES_SELECT))

    def _smote():
        return ("smote", SMOTE(random_state=rs, k_neighbors=5))

    MODEL_REGISTRY = {
        "svm": (
            ImbPipeline([("scaler", StandardScaler()), _smote(), _selector(), ("clf", SVC(kernel="rbf", probability=True, random_state=rs))]),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", "auto"], "clf__class_weight": [None, "balanced"]},
        ),
        "rf": (
            ImbPipeline([("scaler", StandardScaler()), _smote(), _selector(), ("clf", RandomForestClassifier(random_state=rs))]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10], "clf__class_weight": [None, "balanced"]},
        ),
        "xgb": (
            ImbPipeline([("scaler", StandardScaler()), _smote(), _selector(), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=rs))]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]},
        ),
    }

    FIXED_PIPES = {
        "svm": ImbPipeline([("scaler", StandardScaler()), _smote(), _selector(), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=rs))]),
        "rf":  ImbPipeline([("scaler", StandardScaler()), _smote(), _selector(), ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=rs))]),
        "xgb": ImbPipeline([("scaler", StandardScaler()), _smote(), _selector(), ("clf", XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric="logloss", random_state=rs))]),
    }

    outer_cv = StratifiedGroupKFold(n_splits=n_splits)
    inner_cv = StratifiedGroupKFold(n_splits=3)
    rng = np.random.RandomState(rs)
    unique_groups = np.unique(groups)
    group_label_map = {g: y_enc[groups == g][0] for g in unique_groups}
    group_label_vals = np.array([group_label_map[g] for g in unique_groups])

    def _group_permute():
        perm_map = dict(zip(unique_groups, rng.permutation(group_label_vals)))
        return np.array([perm_map[g] for g in groups])

    results = {}
    for name, (pipe, param_grid) in MODEL_REGISTRY.items():
        probas, preds, test_indices, fold_thresholds = [], [], [], []
        for train_idx, test_idx in outer_cv.split(X, y_enc, groups):
            gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="roc_auc", refit=True, n_jobs=1)
            gs.fit(X[train_idx], y_enc[train_idx], groups=groups[train_idx])
            probas.append(gs.predict_proba(X[test_idx])[:, mdd_idx])
            preds.append(gs.predict(X[test_idx]))
            test_indices.append(test_idx)
            tr_proba = gs.predict_proba(X[train_idx])[:, mdd_idx]
            fpr_t, tpr_t, thr_t = roc_curve(y_enc[train_idx], tr_proba, pos_label=mdd_idx)
            fold_thresholds.append(float(thr_t[int(np.argmax(tpr_t - fpr_t))]))

        order = np.argsort(np.concatenate(test_indices))
        proba = np.concatenate(probas)[order]
        pred = np.concatenate(preds)[order]

        auc = roc_auc_score(y_enc, proba)
        acc = accuracy_score(y_enc, pred)
        tn, fp, fn, tp = confusion_matrix(y_enc, pred).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        opt_thresh = float(np.mean(fold_thresholds))
        pred_opt = np.where(proba >= opt_thresh, mdd_idx, 1 - mdd_idx)
        tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_enc, pred_opt).ravel()
        sen_opt = float(tp_o / (tp_o + fn_o)) if (tp_o + fn_o) > 0 else 0.0
        spe_opt = float(tn_o / (tn_o + fp_o)) if (tn_o + fp_o) > 0 else 0.0

        fixed_pipe = FIXED_PIPES[name]
        perm_scores = [
            cross_val_score(fixed_pipe, X, _group_permute(), groups=groups, cv=outer_cv, scoring="roc_auc").mean()
            for _ in range(n_permutations)
        ]
        pvalue = (np.sum(np.array(perm_scores) >= auc) + 1) / (n_permutations + 1)

        results[name] = {
            "auc": float(auc),
            "accuracy": float(acc),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "optimal_threshold": opt_thresh,
            "sensitivity_at_threshold": sen_opt,
            "specificity_at_threshold": spe_opt,
            "permutation_pvalue": float(pvalue),
            "n_permutations": int(n_permutations),
            "n_splits": int(n_splits),
        }

    ens_probas, ens_test_indices = [], []
    for train_idx, test_idx in outer_cv.split(X, y_enc, groups):
        fold_probas = []
        for ename, (epipe, eparam) in MODEL_REGISTRY.items():
            gs = GridSearchCV(epipe, eparam, cv=inner_cv, scoring="roc_auc", refit=True, n_jobs=1)
            gs.fit(X[train_idx], y_enc[train_idx], groups=groups[train_idx])
            fold_probas.append(gs.predict_proba(X[test_idx])[:, mdd_idx])
        ens_probas.append(np.mean(fold_probas, axis=0))
        ens_test_indices.append(test_idx)

    ens_order = np.argsort(np.concatenate(ens_test_indices))
    ens_proba = np.concatenate(ens_probas)[ens_order]
    ens_auc = float(roc_auc_score(y_enc, ens_proba))
    efpr, etpr, ethresholds = roc_curve(y_enc, ens_proba, pos_label=mdd_idx)
    ens_opt_thresh = float(ethresholds[int(np.argmax(etpr - efpr))])
    ens_pred = np.where(ens_proba >= ens_opt_thresh, mdd_idx, 1 - mdd_idx)
    tn_e, fp_e, fn_e, tp_e = confusion_matrix(y_enc, ens_pred).ravel()

    results["ensemble"] = {
        "auc": ens_auc,
        "accuracy": float(accuracy_score(y_enc, ens_pred)),
        "sensitivity": float(tp_e / (tp_e + fn_e)) if (tp_e + fn_e) > 0 else 0.0,
        "specificity": float(tn_e / (tn_e + fp_e)) if (tn_e + fp_e) > 0 else 0.0,
        "optimal_threshold": ens_opt_thresh,
        "permutation_pvalue": None,
        "n_splits": int(n_splits),
    }

    return results