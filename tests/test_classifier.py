import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import classify


def _build_synthetic_data():
    """Build a small, separable dataset with per-subject grouping."""
    rng = np.random.default_rng(42)
    n_per_class = 8

    adhd = rng.normal(loc=-1.0, scale=0.5, size=(n_per_class, 6))
    mdd = rng.normal(loc=1.0, scale=0.5, size=(n_per_class, 6))

    X = np.vstack([adhd, mdd])
    y = np.array(["ADHD"] * n_per_class + ["MDD"] * n_per_class)
    groups = np.array([f"s{i:02d}" for i in range(X.shape[0])])
    return X, y, groups


def test_classify_returns_expected_metrics_and_config():
    X, y, groups = _build_synthetic_data()

    result = classify(
        X,
        y,
        groups,
        n_splits=4,
        n_permutations=20,
        random_state=42,
    )

    expected_keys = {
        "auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "permutation_pvalue",
        "n_permutations",
        "n_splits",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n_permutations"] == 20
    assert result["n_splits"] == 4

    assert 0.0 <= result["auc"] <= 1.0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["sensitivity"] <= 1.0
    assert 0.0 <= result["specificity"] <= 1.0
    assert 0.0 <= result["permutation_pvalue"] <= 1.0


def test_no_group_leakage():
    """Each fold's train and test groups must be disjoint."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder

    X, y, groups = _build_synthetic_data()
    y_enc = LabelEncoder().fit_transform(y)
    cv = StratifiedGroupKFold(n_splits=4)
    for train_idx, test_idx in cv.split(X, y_enc, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_mdd_proba_column_index():
    """LabelEncoder assigns MDD=1 alphabetically; dynamic index must equal 1."""
    from sklearn.preprocessing import LabelEncoder

    for label_input in [["ADHD", "MDD"], ["MDD", "ADHD"], ["MDD"] * 5 + ["ADHD"] * 5]:
        le = LabelEncoder().fit(label_input)
        assert list(le.classes_) == ["ADHD", "MDD"]
        assert list(le.classes_).index("MDD") == 1


def test_result_stability():
    """Same input with fixed random_state produces identical results."""
    X, y, groups = _build_synthetic_data()
    kwargs = dict(n_splits=4, n_permutations=10, random_state=0)
    r1 = classify(X, y, groups, **kwargs)
    r2 = classify(X, y, groups, **kwargs)
    for key in ("auc", "accuracy", "sensitivity", "specificity", "permutation_pvalue"):
        assert r1[key] == r2[key], f"{key} differs: {r1[key]} vs {r2[key]}"
