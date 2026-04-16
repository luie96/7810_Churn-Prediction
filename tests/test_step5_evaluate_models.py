import numpy as np
import pandas as pd

from step5_evaluate_models import feature_importance, plot_confusion, plot_roc


class DummyTree:
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances, dtype=float)


class DummyLinear:
    def __init__(self, coef):
        self.coef_ = np.array([coef], dtype=float)


def test_feature_importance_tree_model():
    fi = feature_importance(DummyTree([0.1, 0.7, 0.2]), ["a", "b", "c"])
    assert fi.iloc[0]["feature"] == "b"


def test_feature_importance_linear_model():
    fi = feature_importance(DummyLinear([0.0, -2.0, 1.0]), ["a", "b", "c"])
    assert fi.iloc[0]["feature"] == "b"


def test_plot_functions_create_files(tmp_path):
    # Minimal model API for plots: predict + predict_proba
    class M:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            p = np.zeros((len(X), 2), dtype=float)
            p[:, 1] = 0.2
            p[:, 0] = 0.8
            return p

    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([0, 1, 0, 1])

    cm_path = plot_confusion("dummy", M(), X, y, tmp_path)
    roc_path = plot_roc("dummy", M(), X, y, tmp_path)
    assert cm_path.exists()
    assert roc_path.exists()

