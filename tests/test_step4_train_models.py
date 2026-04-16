import numpy as np
import pandas as pd

from step4_train_models import train_and_split


def test_train_and_split_trains_models_and_stratifies():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    # Imbalanced but stratifiable
    y = pd.Series([0] * 160 + [1] * 40)

    training_cfg = {
        "test_size": 0.25,
        "random_state": 42,
        "models": {
            "logistic_regression": {"max_iter": 200, "solver": "lbfgs"},
            "decision_tree": {},
            "random_forest": {"n_estimators": 50, "n_jobs": -1},
        },
    }

    X_train, X_test, y_train, y_test, models = train_and_split(X, y, training_cfg)
    assert len(X_train) + len(X_test) == len(X)
    assert set(models.keys()) == {"logistic_regression", "decision_tree", "random_forest"}

    # Stratification: churn ratio should be close between train and test
    r_train = float(y_train.mean())
    r_test = float(y_test.mean())
    assert abs(r_train - r_test) < 0.05

