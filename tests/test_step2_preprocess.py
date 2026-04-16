import numpy as np
import pandas as pd

from step2_preprocess import (
    build_model_ready_dataset,
    encode_categoricals_highfreq_onehot_lowfreq_target,
    fill_missing_values,
    scale_numeric_features,
)


def test_fill_missing_values_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": ["x", None, "x"],
        }
    )
    out, meta = fill_missing_values(df, numeric_strategy="median", categorical_strategy="mode")
    assert out["a"].isna().sum() == 0
    assert out["b"].isna().sum() == 0
    assert "a" in meta["numeric_fills"]
    assert "b" in meta["categorical_fills"]


def test_encode_highfreq_onehot_and_target_encoding():
    df = pd.DataFrame({"cat": ["A"] * 10 + ["B"] * 1 + ["C"] * 1})
    y = pd.Series([0] * 6 + [1] * 6)
    X_enc, meta = encode_categoricals_highfreq_onehot_lowfreq_target(df, y, high_freq_threshold=0.2, smoothing=5.0)

    # High freq category A should become one-hot
    assert any(c.startswith("cat__A") for c in X_enc.columns)
    # Target encoding should exist
    assert "cat__target_enc" in X_enc.columns
    assert "cat" in meta["target_encoding"]


def test_scale_numeric_features_standard():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 20.0, 30.0]})
    out, meta = scale_numeric_features(df, method="standard")
    assert set(meta["scaled_cols"]) == {"x", "y"}
    # StandardScaler -> mean approx 0
    assert abs(float(out["x"].mean())) < 1e-6


def test_build_model_ready_dataset_outputs_label_and_features():
    df_clean = pd.DataFrame(
        {
            "customerID": ["c1", "c2", "c3", "c4"],
            "tenure": [1, 2, 3, 4],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
            "TotalCharges": [10.0, 40.0, 90.0, 160.0],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "Churn": ["No", "Yes", "No", "Yes"],
            "ChurnLabel": [0, 1, 0, 1],
        }
    )
    config = {
        "preprocess": {
            "fill_missing": {"enabled": True, "numeric": "median", "categorical": "mode"},
            "encoding": {"high_freq_threshold": 0.3, "target_encoding_smoothing": 5.0},
            "scaling": {"enabled": True, "method": "standard"},
        }
    }

    model_df, meta = build_model_ready_dataset(df_clean, config)
    assert "ChurnLabel" in model_df.columns
    assert model_df.isna().sum().sum() == 0
    assert meta["encoding"] is not None

