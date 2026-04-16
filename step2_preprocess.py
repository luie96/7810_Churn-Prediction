"""
Telco Customer Churn - Module 2/5: Data Cleaning & Preprocessing

Business goal
- Produce a clean, standardized dataset for modeling
- Handle the known pain point: TotalCharges missing/blank strings + numeric conversion

Dependencies
pip install pandas numpy scikit-learn pyyaml
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from project_utils import Timer, ensure_output_dirs, get_paths, load_config, setup_logger


SCRIPT_PREFIX = "step2_preprocess"


def resolve_config_path(cli_config: str | None) -> Path:
    base = Path(__file__).resolve().parent
    return Path(cli_config).resolve() if cli_config else (base / "config.yaml")


def resolve_dataset_path(inputs_dir: Path, config: dict, cli_path: str | None) -> Path:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    raw_candidates = (config.get("data") or {}).get("raw_candidates", []) if isinstance(config.get("data"), dict) else []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    if not raw_candidates:
        raw_candidates = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    candidates.extend([inputs_dir / str(name) for name in raw_candidates])

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Raw CSV not found.\n"
        f"- Please put the dataset under: {inputs_dir}\n"
        "- Recommended filename: WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
        "- Or pass an explicit path via --csv"
    )


def safe_to_numeric_totalcharges(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan}), errors="coerce")

def validate_required_columns(df: pd.DataFrame, config: dict) -> None:
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    required = data_cfg.get("required_columns", []) if isinstance(data_cfg, dict) else []
    if not required:
        return
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    steps: list[str] = []
    df = df_raw.copy()

    steps.append(f"Raw dataset size: rows={len(df)}, cols={df.shape[1]}")

    # 1) Deduplication: customerID should be unique.
    if "customerID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["customerID"], keep="first")
        steps.append(f"Deduplicate by customerID: {before} -> {len(df)} (dropped {before - len(df)})")
    else:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        steps.append(f"Deduplicate by full row: {before} -> {len(df)} (dropped {before - len(df)})")

    # 2) TotalCharges: convert to numeric and handle missing.
    if "TotalCharges" in df.columns:
        blank_count = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
        df["TotalCharges"] = safe_to_numeric_totalcharges(df["TotalCharges"])
        missing_after = int(df["TotalCharges"].isna().sum())
        steps.append(
            "TotalCharges cleaning: treat blank strings as missing and convert to numeric. "
            f"blank_strings={blank_count}, missing_after_conversion={missing_after}."
        )

        # Business logic: tenure=0 implies newly joined; TotalCharges should be 0.
        if "tenure" in df.columns:
            m = (df["tenure"] == 0) & (df["TotalCharges"].isna())
            filled = int(m.sum())
            df.loc[m, "TotalCharges"] = 0.0
            steps.append(f"Fill TotalCharges=0 when tenure=0 and TotalCharges missing: filled {filled} rows.")

        # Remaining missing: fill with median (robust; keeps linear models stable).
        remaining = int(df["TotalCharges"].isna().sum())
        if remaining > 0:
            med = float(df["TotalCharges"].median(skipna=True))
            df["TotalCharges"] = df["TotalCharges"].fillna(med)
            steps.append(f"Fill remaining TotalCharges missing with median: missing={remaining}, median={med:.4f}.")

    # 3) SeniorCitizen: normalize 0/1 -> Yes/No (clearer for one-hot later).
    if "SeniorCitizen" in df.columns:
        before_unique = df["SeniorCitizen"].unique().tolist()
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"].astype(str))
        after_unique = df["SeniorCitizen"].unique().tolist()
        steps.append(f"Normalize SeniorCitizen: before={before_unique}, after={after_unique}.")

    # 4) Churn label: add numeric label for modeling.
    if "Churn" in df.columns:
        df["ChurnLabel"] = df["Churn"].map({"Yes": 1, "No": 0})
        unmapped = int(df["ChurnLabel"].isna().sum())
        if unmapped > 0:
            df["ChurnLabel"] = pd.to_numeric(df["ChurnLabel"], errors="coerce")
        steps.append(f"Encode churn label: add ChurnLabel (Yes=1, No=0). Unmapped={unmapped}.")

    # 5) Remove obviously invalid rows (negative values).
    invalid = pd.Series(False, index=df.index)
    if "tenure" in df.columns:
        invalid |= pd.to_numeric(df["tenure"], errors="coerce") < 0
    if "MonthlyCharges" in df.columns:
        invalid |= pd.to_numeric(df["MonthlyCharges"], errors="coerce") < 0
    if "TotalCharges" in df.columns:
        invalid |= pd.to_numeric(df["TotalCharges"], errors="coerce") < 0

    invalid_count = int(invalid.sum())
    if invalid_count > 0:
        before = len(df)
        df = df.loc[~invalid].copy()
        steps.append(f"Drop invalid rows (negative charges/tenure): {before} -> {len(df)} (dropped {invalid_count}).")
    else:
        steps.append("No invalid rows found (negative charges/tenure).")

    # 6) Final dtype normalization for numeric columns.
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    steps.append(f"Cleaned dataset size: rows={len(df)}, cols={df.shape[1]}")
    return df, steps


def fill_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Fill missing values by column type.

    Parameters
    - df: input dataframe
    - numeric_strategy: "median" or "mean"
    - categorical_strategy: "mode" (most frequent)

    Returns
    - filled dataframe
    - summary metadata
    """

    out = df.copy()
    meta: dict[str, object] = {"numeric_fills": {}, "categorical_fills": {}}

    for col in out.columns:
        s = out[col]
        if s.isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            if numeric_strategy == "mean":
                fill_value = float(s.mean(skipna=True))
            else:
                fill_value = float(s.median(skipna=True))
            out[col] = s.fillna(fill_value)
            meta["numeric_fills"][col] = fill_value
        else:
            if categorical_strategy != "mode":
                raise ValueError(f"Unsupported categorical missing strategy: {categorical_strategy}")
            mode_vals = s.mode(dropna=True)
            fill_value = "" if mode_vals.empty else str(mode_vals.iloc[0])
            out[col] = s.fillna(fill_value)
            meta["categorical_fills"][col] = fill_value

    return out, meta


def _smoothed_target_mean(count: float, mean: float, global_mean: float, smoothing: float) -> float:
    return float((count * mean + smoothing * global_mean) / (count + smoothing))


def encode_categoricals_highfreq_onehot_lowfreq_target(
    df: pd.DataFrame,
    y: pd.Series,
    high_freq_threshold: float = 0.05,
    smoothing: float = 10.0,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Encoding rule (as requested):
    - For each categorical column:
      - high-frequency categories (share > threshold) -> One-Hot features
      - low-frequency categories (share <= threshold) -> Target encoding

    Implementation detail:
    - One-Hot: create col__<category> features for high-frequency categories
    - Target encoding: create a numeric feature col__target_enc for the whole column,
      computed with smoothed mean churn per category

    Returns
    - encoded feature dataframe (only engineered features; excludes original categoricals)
    - metadata for auditing/testing
    """

    if exclude_cols is None:
        exclude_cols = []

    if len(df) != len(y):
        raise ValueError(f"X/y length mismatch: len(X)={len(df)}, len(y)={len(y)}")

    global_mean = float(pd.to_numeric(y, errors="coerce").mean())
    meta: dict[str, object] = {"onehot": {}, "target_encoding": {}, "high_freq_threshold": high_freq_threshold}

    feature_frames: list[pd.DataFrame] = []
    cat_cols = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])) and c not in exclude_cols]

    for col in cat_cols:
        s = df[col].astype(str).fillna("")
        freq = s.value_counts(normalize=True)
        high_cats = freq[freq > high_freq_threshold].index.tolist()

        # One-hot for high-frequency categories
        if high_cats:
            oh = pd.get_dummies(s.where(s.isin(high_cats), "__OTHER_HIGHFREQ__"), prefix=col, prefix_sep="__")
            # Keep only true high categories; drop the other bucket to avoid redundant feature
            keep_cols = [c for c in oh.columns if c.split("__", 1)[-1] in set(high_cats)]
            oh = oh[keep_cols]
            feature_frames.append(oh)
            meta["onehot"][col] = {"high_categories": high_cats, "n_onehot_features": len(keep_cols)}
        else:
            meta["onehot"][col] = {"high_categories": [], "n_onehot_features": 0}

        # Target encoding (smoothed)
        joined = pd.DataFrame({col: s, "y": pd.to_numeric(y, errors="coerce")})
        stats = joined.groupby(col)["y"].agg(["count", "mean"]).reset_index()
        mapping = {
            str(r[col]): _smoothed_target_mean(float(r["count"]), float(r["mean"]), global_mean, float(smoothing))
            for _, r in stats.iterrows()
        }
        te_col = f"{col}__target_enc"
        feature_frames.append(pd.DataFrame({te_col: s.map(mapping).fillna(global_mean).astype(float)}))
        meta["target_encoding"][col] = {"smoothing": smoothing, "global_mean": global_mean, "n_categories": int(len(mapping))}

    if not feature_frames:
        return pd.DataFrame(index=df.index), meta

    encoded = pd.concat(feature_frames, axis=1)
    return encoded, meta


def scale_numeric_features(df: pd.DataFrame, method: str = "standard") -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Scale numeric columns using StandardScaler or MinMaxScaler.
    """

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df.copy(), {"scaled_cols": [], "method": method}

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    out = df.copy()
    out[numeric_cols] = scaler.fit_transform(out[numeric_cols])
    return out, {"scaled_cols": numeric_cols, "method": method}


def build_model_ready_dataset(df_clean: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Build a model-ready dataset (features + label) from the cleaned table.

    Output columns:
    - Numeric features (optionally scaled)
    - Encoded categorical features (high-freq one-hot + target encoding)
    - ChurnLabel
    """

    if "ChurnLabel" not in df_clean.columns:
        raise ValueError("ChurnLabel is missing. Ensure the raw data has a valid Churn column.")

    preprocess_cfg = config.get("preprocess") if isinstance(config.get("preprocess"), dict) else {}
    fill_cfg = preprocess_cfg.get("fill_missing", {}) if isinstance(preprocess_cfg.get("fill_missing"), dict) else {}
    enc_cfg = preprocess_cfg.get("encoding", {}) if isinstance(preprocess_cfg.get("encoding"), dict) else {}
    scale_cfg = preprocess_cfg.get("scaling", {}) if isinstance(preprocess_cfg.get("scaling"), dict) else {}

    df = df_clean.copy()

    # Split y early (used by target encoding)
    y = df["ChurnLabel"].astype(int)

    # Exclude identifiers/labels from features
    exclude = ["customerID", "Churn", "ChurnLabel"]
    X_base = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").copy()

    # Fill missing values by type (configurable)
    meta: dict[str, object] = {"fill_missing": None, "encoding": None, "scaling": None}
    if bool(fill_cfg.get("enabled", True)):
        X_base, fill_meta = fill_missing_values(
            X_base,
            numeric_strategy=str(fill_cfg.get("numeric", "median")),
            categorical_strategy=str(fill_cfg.get("categorical", "mode")),
        )
        meta["fill_missing"] = fill_meta

    # Separate numeric and categorical
    numeric_cols = [c for c in X_base.columns if pd.api.types.is_numeric_dtype(X_base[c])]
    X_num = X_base[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_base.index)
    X_cat = X_base.drop(columns=numeric_cols, errors="ignore")

    # Scale numeric (configurable)
    if bool(scale_cfg.get("enabled", True)) and not X_num.empty:
        X_num, scale_meta = scale_numeric_features(X_num, method=str(scale_cfg.get("method", "standard")))
        meta["scaling"] = scale_meta

    # Encode categoricals
    X_enc, enc_meta = encode_categoricals_highfreq_onehot_lowfreq_target(
        X_cat,
        y=y,
        high_freq_threshold=float(enc_cfg.get("high_freq_threshold", 0.05)),
        smoothing=float(enc_cfg.get("target_encoding_smoothing", 10.0)),
        exclude_cols=[],
    )
    meta["encoding"] = enc_meta

    X_final = pd.concat([X_num, X_enc], axis=1)
    model_df = pd.concat([X_final, y.rename("ChurnLabel")], axis=1)
    return model_df, meta


def save_artifacts(
    csv_dir: Path,
    reports_dir: Path,
    df_clean: pd.DataFrame,
    df_model_ready: pd.DataFrame,
    steps: list[str],
    preprocess_meta: dict[str, object],
) -> tuple[Path, Path, Path]:
    cleaned_path = csv_dir / f"{SCRIPT_PREFIX}__telco_cleaned.csv"
    steps_path = reports_dir / f"{SCRIPT_PREFIX}__preprocess_steps_summary.txt"
    model_ready_path = csv_dir / "step2__model_ready_dataset.csv"

    df_clean.to_csv(cleaned_path, index=False, encoding="utf-8")
    df_model_ready.to_csv(model_ready_path, index=False, encoding="utf-8")

    content = "\n".join(
        [
            "Telco Customer Churn - Preprocessing Steps Summary",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            *[f"- {s}" for s in steps],
            "",
            "Preprocessing metadata (encoding/scaling) for auditing:",
            json.dumps(preprocess_meta, ensure_ascii=False, indent=2),
            "",
            "Note: this file documents key preprocessing decisions and dataset size changes for reporting.",
        ]
    )
    steps_path.write_text(content, encoding="utf-8", errors="replace")
    return cleaned_path, model_ready_path, steps_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - preprocessing")
    parser.add_argument("--csv", type=str, default=None, help="Raw Telco churn CSV path (optional override)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
        paths = get_paths(config, Path(__file__).resolve().parent)
        ensure_output_dirs(paths)
        logger = setup_logger(SCRIPT_PREFIX, paths.logs_dir)

        timer = Timer()
        csv_path = resolve_dataset_path(paths.inputs_dir, config, args.csv)
        logger.info(f"Loading raw dataset: {csv_path}")
        df_raw = pd.read_csv(csv_path)
        validate_required_columns(df_raw, config)
        logger.info(f"Raw dataset loaded. rows={len(df_raw)}, cols={df_raw.shape[1]}, elapsed_s={timer.elapsed_s():.3f}")

        df_clean, steps = clean_dataset(df_raw)
        df_model_ready, preprocess_meta = build_model_ready_dataset(df_clean, config)
        cleaned_path, model_ready_path, steps_path = save_artifacts(
            paths.csv_dir,
            paths.reports_dir,
            df_clean,
            df_model_ready,
            steps,
            preprocess_meta,
        )

        logger.info(f"Saved cleaned dataset: {cleaned_path}")
        logger.info(f"Saved model-ready dataset: {model_ready_path}")
        logger.info(f"Saved steps summary: {steps_path}")

        print("=== Preprocessing complete ===")
        print(f"Saved cleaned dataset: {cleaned_path}")
        print(f"Saved model-ready dataset: {model_ready_path}")
        print(f"Saved steps summary: {steps_path}")
        print()
        print("Steps (copy/paste into report):")
        for s in steps:
            print("-", s)
    except (FileNotFoundError, ValueError) as e:
        msg = f"{type(e).__name__}: {e}"
        print(msg)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        msg = f"UnexpectedError({type(e).__name__}): {e}"
        print(msg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

