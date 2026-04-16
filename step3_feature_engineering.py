"""
Telco Customer Churn - Module 3/5: Feature Engineering & Feature Selection

Business goal
- Enhance features for modeling (auto feature generation + selection)
- Produce correlation-based insights to support retention strategy narratives

Dependencies
pip install pandas numpy scikit-learn featuretools pyyaml
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from project_utils import Timer, ensure_output_dirs, get_paths, load_config, setup_logger


SCRIPT_PREFIX = "step3_feature_engineering"
UPSTREAM_PREFIX = "step2_preprocess"


def resolve_config_path(cli_config: str | None) -> Path:
    base = Path(__file__).resolve().parent
    return Path(cli_config).resolve() if cli_config else (base / "config.yaml")


def resolve_model_ready_input_path(cli_path: str | None) -> Path:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "outputs" / "csv" / "step2__model_ready_dataset.csv",
        ]
    )

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Model-ready dataset not found. Please run step2_preprocess.py first, or pass --input explicitly."
    )


def feature_churn_correlations(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    y = y.astype(float).reshape(-1)
    rows = []
    for i, name in enumerate(feature_names):
        xi = X[:, i].astype(float)
        if np.nanstd(xi) == 0 or np.nanstd(y) == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(xi, y)[0, 1])
            if np.isnan(r):
                r = 0.0
        rows.append((name, r, abs(r)))

    df_corr = pd.DataFrame(rows, columns=["feature", "corr_with_churn", "abs_corr"])
    return df_corr.sort_values("abs_corr", ascending=False).reset_index(drop=True)

def generate_featuretools_features(
    X: pd.DataFrame,
    config: dict,
    logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Auto feature generation using Featuretools (single-table DFS).
    For this dataset, it mainly creates numeric interaction features.
    """

    fe_cfg = config.get("feature_engineering") if isinstance(config.get("feature_engineering"), dict) else {}
    ft_cfg = fe_cfg.get("featuretools", {}) if isinstance(fe_cfg.get("featuretools"), dict) else {}
    enabled = bool(ft_cfg.get("enabled", True))
    if not enabled:
        return X.copy(), {"enabled": False}

    try:
        import featuretools as ft  # type: ignore
    except Exception as e:  # noqa: BLE001
        # Fallback: create a small number of safe interaction features so the pipeline remains runnable.
        logger.warning(f"featuretools unavailable ({type(e).__name__}: {e}); using fallback interaction features.")
        df = X.copy()
        cols = list(df.columns)[:10]  # limit to avoid explosion
        added = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                new_col = f"ft_fallback__{c1}__x__{c2}"
                df[new_col] = df[c1].astype(float) * df[c2].astype(float)
                added += 1
                if added >= 25:
                    break
            if added >= 25:
                break
        return df, {"enabled": False, "reason": "fallback_interactions", "added_features": added}

    max_depth = int(ft_cfg.get("max_depth", 1))
    trans_primitives = ft_cfg.get("trans_primitives", []) if isinstance(ft_cfg.get("trans_primitives"), list) else []
    if not trans_primitives:
        trans_primitives = ["multiply_numeric", "divide_numeric", "add_numeric", "subtract_numeric"]

    df = X.copy()
    df = df.reset_index(drop=True)
    df.insert(0, "_row_id", np.arange(len(df), dtype=int))

    es = ft.EntitySet(id="telco")
    es = es.add_dataframe(dataframe_name="customers", dataframe=df, index="_row_id")

    logger.info(f"Running featuretools DFS (max_depth={max_depth}, trans_primitives={trans_primitives})...")
    t = Timer()
    fm, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="customers",
        trans_primitives=trans_primitives,
        agg_primitives=[],
        max_depth=max_depth,
        verbose=False,
    )
    elapsed = t.elapsed_s()

    # featuretools returns _row_id index; drop it from features
    fm = fm.reset_index(drop=True)
    if "_row_id" in fm.columns:
        fm = fm.drop(columns=["_row_id"], errors="ignore")

    meta = {"enabled": True, "n_input_features": int(X.shape[1]), "n_output_features": int(fm.shape[1]), "elapsed_s": elapsed, "n_feature_defs": int(len(feature_defs))}
    logger.info(f"Featuretools DFS done. output_features={fm.shape[1]}, elapsed_s={elapsed:.3f}")
    return fm, meta


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
    logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fe_cfg = config.get("feature_engineering") if isinstance(config.get("feature_engineering"), dict) else {}
    sel_cfg = fe_cfg.get("selection", {}) if isinstance(fe_cfg.get("selection"), dict) else {}
    method = str(sel_cfg.get("method", "selectkbest")).lower()

    if method == "none":
        return X.copy(), {"method": "none"}

    if method == "pca":
        n = int(sel_cfg.get("pca_components", 20))
        n = max(2, min(n, X.shape[1]))
        logger.info(f"Applying PCA feature selection: n_components={n}")
        pca = PCA(n_components=n, random_state=0)
        X_pca = pca.fit_transform(X.to_numpy(dtype=float))
        cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=cols), {"method": "pca", "n_components": n, "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))}

    # default: SelectKBest
    k = int(sel_cfg.get("k_best", 40))
    k = max(5, min(k, X.shape[1]))
    logger.info(f"Applying SelectKBest feature selection: k={k}")
    skb = SelectKBest(score_func=f_classif, k=k)
    X_new = skb.fit_transform(X.to_numpy(dtype=float), y.astype(int))
    selected = skb.get_support(indices=True).tolist()
    cols = [X.columns[i] for i in selected]
    return pd.DataFrame(X_new, columns=cols), {"method": "selectkbest", "k": k, "selected_features": cols}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Telco churn - feature engineering")
    parser.add_argument("--input", type=str, default=None, help="Model-ready CSV path (default: outputs/csv/step2__model_ready_dataset.csv)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
        paths = get_paths(config, Path(__file__).resolve().parent)
        ensure_output_dirs(paths)
        logger = setup_logger(SCRIPT_PREFIX, paths.logs_dir)

        timer = Timer()
        input_path = resolve_model_ready_input_path(args.input)
        logger.info(f"Loading model-ready dataset: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Model-ready dataset loaded. rows={len(df)}, cols={df.shape[1]}, elapsed_s={timer.elapsed_s():.3f}")
    except (FileNotFoundError, ValueError) as e:
        msg = f"{type(e).__name__}: {e}"
        print(msg)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        msg = f"UnexpectedError({type(e).__name__}): {e}"
        print(msg)
        raise SystemExit(1)

    if "ChurnLabel" not in df.columns:
        raise ValueError("Missing ChurnLabel in model-ready dataset.")

    y = df["ChurnLabel"].to_numpy(dtype=int)
    X_raw = df.drop(columns=["ChurnLabel"], errors="ignore").copy()
    # Ensure all features are numeric at this stage
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
    if X_raw.isna().any().any():
        # Avoid training-time NaNs; keep a strict failure to surface issues early
        raise ValueError("Model-ready dataset contains NaNs after numeric coercion. Check step2 preprocessing.")

    # Featuretools auto feature generation (optional)
    X_ft, ft_meta = generate_featuretools_features(X_raw, config, logger)

    # Feature selection (SelectKBest/PCA/none)
    X_sel, sel_meta = select_features(X_ft, y, config, logger)

    X = X_sel.to_numpy(dtype=float)
    feature_names = list(X_sel.columns)

    corr_df = feature_churn_correlations(X, y, feature_names)

    top_n_for_matrix = 30
    top_features = corr_df.head(top_n_for_matrix)["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    X_top = X[:, top_idx]
    corr_matrix = pd.DataFrame(np.corrcoef(X_top, rowvar=False), index=top_features, columns=top_features)

    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame({"ChurnLabel": y.astype(int)})

    X_path = paths.csv_dir / f"{SCRIPT_PREFIX}__features_X.csv"
    y_path = paths.csv_dir / f"{SCRIPT_PREFIX}__labels_y.csv"
    engineered_path = paths.csv_dir / "step3__engineered_features.csv"
    corr_path = paths.csv_dir / f"{SCRIPT_PREFIX}__feature_churn_correlation.csv"
    matrix_path = paths.csv_dir / f"{SCRIPT_PREFIX}__top_feature_correlation_matrix.csv"
    meta_path = paths.reports_dir / f"{SCRIPT_PREFIX}__metadata.json"
    summary_path = paths.reports_dir / f"{SCRIPT_PREFIX}__summary.txt"

    X_df.to_csv(X_path, index=False, encoding="utf-8")
    y_df.to_csv(y_path, index=False, encoding="utf-8")
    # Required by tips1.txt
    pd.concat([X_df, y_df], axis=1).to_csv(engineered_path, index=False, encoding="utf-8")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8")
    corr_matrix.to_csv(matrix_path, encoding="utf-8")

    meta = {
        "input_model_ready_csv": str(input_path),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "featuretools": ft_meta,
        "selection": sel_meta,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")

    top_k = 15
    top_k_df = corr_df.head(top_k)

    summary_lines: list[str] = []
    summary_lines.append("Telco Customer Churn - Feature Engineering Summary")
    summary_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Input dataset: {input_path}")
    summary_lines.append("")
    summary_lines.append("1) Transformation strategy")
    summary_lines.append(f"- Featuretools enabled: {ft_meta.get('enabled', False)}")
    summary_lines.append(f"- Selection method: {sel_meta.get('method')}")
    summary_lines.append("")
    summary_lines.append("2) Top features by absolute correlation with churn (heuristic)")
    summary_lines.append(top_k_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Note: correlation is for interpretability and triage, not causality.")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8", errors="replace")

    print("=== Feature engineering complete ===")
    print(f"X features: {X_path}")
    print(f"y labels: {y_path}")
    print(f"Engineered dataset (required): {engineered_path}")
    print(f"Feature-churn correlation: {corr_path}")
    print(f"Top-feature correlation matrix: {matrix_path}")
    print(f"Metadata: {meta_path}")
    print(f"Summary: {summary_path}")
    logger.info(f"Saved features: {X_path}")
    logger.info(f"Saved labels: {y_path}")
    logger.info(f"Saved engineered dataset: {engineered_path}")
    logger.info(f"Saved correlation CSV: {corr_path}")
    logger.info(f"Saved top correlation matrix CSV: {matrix_path}")
    logger.info(f"Saved metadata: {meta_path}")
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

