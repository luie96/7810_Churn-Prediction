# Telco Customer Churn (7810)

This project implements an end-to-end Telco customer churn prediction pipeline in 5 steps.

It is designed for:
- **Business**: identify churn drivers and segment high-risk customers to support retention strategy.
- **Technical**: build multiple baseline models and evaluate them consistently.
- **Engineering**: reproducible runs via `config.yaml`, structured outputs, logging, and automated tests.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Dataset

Put the raw dataset under the project root `inputs/` folder (recommended filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
If your file is elsewhere, you can run `python main.py --csv "path\\to\\csv"` and it will be copied into `inputs/`.

## How to run

### One-command pipeline (recommended)

```bash
python main.py
```

### Step-by-step (optional)

```bash
python step1_data_exploration.py --config config.yaml
python step2_preprocess.py --config config.yaml
python step3_feature_engineering.py --config config.yaml
python step4_train_models.py --config config.yaml
python step5_evaluate_models.py --config config.yaml
```

## Configuration

All key parameters are centralized in `config.yaml`:
- **paths**: where inputs/outputs/logs live
- **preprocess**: missing-value policy, encoding rules, scaling method
- **feature_engineering**: featuretools + SelectKBest/PCA settings
- **training**: split ratio, random seed, model hyperparameters
- **evaluation**: top-k importance, SHAP settings

## Outputs (all under `outputs/`)

- `outputs/csv/`: all CSV outputs (filenames include the step prefix)
- `outputs/reports/`: text/json reports (filenames include the step prefix)
- `outputs/models/`: trained models (`.joblib`)
- `outputs/plots/`: confusion matrices and ROC curves (`.png`)

Additional outputs:
- `outputs/logs/`: step logs (`<script>_YYYYMMDD.log`)

## Output files guide (what to look at)

### `outputs/csv/`
- `step2_preprocess__telco_cleaned.csv`: cleaned table (types fixed, `ChurnLabel` added)
- `step2__model_ready_dataset.csv`: model-ready dataset after encoding/scaling (features + `ChurnLabel`)
- `step3__engineered_features.csv`: engineered/selected features (features + `ChurnLabel`)
- `step5_evaluate_models__model_metrics.csv`: metrics table per model on the test split
- `step5_evaluate_models__best_model_feature_importance.csv`: feature importance table for the selected best model

### `outputs/reports/`
- `step1__auto_eda_report.html`: automated EDA report (uses `ydata-profiling` when available; otherwise a fallback HTML)
- `step2_preprocess_report.md`: preprocessing transparency report (missing/outliers/shape changes/strategy used)
- `step5_evaluation_report.md`: comprehensive evaluation report (metrics + ROC/PR + stability + explainability)
- `step5__churn_customer_profile.txt`: segment-based churn profile with numeric support

### `outputs/plots/`
- `step2_<feature>_hist.png`: histograms for numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`)
- `step2_<feature>_pie.png`: pie charts for categorical distributions (e.g., `Contract`, `PaymentMethod`)
- `step5_pr_curve.png`: PR curve comparing all models
- `step5_feature_importance.png`: Top10 churn drivers bar chart
- `step5_shap_summary.png`: SHAP summary plot for the best model

## Data dictionary (Telco churn common fields)

| Column | Type (raw) | Meaning | Notes |
|---|---:|---|---|
| customerID | string | Unique customer identifier | ID only (not a feature) |
| gender | category | Customer gender | |
| SeniorCitizen | int / category | Senior citizen flag | step2 normalizes 0/1 -> Yes/No |
| Partner | category | Has partner | Yes/No |
| Dependents | category | Has dependents | Yes/No |
| tenure | int | Tenure (months) | |
| PhoneService | category | Phone service | Yes/No |
| MultipleLines | category | Multiple lines | includes "No phone service" |
| InternetService | category | Internet service type | DSL / Fiber optic / No |
| OnlineSecurity | category | Online security add-on | includes "No internet service" |
| OnlineBackup | category | Online backup add-on | includes "No internet service" |
| DeviceProtection | category | Device protection add-on | includes "No internet service" |
| TechSupport | category | Tech support add-on | includes "No internet service" |
| StreamingTV | category | Streaming TV add-on | includes "No internet service" |
| StreamingMovies | category | Streaming movies add-on | includes "No internet service" |
| Contract | category | Contract type | Month-to-month / One year / Two year |
| PaperlessBilling | category | Paperless billing | Yes/No |
| PaymentMethod | category | Payment method | |
| MonthlyCharges | float | Monthly charge | |
| TotalCharges | string/float | Total charge | raw can include blanks; step2 converts to numeric |
| Churn | category | Churn label | Yes/No |

Missing values:
- `TotalCharges` can contain blank strings in the original dataset; step2 converts them to NaN and applies a tenure-aware fill.

## Step scripts overview (step1–step5)

- **step1_data_exploration.py**
  - Loads raw data from `inputs/`
  - Prints key distributions and data-quality checks
  - Generates automated EDA HTML report: `outputs/reports/step1__auto_eda_report.html` (via `ydata-profiling`)

- **step2_preprocess.py**
  - Cleans the raw table (dedup, `TotalCharges` conversion, label encoding)
  - Missing values: numeric -> median/mean, categorical -> mode (configurable)
  - Encoding:
    - high-frequency categories (>5% by default) -> one-hot
    - low-frequency categories (<=5%) -> target encoding (smoothed)
  - Scaling: StandardScaler or MinMaxScaler (configurable)
  - Outputs:
    - cleaned table: `outputs/csv/step2_preprocess__telco_cleaned.csv`
    - model-ready dataset: `outputs/csv/step2__model_ready_dataset.csv`

- **step3_feature_engineering.py**
  - Reads `step2__model_ready_dataset.csv`
  - Optional Featuretools DFS to create interaction features
  - Feature selection:
    - SelectKBest (ANOVA F-test) or PCA (configurable)
  - Required engineered output:
    - `outputs/csv/step3__engineered_features.csv`

- **step4_train_models.py**
  - Stratified train/test split (configurable ratio + random seed)
  - Trains: Logistic Regression, Decision Tree, Random Forest
  - Saves `.joblib` models and split CSVs

- **step5_evaluate_models.py**
  - Evaluates all models: Accuracy / Precision / Recall / F1 / AUC
  - Generates plots: confusion matrices + ROC curves
  - SHAP explainability for the best model (configurable)
  - Writes a segment-based “high-risk customer profile” report:
    - `outputs/reports/step5__churn_customer_profile.txt`

## Metrics definitions (classification)

- **Accuracy**: \((TP+TN)/(TP+TN+FP+FN)\)
- **Precision**: \(TP/(TP+FP)\)
- **Recall**: \(TP/(TP+FN)\)
- **F1**: harmonic mean of precision and recall
- **AUC**: area under ROC curve using predicted scores/probabilities

## Testing

Run all tests:

```bash
python -m pytest -q
```

## Example results (illustration)

After running `python main.py`, check `outputs/reports/step5_evaluation_report.md` for the latest metrics.
For example, a typical run might show something like:
- AUC ≈ 0.82–0.86
- F1 ≈ 0.52–0.65

And the key plots include:
- Confusion matrices and ROC curves under `outputs/plots/`
- `step5_pr_curve.png` (PR curve)

