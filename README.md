# Telco Customer Churn (7810)

This project implements an end-to-end Telco customer churn prediction pipeline in 5 steps and generates datasets, reports, models, and plots under `outputs/`.

## 环境准备

```bash
python -m pip install -r requirements.txt
```

## 数据文件

Put the raw dataset under the project root `inputs/` folder (recommended filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
If your file is elsewhere, you can run `python main.py --csv "path\\to\\csv"` and it will be copied into `inputs/`.

## How to run (recommended)

```bash
python main.py
```

## Outputs (all under `outputs/`)

- `outputs/csv/`: all CSV outputs (filenames include the step prefix)
- `outputs/reports/`: text/json reports (filenames include the step prefix)
- `outputs/models/`: trained models (`.joblib`)
- `outputs/plots/`: confusion matrices and ROC curves (`.png`)

