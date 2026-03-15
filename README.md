# Predicting Startup Success Using ML

Machine learning pipeline to predict startup success using Crunchbase, Wellfound (AngelList), and PitchBook-style data.

## CI Status
[![CI](https://github.com/admossie/Predicting-Startup-Success-Using-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/admossie/Predicting-Startup-Success-Using-ML/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/admossie/Predicting-Startup-Success-Using-ML)](https://github.com/admossie/Predicting-Startup-Success-Using-ML/commits/main)

## Prerequisites
- Python 3.9+
- pip
- Git

## Installation (Windows)
```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Structure
- `build_dataset.py` — Merge source datasets into one unified table.
- `prepare_data.py` — Cleaning, EDA, missing-value handling, outlier treatment, feature engineering.
- `train_baseline.py` — Baseline modeling and diagnostics.
- `select_final_model.py` — Multi-model tuning and final model selection.
- `predict_startup.py` — Score new startup records.
- `run_shap_analysis.py` — SHAP explainability pipeline.

## Dataset and Key Files
> Dataset is synthetic/educational. Replace with real commercial data for production use.

### Data source references
- Crunchbase: https://www.crunchbase.com/
- Wellfound (AngelList): https://wellfound.com/
- PitchBook: https://pitchbook.com/

### Processed files
- `data/processed/startup_dataset.csv`
- `data/processed/startup_dataset_cleaned.csv`
- `data/processed/startup_dataset_baseline_ready.csv`
- `data/processed/new_startups_to_score.csv`

## Run End-to-End
```bash
python build_dataset.py
python prepare_data.py
python train_baseline.py
python select_final_model.py
```

## Score New Startups
```bash
python predict_startup.py --input data/processed/new_startups_to_score.csv --output reports/new_startup_predictions.csv
```

## SHAP Explainability
```bash
python run_shap_analysis.py
```

This generates:
- `reports/shap_feature_importance.csv` (global SHAP importance)
- `reports/shap_local_explanations.csv` (row-level top contributors)
- `reports/shap_report.md` (summary report)

## SHAP Visuals
### Global Feature Importance (Bar)
![SHAP Bar Plot](reports/shap_summary_bar.png)

### Feature Impact Distribution (Beeswarm)
![SHAP Beeswarm Plot](reports/shap_summary_beeswarm.png)

### Sample Prediction Output (optional)
![Prediction Output Preview](reports/new_startup_predictions_preview.png)

## How to Interpret SHAP
- **Bar plot**: higher bar means higher global importance.
- **Beeswarm**: each dot is one startup; left lowers prediction, right raises prediction.
- **Color**: blue = lower feature value, red = higher feature value.
- Pair with `reports/shap_local_explanations.csv` for row-level explanations.

## Model Performance

_Best model: **ExtraTrees**_

| Metric | Validation (CV Mean) | Test |
|---|---:|---:|
| Accuracy | 0.9083 | 1.0000 |
| Precision | N/A | 1.0000 |
| Recall | N/A | 1.0000 |
| F1-score | 0.7000 | 1.0000 |
| ROC-AUC | 1.0000 | 1.0000 |

> Source: `reports/model_selection_leaderboard.csv` (ExtraTrees row; `cv_*_mean` for validation and `test_*` for test).

## Tests
```powershell
pytest -q
```

Run functional tests only:
```powershell
pytest -q -m functional
```

## Outputs
- `models/final_model.joblib`
- `models/final_model_metadata.json`
- `reports/model_selection_leaderboard.csv`
- `reports/model_selection_report.md`
- `reports/new_startup_predictions.csv`
- `reports/shap_feature_importance.csv`
- `reports/shap_local_explanations.csv`
- `reports/shap_report.md`

## Reproducibility Notes
- Random seeds fixed across train/validation splits.
- Model artifacts are versioned in `models/`.
- Reports are stored in `reports/`.
- Use relative paths to keep scripts portable.

## Limitations
- Current dataset is very small (~10 rows); results are directional.
- Performance should be re-validated on larger real-world data.
- Educational/research scope unless production validation is completed.

## Conclusion
- Built a full ML pipeline for startup success prediction.
- Final selected model: `ExtraTrees`.
- Top SHAP drivers include `funding_stage_rank`, `employees`, `funding_rounds`, `team_size`, and `total_funding_usd`.
- Inference supports production-style CSV scoring.

## License
[MIT](LICENSE) © 2026 admossie