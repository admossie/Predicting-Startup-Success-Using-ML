# Baseline Modeling Report

## Data
- Input dataset: C:\Users\Owner\source\my-new-project\data\processed\startup_dataset_baseline_ready.csv
- Rows: 10
- Features used: 40
- Target: has_exit

## Train/Test Split
- Train rows: 7
- Test rows: 3

## Model Performance (Test)
### LogisticRegression
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1: 1.0000
- ROC-AUC: 1.0
- PR-AUC: 1.0
- Brier score: 0.1168
- ECE (5-bin): 0.3241
- Repeated CV setup: 2 folds x 5 repeats
- Repeated CV Accuracy (train, mean±std): 1.0000 ± 0.0000
- Repeated CV F1 (train, mean±std): 1.0000 ± 0.0000
- Repeated CV ROC-AUC (train, mean±std): 1.0000 ± 0.0000
- Repeated CV PR-AUC (train, mean±std): 1.0000 ± 0.0000

### RandomForest
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1: 1.0000
- ROC-AUC: 1.0
- PR-AUC: 1.0
- Brier score: 0.0870
- ECE (5-bin): 0.2944
- Repeated CV setup: 2 folds x 5 repeats
- Repeated CV Accuracy (train, mean±std): 0.7833 ± 0.1453
- Repeated CV F1 (train, mean±std): 0.3000 ± 0.4583
- Repeated CV ROC-AUC (train, mean±std): 1.0000 ± 0.0000
- Repeated CV PR-AUC (train, mean±std): 1.0000 ± 0.0000

## Best Baseline
- Model: LogisticRegression
- F1: 1.0000
- Accuracy: 1.0000

## Out-of-Fold Diagnostics (full dataset)
### LogisticRegression
- OOF Accuracy: 0.9000
- OOF Precision: 1.0000
- OOF Recall: 0.6667
- OOF F1: 0.8000
- OOF ROC-AUC: 0.9523809523809524
- OOF PR-AUC: 0.9166666666666665
- OOF Brier: 0.0830
- OOF ECE (5-bin): 0.0764

### RandomForest
- OOF Accuracy: 1.0000
- OOF Precision: 1.0000
- OOF Recall: 1.0000
- OOF F1: 1.0000
- OOF ROC-AUC: 1.0
- OOF PR-AUC: 1.0
- OOF Brier: 0.0533
- OOF ECE (5-bin): 0.1747

## Confusion Matrix (OOF)
- LogisticRegression: TN=7, FP=0, FN=1, TP=2
- RandomForest: TN=7, FP=0, FN=0, TP=3

## Top Features (combined rank)
- employees: logistic_abs_coef=0.2168, rf_importance=0.0917
- team_size: logistic_abs_coef=0.2182, rf_importance=0.0752
- startup_age_years: logistic_abs_coef=0.2166, rf_importance=0.0752
- latest_valuation_usd: logistic_abs_coef=0.1910, rf_importance=0.0987
- remote_friendly: logistic_abs_coef=0.2544, rf_importance=0.0436
- total_funding_usd: logistic_abs_coef=0.1904, rf_importance=0.1053
- founded_year: logistic_abs_coef=0.2166, rf_importance=0.0498
- log1p_total_funding_usd: logistic_abs_coef=0.1643, rf_importance=0.0925
- funding_rounds: logistic_abs_coef=0.1907, rf_importance=0.0501
- funding_stage_rank: logistic_abs_coef=0.1907, rf_importance=0.0654