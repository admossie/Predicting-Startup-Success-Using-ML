# Predicting Startup Success Using ML — Project Summary

## Abstract
This project builds an end-to-end machine learning pipeline to predict startup success using integrated Crunchbase, AngelList, and PitchBook-style datasets. The workflow covers data integration, cleaning, exploratory analysis, outlier handling, feature engineering, baseline modeling, hyperparameter tuning, final model selection, inference, and SHAP explainability.

## Problem Statement
Goal: predict whether a startup will be successful (`has_exit` as target) using company, funding, and market-related features.

## Data and Preparation
- Source integration from three startup-data tables into a unified dataset.
- Processed dataset used for modeling: 10 rows, 40 features.
- Preparation steps included:
  - duplicate handling,
  - missing-value imputation,
  - IQR-based outlier capping,
  - encoded categorical variables,
  - leakage-safe feature set for baseline/final modeling.

## Modeling Approach
Compared multiple models with hyperparameter tuning (`GridSearchCV`, scoring=`f1`):
- Logistic Regression
- Random Forest
- Extra Trees
- SVC

Selection logic:
- repeated stratified cross-validation (2 folds × 5 repeats),
- ranking by `selection_score = cv_f1_mean * test_f1`, then CV/Test tie-break metrics.

## Final Results
Selected final model: **ExtraTrees**

Key metrics (winner):
- CV F1: 0.7000
- CV PR-AUC: 1.0000
- Test F1: 1.0000
- Selection score: 0.7000

Top SHAP global drivers:
1. `funding_stage_rank`
2. `employees`
3. `last_funding_stage_Series B`
4. `funding_rounds`
5. `team_size`

## Inference Example
Scoring new startups produced:
- `QuantumLeaf`: probability 0.021, label 0
- `RoboLedger`: probability 0.090, label 0
- `SolarMesh`: probability 0.943, label 1

## Artifacts
- Final model: `models/final_model.joblib`
- Metadata: `models/final_model_metadata.json`
- Leaderboard: `reports/model_selection_leaderboard.csv`
- Final selection report: `reports/model_selection_report.md`
- Prediction output: `reports/new_startup_predictions.csv`
- SHAP report: `reports/shap_report.md`
- SHAP plots: `reports/shap_summary_bar.png`, `reports/shap_summary_beeswarm.png`

## Conclusion
The project demonstrates a complete and interpretable ML workflow for startup success prediction. The final model and explainability outputs are ready for demonstration and extension. Because the current dataset is very small, results should be treated as directional; expanding to larger real-world startup records is the next priority for stronger generalization and reliability.
