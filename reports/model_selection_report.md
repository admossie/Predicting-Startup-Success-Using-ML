# Predicting Startup Success Using ML - Final Model Selection Report

## Data
- Input dataset: C:\Users\Owner\source\my-new-project\data\processed\startup_dataset_baseline_ready.csv
- Rows: 10
- Features: 40
- Target: has_exit

## Selection Strategy
- Compared multiple models with hyperparameter tuning (GridSearchCV, scoring=f1).
- Ranked models by selection_score = CV_F1 x Test_F1, then CV F1, test F1, CV PR-AUC, test PR-AUC.
- CV setup: 2 folds x 5 repeats (repeated stratified CV for stability).

## Leaderboard (Top Models)
### ExtraTrees
- Tuning best F1: 1.0000
- Selection score (CV_F1 x Test_F1): 0.7000
- CV F1: 0.7000 ± 0.4583
- CV PR-AUC: 1.0000 ± 0.0000
- Test F1: 1.0000
- Test PR-AUC: 1.0
- Best params: {"classifier__max_depth": 3, "classifier__min_samples_split": 2, "classifier__n_estimators": 100}

### LogisticRegression
- Tuning best F1: 0.8333
- Selection score (CV_F1 x Test_F1): 0.6444
- CV F1: 0.9667 ± 0.1000
- CV PR-AUC: 1.0000 ± 0.0000
- Test F1: 0.6667
- Test PR-AUC: 1.0
- Best params: {"classifier__C": 1.0, "classifier__penalty": "l2"}

### RandomForest
- Tuning best F1: 0.5000
- Selection score (CV_F1 x Test_F1): 0.3000
- CV F1: 0.3000 ± 0.4583
- CV PR-AUC: 1.0000 ± 0.0000
- Test F1: 1.0000
- Test PR-AUC: 1.0
- Best params: {"classifier__max_depth": 3, "classifier__min_samples_leaf": 1, "classifier__min_samples_split": 2, "classifier__n_estimators": 100}

### SVC
- Tuning best F1: 1.0000
- Selection score (CV_F1 x Test_F1): 0.0000
- CV F1: 1.0000 ± 0.0000
- CV PR-AUC: 1.0000 ± 0.0000
- Test F1: 0.0000
- Test PR-AUC: 0.3333333333333333
- Best params: {"classifier__C": 0.5, "classifier__gamma": "scale", "classifier__kernel": "linear"}

## Selected Final Model
- Winner: ExtraTrees
- CV F1: 0.7000
- CV PR-AUC: 1.0000
- Test F1: 1.0000
- Saved model: C:\Users\Owner\source\my-new-project\models\final_model.joblib
- Metadata: C:\Users\Owner\source\my-new-project\models\final_model_metadata.json
- Predictions: C:\Users\Owner\source\my-new-project\reports\final_model_predictions.csv