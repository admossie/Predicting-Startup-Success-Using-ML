from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _safe_score(func, y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(func(y_true, y_prob))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return value


def _ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    total = len(y_true)
    if total == 0:
        return 0.0

    error = 0.0
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        error += abs(accuracy - confidence) * (np.sum(mask) / total)
    return float(error)


def _cv_setup(y: pd.Series) -> tuple[int, int]:
    class_counts = y.value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    folds = min(3, min_class)
    repeats = 10 if len(y) >= 40 else 5
    return folds, repeats


def get_model_space(random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    model_space: dict[str, tuple[Pipeline, dict]] = {}

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=4000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    logistic_grid = {
        "classifier__C": [0.1, 1.0, 5.0, 10.0],
        "classifier__penalty": ["l1", "l2"],
    }
    model_space["LogisticRegression"] = (logistic, logistic_grid)

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf_grid = {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 4],
        "classifier__min_samples_leaf": [1, 2],
    }
    model_space["RandomForest"] = (rf, rf_grid)

    extra_trees = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    extra_trees_grid = {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 4],
    }
    model_space["ExtraTrees"] = (extra_trees, extra_trees_grid)

    svc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    svc_grid = {
        "classifier__C": [0.5, 1.0, 5.0],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
    }
    model_space["SVC"] = (svc, svc_grid)

    return model_space


def evaluate_holdout(estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | None]:
    y_pred = estimator.predict(x_test)
    y_prob = estimator.predict_proba(x_test)[:, 1] if hasattr(estimator, "predict_proba") else y_pred.astype(float)

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": _safe_score(roc_auc_score, y_test, y_prob),
        "test_pr_auc": _safe_score(average_precision_score, y_test, y_prob),
        "test_brier": float(brier_score_loss(y_test, y_prob)),
        "test_ece": _ece(y_test.to_numpy(), y_prob),
    }


def evaluate_repeated_cv(estimator: Pipeline, x: pd.DataFrame, y: pd.Series, folds: int, repeats: int, random_state: int) -> dict:
    if folds < 2:
        return {
            "cv_folds": 0,
            "cv_repeats": 0,
            "cv_accuracy_mean": None,
            "cv_accuracy_std": None,
            "cv_f1_mean": None,
            "cv_f1_std": None,
            "cv_roc_auc_mean": None,
            "cv_roc_auc_std": None,
            "cv_pr_auc_mean": None,
            "cv_pr_auc_std": None,
            "cv_brier_mean": None,
            "cv_brier_std": None,
        }

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "brier": "neg_brier_score",
    }
    result = cross_validate(estimator, x, y, cv=cv, scoring=scoring, error_score=np.nan)

    brier_scores = -result["test_brier"]

    return {
        "cv_folds": int(folds),
        "cv_repeats": int(repeats),
        "cv_accuracy_mean": _safe_float(np.nanmean(result["test_accuracy"])),
        "cv_accuracy_std": _safe_float(np.nanstd(result["test_accuracy"])),
        "cv_f1_mean": _safe_float(np.nanmean(result["test_f1"])),
        "cv_f1_std": _safe_float(np.nanstd(result["test_f1"])),
        "cv_roc_auc_mean": _safe_float(np.nanmean(result["test_roc_auc"])),
        "cv_roc_auc_std": _safe_float(np.nanstd(result["test_roc_auc"])),
        "cv_pr_auc_mean": _safe_float(np.nanmean(result["test_pr_auc"])),
        "cv_pr_auc_std": _safe_float(np.nanstd(result["test_pr_auc"])),
        "cv_brier_mean": _safe_float(np.nanmean(brier_scores)),
        "cv_brier_std": _safe_float(np.nanstd(brier_scores)),
    }


def main() -> None:
    random_state = 42
    project_name = "Predicting Startup Success Using ML"

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "processed" / "startup_dataset_baseline_ready.csv"

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "has_exit" not in df.columns:
        raise ValueError("Target column 'has_exit' is missing.")

    y = df["has_exit"].astype(int)
    x = df.drop(columns=[c for c in ["has_exit", "startup_id", "startup_name"] if c in df.columns])

    bool_cols = x.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        x[bool_cols] = x[bool_cols].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    folds, repeats = _cv_setup(y_train)
    if folds < 2:
        raise ValueError("Not enough minority-class samples in train split for stratified cross-validation.")

    tuning_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    model_space = get_model_space(random_state=random_state)
    leaderboard_rows: list[dict] = []
    best_estimators: dict[str, Pipeline] = {}

    for model_name, (pipeline, param_grid) in model_space.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=tuning_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        best_estimator = search.best_estimator_
        best_estimators[model_name] = best_estimator

        holdout_metrics = evaluate_holdout(best_estimator, x_test, y_test)
        cv_metrics = evaluate_repeated_cv(best_estimator, x_train, y_train, folds=folds, repeats=repeats, random_state=random_state)

        row = {
            "model": model_name,
            "best_params": json.dumps(search.best_params_),
            "tuning_best_f1": float(search.best_score_),
        }
        row.update(holdout_metrics)
        row.update(cv_metrics)
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard["selection_score"] = leaderboard["cv_f1_mean"] * leaderboard["test_f1"]
    leaderboard = leaderboard.sort_values(
        by=["selection_score", "cv_f1_mean", "test_f1", "cv_pr_auc_mean", "test_pr_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    winner_name = str(leaderboard.iloc[0]["model"])
    winner_estimator = best_estimators[winner_name]

    winner_estimator.fit(x, y)

    model_path = models_dir / "final_model.joblib"
    metadata_path = models_dir / "final_model_metadata.json"
    predictions_path = reports_dir / "final_model_predictions.csv"
    leaderboard_path = reports_dir / "model_selection_leaderboard.csv"
    summary_path = reports_dir / "model_selection_report.md"

    joblib.dump(winner_estimator, model_path)

    metadata = {
        "model_name": winner_name,
        "selected_by": ["selection_score=cv_f1_mean*test_f1", "cv_f1_mean", "test_f1", "cv_pr_auc_mean", "test_pr_auc"],
        "features": x.columns.tolist(),
        "target": "has_exit",
        "train_rows": int(len(x)),
        "cv_folds": int(folds),
        "cv_repeats": int(repeats),
        "data_path": str(data_path),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_prob = winner_estimator.predict_proba(x)[:, 1] if hasattr(winner_estimator, "predict_proba") else winner_estimator.predict(x)
    prediction_df = pd.DataFrame(
        {
            "startup_id": df.get("startup_id", pd.Series(range(len(df)))).values,
            "startup_name": df.get("startup_name", pd.Series(["Unknown"] * len(df))).values,
            "actual_has_exit": y.values,
            "predicted_probability_success": all_prob,
            "predicted_label": (all_prob >= 0.5).astype(int),
        }
    )
    prediction_df.to_csv(predictions_path, index=False)

    leaderboard.to_csv(leaderboard_path, index=False)

    winner_row = leaderboard.iloc[0]
    report_lines = [
        f"# {project_name} - Final Model Selection Report",
        "",
        "## Data",
        f"- Input dataset: {data_path}",
        f"- Rows: {df.shape[0]}",
        f"- Features: {x.shape[1]}",
        f"- Target: has_exit",
        "",
        "## Selection Strategy",
        "- Compared multiple models with hyperparameter tuning (GridSearchCV, scoring=f1).",
        "- Ranked models by selection_score = CV_F1 x Test_F1, then CV F1, test F1, CV PR-AUC, test PR-AUC.",
        f"- CV setup: {folds} folds x {repeats} repeats (repeated stratified CV for stability).",
        "",
        "## Leaderboard (Top Models)",
    ]

    for _, row in leaderboard.head(4).iterrows():
        report_lines.extend(
            [
                f"### {row['model']}",
                f"- Tuning best F1: {row['tuning_best_f1']:.4f}",
                f"- Selection score (CV_F1 x Test_F1): {row['selection_score']:.4f}",
                f"- CV F1: {row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}",
                f"- CV PR-AUC: {row['cv_pr_auc_mean']:.4f} ± {row['cv_pr_auc_std']:.4f}",
                f"- Test F1: {row['test_f1']:.4f}",
                f"- Test PR-AUC: {row['test_pr_auc'] if pd.notna(row['test_pr_auc']) else 'N/A'}",
                f"- Best params: {row['best_params']}",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Selected Final Model",
            f"- Winner: {winner_name}",
            f"- CV F1: {winner_row['cv_f1_mean']:.4f}",
            f"- CV PR-AUC: {winner_row['cv_pr_auc_mean']:.4f}",
            f"- Test F1: {winner_row['test_f1']:.4f}",
            f"- Saved model: {model_path}",
            f"- Metadata: {metadata_path}",
            f"- Predictions: {predictions_path}",
        ]
    )

    summary_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Final model selection completed.")
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Report: {summary_path}")
    print(f"Saved model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Predictions: {predictions_path}")
    print("\nTop leaderboard rows:")
    print(leaderboard.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
