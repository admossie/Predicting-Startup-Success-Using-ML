from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0

    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += abs(accuracy - confidence) * (np.sum(mask) / total)

    return float(ece)


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    return float(value)


def _cv_folds_from_y(y: pd.Series) -> int:
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    return min(3, min_class_count)


def repeated_cv_summary(model: Pipeline, x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    folds = _cv_folds_from_y(y)
    if folds < 2:
        return {
            "repeated_cv_folds": 0,
            "repeated_cv_repeats": 0,
            "repeated_cv_accuracy_mean": None,
            "repeated_cv_accuracy_std": None,
            "repeated_cv_f1_mean": None,
            "repeated_cv_f1_std": None,
            "repeated_cv_roc_auc_mean": None,
            "repeated_cv_roc_auc_std": None,
            "repeated_cv_pr_auc_mean": None,
            "repeated_cv_pr_auc_std": None,
        }

    repeats = 10 if len(y) >= 30 else 5
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    scores = cross_validate(model, x, y, cv=cv, scoring=scoring, error_score=np.nan)

    return {
        "repeated_cv_folds": int(folds),
        "repeated_cv_repeats": int(repeats),
        "repeated_cv_accuracy_mean": _safe_float(np.nanmean(scores["test_accuracy"])),
        "repeated_cv_accuracy_std": _safe_float(np.nanstd(scores["test_accuracy"])),
        "repeated_cv_f1_mean": _safe_float(np.nanmean(scores["test_f1"])),
        "repeated_cv_f1_std": _safe_float(np.nanstd(scores["test_f1"])),
        "repeated_cv_roc_auc_mean": _safe_float(np.nanmean(scores["test_roc_auc"])),
        "repeated_cv_roc_auc_std": _safe_float(np.nanstd(scores["test_roc_auc"])),
        "repeated_cv_pr_auc_mean": _safe_float(np.nanmean(scores["test_pr_auc"])),
        "repeated_cv_pr_auc_std": _safe_float(np.nanstd(scores["test_pr_auc"])),
    }


def oof_diagnostics(name: str, model: Pipeline, x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[dict, dict]:
    folds = _cv_folds_from_y(y)
    if folds < 2:
        metrics = {
            "model": name,
            "oof_accuracy": None,
            "oof_precision": None,
            "oof_recall": None,
            "oof_f1": None,
            "oof_roc_auc": None,
            "oof_pr_auc": None,
            "oof_brier": None,
            "oof_ece": None,
        }
        matrix = {"model": name, "tn": None, "fp": None, "fn": None, "tp": None}
        return metrics, matrix

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(x, y):
        x_tr = x.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        x_te = x.iloc[test_idx]

        model.fit(x_tr, y_tr)
        preds = model.predict(x_te)
        probs = model.predict_proba(x_te)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

        oof_pred[test_idx] = preds
        oof_prob[test_idx] = probs

    tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": name,
        "oof_accuracy": _safe_float(accuracy_score(y, oof_pred)),
        "oof_precision": _safe_float(precision_score(y, oof_pred, zero_division=0)),
        "oof_recall": _safe_float(recall_score(y, oof_pred, zero_division=0)),
        "oof_f1": _safe_float(f1_score(y, oof_pred, zero_division=0)),
        "oof_roc_auc": _safe_roc_auc(y, oof_prob),
        "oof_pr_auc": _safe_pr_auc(y, oof_prob),
        "oof_brier": _safe_float(brier_score_loss(y, oof_prob)),
        "oof_ece": _safe_float(_compute_ece(y.to_numpy(), oof_prob)),
    }
    matrix = {
        "model": name,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, matrix


def evaluate_model(name: str, model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_test, y_prob),
        "pr_auc": _safe_pr_auc(y_test, y_prob),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "ece": _compute_ece(y_test.to_numpy(), y_prob),
    }
    metrics.update(repeated_cv_summary(model, x_train, y_train, random_state=42))
    return metrics


def extract_feature_importance(
    logistic_pipeline: Pipeline,
    rf_pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    logistic_model = logistic_pipeline.named_steps["classifier"]
    rf_model = rf_pipeline.named_steps["classifier"]

    logistic_importance = np.abs(logistic_model.coef_[0])
    rf_importance = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "logistic_abs_coef": logistic_importance,
            "random_forest_importance": rf_importance,
        }
    )

    importance_df["combined_rank_score"] = (
        importance_df["logistic_abs_coef"].rank(ascending=False, method="average")
        + importance_df["random_forest_importance"].rank(ascending=False, method="average")
    )

    return importance_df.sort_values("combined_rank_score").reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "processed" / "startup_dataset_baseline_ready.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "has_exit" not in df.columns:
        raise ValueError("Target column 'has_exit' is missing from baseline dataset.")

    y = df["has_exit"].astype(int)

    drop_columns = ["has_exit", "startup_id", "startup_name"]
    x = df.drop(columns=[column for column in drop_columns if column in df.columns])

    bool_columns = x.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        x[bool_columns] = x[bool_columns].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    max_depth=5,
                ),
            ),
        ]
    )

    logistic_metrics = evaluate_model("LogisticRegression", logistic_pipeline, x_train, x_test, y_train, y_test)
    rf_metrics = evaluate_model("RandomForest", rf_pipeline, x_train, x_test, y_train, y_test)

    logistic_oof_metrics, logistic_oof_matrix = oof_diagnostics("LogisticRegression", logistic_pipeline, x, y)
    rf_oof_metrics, rf_oof_matrix = oof_diagnostics("RandomForest", rf_pipeline, x, y)

    # Fit once more to full train for importance extraction
    logistic_pipeline.fit(x_train, y_train)
    rf_pipeline.fit(x_train, y_train)

    feature_importance_df = extract_feature_importance(
        logistic_pipeline,
        rf_pipeline,
        feature_names=x.columns.tolist(),
    )

    metrics_df = pd.DataFrame([logistic_metrics, rf_metrics]).sort_values(
        by=["repeated_cv_f1_mean", "repeated_cv_pr_auc_mean", "f1", "accuracy"],
        ascending=False,
        na_position="last",
    )
    oof_metrics_df = pd.DataFrame([logistic_oof_metrics, rf_oof_metrics])
    confusion_df = pd.DataFrame([logistic_oof_matrix, rf_oof_matrix])

    metrics_csv_path = reports_dir / "baseline_metrics.csv"
    metrics_json_path = reports_dir / "baseline_metrics.json"
    oof_metrics_path = reports_dir / "baseline_oof_metrics.csv"
    confusion_path = reports_dir / "baseline_confusion_matrix.csv"
    importance_path = reports_dir / "baseline_feature_importance.csv"
    summary_md_path = reports_dir / "baseline_report.md"

    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_json_path.write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")
    oof_metrics_df.to_csv(oof_metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    feature_importance_df.to_csv(importance_path, index=False)

    best_model = metrics_df.iloc[0].to_dict()

    summary_lines = [
        "# Baseline Modeling Report",
        "",
        "## Data",
        f"- Input dataset: {data_path}",
        f"- Rows: {df.shape[0]}",
        f"- Features used: {x.shape[1]}",
        f"- Target: has_exit",
        "",
        "## Train/Test Split",
        f"- Train rows: {x_train.shape[0]}",
        f"- Test rows: {x_test.shape[0]}",
        "",
        "## Model Performance (Test)",
    ]

    for _, row in metrics_df.iterrows():
        summary_lines.extend(
            [
                f"### {row['model']}",
                f"- Accuracy: {row['accuracy']:.4f}",
                f"- Precision: {row['precision']:.4f}",
                f"- Recall: {row['recall']:.4f}",
                f"- F1: {row['f1']:.4f}",
                f"- ROC-AUC: {row['roc_auc'] if pd.notna(row['roc_auc']) else 'N/A'}",
                f"- PR-AUC: {row['pr_auc'] if pd.notna(row['pr_auc']) else 'N/A'}",
                f"- Brier score: {row['brier']:.4f}",
                f"- ECE (5-bin): {row['ece']:.4f}",
                f"- Repeated CV setup: {int(row['repeated_cv_folds'])} folds x {int(row['repeated_cv_repeats'])} repeats",
                f"- Repeated CV Accuracy (train, mean±std): {row['repeated_cv_accuracy_mean']:.4f} ± {row['repeated_cv_accuracy_std']:.4f}",
                f"- Repeated CV F1 (train, mean±std): {row['repeated_cv_f1_mean']:.4f} ± {row['repeated_cv_f1_std']:.4f}",
                f"- Repeated CV ROC-AUC (train, mean±std): {row['repeated_cv_roc_auc_mean']:.4f} ± {row['repeated_cv_roc_auc_std']:.4f}",
                f"- Repeated CV PR-AUC (train, mean±std): {row['repeated_cv_pr_auc_mean']:.4f} ± {row['repeated_cv_pr_auc_std']:.4f}",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Best Baseline",
            f"- Model: {best_model['model']}",
            f"- F1: {best_model['f1']:.4f}",
            f"- Accuracy: {best_model['accuracy']:.4f}",
            "",
            "## Out-of-Fold Diagnostics (full dataset)",
        ]
    )

    for _, row in oof_metrics_df.iterrows():
        summary_lines.extend(
            [
                f"### {row['model']}",
                f"- OOF Accuracy: {row['oof_accuracy']:.4f}",
                f"- OOF Precision: {row['oof_precision']:.4f}",
                f"- OOF Recall: {row['oof_recall']:.4f}",
                f"- OOF F1: {row['oof_f1']:.4f}",
                f"- OOF ROC-AUC: {row['oof_roc_auc'] if pd.notna(row['oof_roc_auc']) else 'N/A'}",
                f"- OOF PR-AUC: {row['oof_pr_auc'] if pd.notna(row['oof_pr_auc']) else 'N/A'}",
                f"- OOF Brier: {row['oof_brier']:.4f}",
                f"- OOF ECE (5-bin): {row['oof_ece']:.4f}",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Confusion Matrix (OOF)",
        ]
    )

    for _, row in confusion_df.iterrows():
        summary_lines.append(
            f"- {row['model']}: TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, TP={int(row['tp'])}"
        )

    summary_lines.extend(
        [
            "",
            "## Top Features (combined rank)",
        ]
    )

    for _, row in feature_importance_df.head(10).iterrows():
        summary_lines.append(
            f"- {row['feature']}: logistic_abs_coef={row['logistic_abs_coef']:.4f}, "
            f"rf_importance={row['random_forest_importance']:.4f}"
        )

    summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Baseline modeling completed.")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Metrics JSON: {metrics_json_path}")
    print(f"OOF metrics: {oof_metrics_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Summary report: {summary_md_path}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
