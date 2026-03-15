from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    outlier_report: dict[str, dict[str, float]] = {}

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            outlier_report[column] = {
                "q1": float(q1) if pd.notna(q1) else np.nan,
                "q3": float(q3) if pd.notna(q3) else np.nan,
                "lower_bound": float(q1) if pd.notna(q1) else np.nan,
                "upper_bound": float(q3) if pd.notna(q3) else np.nan,
                "outliers_before": 0,
                "outliers_after": 0,
            }
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_before = int(((series < lower_bound) | (series > upper_bound)).sum())

        clipped = series.clip(lower=lower_bound, upper=upper_bound)
        outliers_after = int(((clipped < lower_bound) | (clipped > upper_bound)).sum())

        df[column] = clipped
        outlier_report[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers_before": outliers_before,
            "outliers_after": outliers_after,
        }

    return df, outlier_report


def prepare_dataset(project_root: Path) -> None:
    input_path = project_root / "data" / "processed" / "startup_dataset.csv"
    cleaned_path = project_root / "data" / "processed" / "startup_dataset_cleaned.csv"
    model_ready_path = project_root / "data" / "processed" / "startup_dataset_model_ready.csv"
    baseline_ready_path = project_root / "data" / "processed" / "startup_dataset_baseline_ready.csv"
    report_path = project_root / "reports" / "eda_report.md"
    report_json_path = project_root / "reports" / "eda_summary.json"

    df = pd.read_csv(input_path)
    original_shape = df.shape

    df.columns = [column.strip() for column in df.columns]

    text_columns = ["startup_name", "industry", "country", "last_funding_stage", "exit_type"]
    df = normalize_text_columns(df, text_columns)

    duplicates_removed = int(df.duplicated(subset=["startup_id"]).sum())
    df = df.drop_duplicates(subset=["startup_id"]).reset_index(drop=True)

    missing_before = df.isna().sum().to_dict()

    # Preserve missing flags before imputation for downstream modeling signal
    df["years_to_exit_missing_flag"] = df["years_to_exit"].isna().astype(int)
    df["exit_type_missing_flag"] = df["exit_type"].isna().astype(int)

    # Domain-aware imputation for exit-related fields
    if "has_exit" in df.columns:
        df.loc[df["has_exit"] == 0, "exit_type"] = "No Exit"

        exited_mask = df["has_exit"] == 1
        exited_years_median = df.loc[exited_mask, "years_to_exit"].median()
        if pd.isna(exited_years_median):
            exited_years_median = 0

        df.loc[~exited_mask, "years_to_exit"] = 0
        df.loc[exited_mask, "years_to_exit"] = df.loc[exited_mask, "years_to_exit"].fillna(exited_years_median)
        df.loc[exited_mask & df["exit_type"].isna(), "exit_type"] = "Other Exit"

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    numeric_impute_columns = [column for column in numeric_columns if column not in ["years_to_exit"]]
    for column in numeric_impute_columns:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    categorical_impute_columns = [column for column in categorical_columns if column not in ["exit_type"]]
    for column in categorical_impute_columns:
        mode = df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)

    df["last_funding_stage"] = df["last_funding_stage"].str.title()
    df["industry"] = df["industry"].str.title()
    df["country"] = df["country"].str.title()
    df["exit_type"] = df["exit_type"].str.title()

    outlier_columns = [
        "total_funding_usd",
        "employees",
        "team_size",
        "followers",
        "job_openings",
        "monthly_web_visits",
        "investor_count",
        "latest_valuation_usd",
        "years_to_exit",
    ]

    df, outlier_report = cap_outliers_iqr(df, [c for c in outlier_columns if c in df.columns])

    # Additional preparation for modeling
    stage_map = {
        "Pre-Seed": 0,
        "Seed": 1,
        "Series A": 2,
        "Series B": 3,
        "Series C": 4,
        "Series D": 5,
        "Series E": 6,
    }
    df["funding_stage_rank"] = df["last_funding_stage"].map(stage_map).fillna(0).astype(int)

    for column in ["total_funding_usd", "latest_valuation_usd", "monthly_web_visits", "followers"]:
        if column in df.columns:
            df[f"log1p_{column}"] = np.log1p(df[column])

    missing_after = df.isna().sum().to_dict()
    cleaned_shape = df.shape

    numeric_describe = df.describe(include=["number"]).round(3)
    categorical_describe = df.describe(include=["object", "string"]).transpose()

    corr_with_target = {}
    if "has_exit" in df.columns:
        numeric_df = df.select_dtypes(include=["number"])
        correlations = numeric_df.corr(numeric_only=True)["has_exit"].drop(labels=["has_exit"], errors="ignore")
        correlations = correlations.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        corr_with_target = {k: float(v) for k, v in correlations.items()}

    # Save cleaned dataset
    df.to_csv(cleaned_path, index=False)

    # Save model-ready data (one-hot for categorical features)
    model_ready = pd.get_dummies(
        df,
        columns=["industry", "country", "last_funding_stage", "exit_type"],
        drop_first=False,
    )
    model_ready.to_csv(model_ready_path, index=False)

    # Save baseline-ready data (drops leakage-prone post-outcome columns)
    leakage_columns = [
        "exit_type",
        "years_to_exit",
        "years_to_exit_missing_flag",
        "exit_type_missing_flag",
    ]
    baseline_df = df.drop(columns=[column for column in leakage_columns if column in df.columns])
    baseline_ready = pd.get_dummies(
        baseline_df,
        columns=["industry", "country", "last_funding_stage"],
        drop_first=False,
    )
    baseline_ready.to_csv(baseline_ready_path, index=False)

    report_lines = [
        "# Startup Dataset EDA & Preparation Report",
        "",
        "## 1) Dataset Overview",
        f"- Original shape: {original_shape}",
        f"- Cleaned shape: {cleaned_shape}",
        f"- Duplicates removed (by startup_id): {duplicates_removed}",
        "",
        "## 2) Missing Values",
        "### Before imputation",
    ]

    for key, value in missing_before.items():
        report_lines.append(f"- {key}: {value}")

    report_lines.extend(["", "### After imputation"])
    for key, value in missing_after.items():
        report_lines.append(f"- {key}: {value}")

    report_lines.extend(["", "## 3) Outlier Treatment (IQR Capping)"])
    for column, details in outlier_report.items():
        report_lines.append(
            f"- {column}: outliers_before={details['outliers_before']}, "
            f"outliers_after={details['outliers_after']}, "
            f"bounds=({details['lower_bound']:.2f}, {details['upper_bound']:.2f})"
        )

    report_lines.extend(["", "## 4) Numeric Summary", "```", numeric_describe.to_string(), "```"])

    report_lines.extend(["", "## 5) Categorical Summary", "```", categorical_describe.to_string(), "```"])

    report_lines.extend(["", "## 6) Correlation with has_exit (numeric features)"])
    if corr_with_target:
        for key, value in corr_with_target.items():
            report_lines.append(f"- {key}: {value:.4f}")
    else:
        report_lines.append("- Target column has_exit not found or no numeric correlations available.")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary_payload = {
        "original_shape": original_shape,
        "cleaned_shape": cleaned_shape,
        "duplicates_removed": duplicates_removed,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "outlier_report": outlier_report,
        "correlation_with_has_exit": corr_with_target,
        "cleaned_dataset": str(cleaned_path),
        "model_ready_dataset": str(model_ready_path),
        "baseline_ready_dataset": str(baseline_ready_path),
        "report_markdown": str(report_path),
    }
    report_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("EDA and data preparation completed.")
    print(f"Cleaned dataset: {cleaned_path}")
    print(f"Model-ready dataset: {model_ready_path}")
    print(f"Baseline-ready dataset (leakage-safe): {baseline_ready_path}")
    print(f"EDA report: {report_path}")
    print("\nCleaned shape:", cleaned_shape)
    print("\nCleaned head:")
    print(df.head())
    print("\nCleaned tail:")
    print(df.tail())
    print("\nCleaned describe (numeric):")
    print(df.describe(include=["number"]))


if __name__ == "__main__":
    prepare_dataset(Path(__file__).resolve().parent)
