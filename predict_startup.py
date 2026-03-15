from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "founded_year" in data.columns and "startup_age_years" not in data.columns:
        data["startup_age_years"] = 2026 - pd.to_numeric(data["founded_year"], errors="coerce")

    stage_map = {
        "Pre-Seed": 0,
        "Seed": 1,
        "Series A": 2,
        "Series B": 3,
        "Series C": 4,
        "Series D": 5,
        "Series E": 6,
    }
    if "last_funding_stage" in data.columns and "funding_stage_rank" not in data.columns:
        data["funding_stage_rank"] = data["last_funding_stage"].astype(str).str.strip().str.title().map(stage_map).fillna(0)

    for source_col in ["total_funding_usd", "latest_valuation_usd", "monthly_web_visits", "followers"]:
        target_col = f"log1p_{source_col}"
        if source_col in data.columns and target_col not in data.columns:
            data[target_col] = np.log1p(pd.to_numeric(data[source_col], errors="coerce").clip(lower=0))

    for col in ["industry", "country", "last_funding_stage"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.title()

    return data


def _to_model_features(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    prepared = _build_engineered_features(df)

    candidate = prepared.drop(columns=[c for c in ["startup_id", "startup_name", "has_exit"] if c in prepared.columns])

    one_hot_cols = [c for c in ["industry", "country", "last_funding_stage"] if c in candidate.columns]
    if one_hot_cols:
        candidate = pd.get_dummies(candidate, columns=one_hot_cols, drop_first=False)

    for bool_col in candidate.select_dtypes(include=["bool"]).columns:
        candidate[bool_col] = candidate[bool_col].astype(int)

    # Ensure all expected features exist and are numeric where possible
    aligned = pd.DataFrame(index=candidate.index)
    for feature in expected_features:
        if feature in candidate.columns:
            aligned[feature] = candidate[feature]
        else:
            aligned[feature] = 0

    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0)

    return aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score startup success probabilities using saved final model.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (raw features or baseline-ready rows).",
    )
    parser.add_argument(
        "--model",
        default="models/final_model.joblib",
        help="Path to saved model file (.joblib).",
    )
    parser.add_argument(
        "--metadata",
        default="models/final_model_metadata.json",
        help="Path to metadata JSON with expected feature list.",
    )
    parser.add_argument(
        "--output",
        default="reports/new_startup_predictions.csv",
        help="Path to save prediction output CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for predicted label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    input_path = (project_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    model_path = (project_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    metadata_path = (project_root / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata)
    output_path = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_features = payload.get("features", [])
    if not expected_features:
        raise ValueError("Metadata does not contain expected feature list.")

    model = joblib.load(model_path)

    raw_df = pd.read_csv(input_path)
    x = _to_model_features(raw_df, expected_features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[:, 1]
    else:
        probabilities = model.predict(x).astype(float)

    labels = (probabilities >= args.threshold).astype(int)

    result = raw_df.copy()
    result["predicted_probability_success"] = probabilities
    result["predicted_label"] = labels

    output_cols = []
    for col in ["startup_id", "startup_name"]:
        if col in result.columns:
            output_cols.append(col)
    output_cols.extend(["predicted_probability_success", "predicted_label"])

    final_df = result[output_cols] if output_cols else result[["predicted_probability_success", "predicted_label"]]
    final_df.to_csv(output_path, index=False)

    print("Prediction completed.")
    print(f"Input: {input_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print("\nPreview:")
    print(final_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
