from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def resolve_shap_values(shap_values_obj) -> np.ndarray:
    if isinstance(shap_values_obj, list):
        if len(shap_values_obj) >= 2:
            values = shap_values_obj[1]
        else:
            values = shap_values_obj[0]
    elif hasattr(shap_values_obj, "values"):
        values = shap_values_obj.values
    else:
        values = shap_values_obj

    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] > 1:
        values = values[:, :, 1]
    elif values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]

    return values


def main() -> None:
    project_name = "Predicting Startup Success Using ML"
    project_root = Path(__file__).resolve().parent

    model_path = project_root / "models" / "final_model.joblib"
    metadata_path = project_root / "models" / "final_model_metadata.json"
    data_path = project_root / "data" / "processed" / "startup_dataset_baseline_ready.csv"

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    global_csv_path = reports_dir / "shap_feature_importance.csv"
    local_csv_path = reports_dir / "shap_local_explanations.csv"
    shap_bar_plot_path = reports_dir / "shap_summary_bar.png"
    shap_beeswarm_plot_path = reports_dir / "shap_summary_beeswarm.png"
    summary_md_path = reports_dir / "shap_report.md"
    summary_json_path = reports_dir / "shap_summary.json"

    if not model_path.exists() or not metadata_path.exists() or not data_path.exists():
        raise FileNotFoundError("Model, metadata, or baseline dataset is missing.")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_features = metadata.get("features", [])

    if not expected_features:
        raise ValueError("No feature list found in metadata.")

    data = pd.read_csv(data_path)

    identifiers = pd.DataFrame(index=data.index)
    if "startup_id" in data.columns:
        identifiers["startup_id"] = data["startup_id"]
    if "startup_name" in data.columns:
        identifiers["startup_name"] = data["startup_name"]

    x = data.drop(columns=[c for c in ["has_exit", "startup_id", "startup_name"] if c in data.columns])

    bool_columns = x.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        x[bool_columns] = x[bool_columns].astype(int)

    for feature in expected_features:
        if feature not in x.columns:
            x[feature] = 0

    x = x[expected_features]
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    classifier = model
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]

    if hasattr(classifier, "predict_proba") and classifier.__class__.__name__.lower().endswith(("forestclassifier", "treesclassifier")):
        explainer = shap.TreeExplainer(classifier)
        shap_values_obj = explainer.shap_values(x)
        shap_values = resolve_shap_values(shap_values_obj)
        expected_value = explainer.expected_value
    else:
        explainer = shap.Explainer(model, x)
        shap_values_obj = explainer(x)
        shap_values = resolve_shap_values(shap_values_obj)
        expected_value = getattr(explainer, "expected_value", None)

    if shap_values.shape != x.shape:
        raise ValueError(f"Unexpected SHAP matrix shape {shap_values.shape}, expected {x.shape}.")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_signed_shap = np.mean(shap_values, axis=0)

    global_importance = pd.DataFrame(
        {
            "feature": expected_features,
            "mean_abs_shap": mean_abs_shap,
            "mean_signed_shap": mean_signed_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap.summary_plot(shap_values, x, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(shap_bar_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, x, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(shap_beeswarm_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    global_importance.to_csv(global_csv_path, index=False)

    top_n = min(5, len(expected_features))
    local_rows: list[dict] = []
    for row_index in range(x.shape[0]):
        row_values = shap_values[row_index]
        top_indices = np.argsort(np.abs(row_values))[::-1][:top_n]

        row_dict: dict[str, object] = {
            "row_index": int(row_index),
        }
        if "startup_id" in identifiers.columns:
            row_dict["startup_id"] = identifiers.iloc[row_index]["startup_id"]
        if "startup_name" in identifiers.columns:
            row_dict["startup_name"] = identifiers.iloc[row_index]["startup_name"]

        for rank, feature_index in enumerate(top_indices, start=1):
            feature_name = expected_features[feature_index]
            row_dict[f"top{rank}_feature"] = feature_name
            row_dict[f"top{rank}_shap"] = float(row_values[feature_index])
            row_dict[f"top{rank}_feature_value"] = float(x.iloc[row_index, feature_index])

        local_rows.append(row_dict)

    local_explanations = pd.DataFrame(local_rows)
    local_explanations.to_csv(local_csv_path, index=False)

    summary_lines = [
        "# SHAP Explainability Report",
        "",
        f"- Project: {project_name}",
        f"- Model: {metadata.get('model_name', 'Unknown')}",
        f"- Rows explained: {x.shape[0]}",
        f"- Features explained: {x.shape[1]}",
        "",
        "## Top Global SHAP Features",
    ]

    for _, row in global_importance.head(10).iterrows():
        summary_lines.append(
            f"- {row['feature']}: mean_abs_shap={row['mean_abs_shap']:.6f}, mean_signed_shap={row['mean_signed_shap']:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "## Output Files",
            f"- Global feature importance: {global_csv_path}",
            f"- Local explanations: {local_csv_path}",
            f"- SHAP bar plot: {shap_bar_plot_path}",
            f"- SHAP beeswarm plot: {shap_beeswarm_plot_path}",
        ]
    )

    summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    payload = {
        "project": project_name,
        "model_name": metadata.get("model_name", "Unknown"),
        "rows_explained": int(x.shape[0]),
        "features_explained": int(x.shape[1]),
        "expected_value": str(expected_value),
        "top_global_features": global_importance.head(10).to_dict(orient="records"),
        "global_csv": str(global_csv_path),
        "local_csv": str(local_csv_path),
        "bar_plot": str(shap_bar_plot_path),
        "beeswarm_plot": str(shap_beeswarm_plot_path),
        "summary_md": str(summary_md_path),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("SHAP analysis completed.")
    print(f"Global feature SHAP: {global_csv_path}")
    print(f"Local explanations: {local_csv_path}")
    print(f"SHAP bar plot: {shap_bar_plot_path}")
    print(f"SHAP beeswarm plot: {shap_beeswarm_plot_path}")
    print(f"Summary report: {summary_md_path}")
    print("\nTop global SHAP features:")
    print(global_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
