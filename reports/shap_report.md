# SHAP Explainability Report

- Project: Predicting Startup Success Using ML
- Model: ExtraTrees
- Rows explained: 10
- Features explained: 40

## Top Global SHAP Features
- funding_stage_rank: mean_abs_shap=0.052839, mean_signed_shap=-0.024501
- employees: mean_abs_shap=0.049052, mean_signed_shap=-0.018484
- last_funding_stage_Series B: mean_abs_shap=0.041853, mean_signed_shap=-0.015382
- funding_rounds: mean_abs_shap=0.041212, mean_signed_shap=-0.019704
- team_size: mean_abs_shap=0.037649, mean_signed_shap=-0.015107
- total_funding_usd: mean_abs_shap=0.037088, mean_signed_shap=-0.014165
- startup_age_years: mean_abs_shap=0.035833, mean_signed_shap=-0.014619
- latest_valuation_usd: mean_abs_shap=0.031223, mean_signed_shap=-0.011315
- log1p_total_funding_usd: mean_abs_shap=0.025164, mean_signed_shap=-0.009032
- last_funding_stage_Series A: mean_abs_shap=0.022824, mean_signed_shap=-0.004583

## Output Files
- Global feature importance: C:\Users\Owner\source\my-new-project\reports\shap_feature_importance.csv
- Local explanations: C:\Users\Owner\source\my-new-project\reports\shap_local_explanations.csv
- SHAP bar plot: C:\Users\Owner\source\my-new-project\reports\shap_summary_bar.png
- SHAP beeswarm plot: C:\Users\Owner\source\my-new-project\reports\shap_summary_beeswarm.png