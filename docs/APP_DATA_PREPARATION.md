# FloodSense Application Data Preparation

## Purpose

Precomputed datasets for the FloodSense research application.

The application uses offline model predictions and SHAP
explanations. No model training is required at runtime.

## Reproduction

From the repository root, using Python 3.11:

```powershell
python -m pip install -r scripts/requirements-app-prep.txt
python -m scripts.prepare_app_data
```

The original research requirements remain unchanged.
The application preparation environment uses SHAP 0.51.0
for Python 3.11 compatibility.

## Model methodology

- Model: LightGBM without rainfall.
- Training observations: 66,514.
- Frozen classification cutoff: 0.50.
- Predictions generated for 62,138 unique locations.
- Training uses all development observations.
- Deduplication applies only to geographic inference.
- Independent test observations are not used for fitting.

## Application datasets

| File | Contents |
|---|---|
| location_predictions.parquet | 62,138 locations |
| grid_summary.parquet | 4,492 geographic cells |
| grid_boundaries.geojson | 4,492 grid polygons |
| historical_predictions.parquet | 85,678 observations |
| historical_grid_summary.parquet | 15,873 grid-storm summaries |
| storm_metrics.csv | Four storm metric summaries |
| global_shap_importance.csv | Eight encoded SHAP features |
| model_metadata.json | Configuration and methodology |

## Dataset schemas

### location_predictions.parquet

- `lat`
- `lon`
- `grid_id`
- `elevation_m`
- `slope_degrees`
- `distance_to_bayou_m`
- `paved_pct`
- `land_cover_type`
- `susceptibility_score`
- `predicted_at_frozen_cutoff`
- `shap_elevation_m`
- `shap_slope_degrees`
- `shap_distance_to_bayou_m`
- `shap_paved_pct`
- `shap_land_cover_type`
- `shap_base_value`
- `raw_model_score`

### grid_summary.parquet

- `grid_id`
- `sample_count`
- `mean_score`
- `min_score`
- `max_score`
- `median_score`
- `std_score`
- `locations_above_cutoff`
- `share_above_cutoff`
- `mean_elevation_m`
- `mean_slope_degrees`
- `mean_distance_to_bayou_m`
- `mean_paved_pct`
- `dominant_land_cover`
- `mean_raw_model_score`
- `mean_shap_base_value`
- `shap_elevation_m`
- `shap_slope_degrees`
- `shap_distance_to_bayou_m`
- `shap_paved_pct`
- `shap_land_cover_type`
- `development_observation_count`
- `development_report_positive_count`
- `coverage_category`
- `dominant_positive_factor`
- `dominant_negative_factor`
- `center_lat`
- `center_lon`
- `south`
- `north`
- `west`
- `east`

### historical_predictions.parquet

- `year`
- `storm_name`
- `lat`
- `lon`
- `grid_id`
- `elevation_m`
- `slope_degrees`
- `distance_to_bayou_m`
- `paved_pct`
- `land_cover_type`
- `row_index`
- `y_true`
- `y_prob`
- `fold`
- `evaluation_protocol`
- `historical_id`
- `y_pred`
- `classification`
- `classification_correct`
- `frozen_cutoff`

### historical_grid_summary.parquet

- `year`
- `storm_name`
- `grid_id`
- `evaluation_protocol`
- `sample_count`
- `mean_score`
- `min_score`
- `max_score`
- `report_positive_count`
- `report_positive_share`
- `predicted_positive_count`
- `predicted_positive_share`
- `correct_count`
- `accuracy`
- `tp_count`
- `tn_count`
- `fp_count`
- `fn_count`
- `tp_share`
- `tn_share`
- `fp_share`
- `fn_share`
- `coverage_category`
- `center_lat`
- `center_lon`
- `south`
- `north`
- `west`
- `east`

### storm_metrics.csv

- `year`
- `storm_name`
- `evaluation_protocol`
- `observation_count`
- `grid_count`
- `frozen_cutoff`
- `report_positive_count`
- `predicted_positive_count`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `pr_auc`
- `tp`
- `tn`
- `fp`
- `fn`

### global_shap_importance.csv

- `feature`
- `mean_abs_shap`
- `shap_rank`
- `lightgbm_split_importance`
- `lightgbm_split_rank`

## Important field interpretations

- `susceptibility_score`: positive-class model output.
- `mean_score`: average score across sampled locations.
- `sample_count`: number of sampled locations or
  observations, depending on the dataset.
- `share_above_cutoff`: fraction at or above 0.50.
- `development_report_positive_count`: cleaned
  report-associated observations, not raw report volume.
- `shap_*`: signed contributions in raw log-odds.
- `dominant_positive_factor`: strongest positive mean SHAP.
- `report_positive_share`: fraction of sampled observations
  associated with a recorded report.
- `y_true`: original historical dataset label.
- `y_prob`: saved historical prediction probability.
- `y_pred`: classification using the frozen cutoff.
- `classification`: TP, TN, FP, or FN relative to labels.

## Geographic methodology

The original 0.01-degree research grid is reconstructed
from grid identifiers. Grid dimensions are approximately
1.11 km north-south and 0.96 km east-west in Houston.

Unsampled cells must be displayed as no data.
The grid is not clipped to Houston municipal boundaries.

## SHAP methodology

Local explanations use the full-development fitted model.
One-hot land cover contributions are summed per location.
Grid explanations average signed location contributions.
These explain mean raw log-odds, not mean probability.

Original global SHAP plots remain in reports/figures.
The original encoded-feature importance and newly grouped
importance must be identified as separate analyses.

## Historical evaluation

Memorial Day, Tax Day and Harvey use saved
leave-one-storm-out development predictions.
Imelda uses saved independent test predictions.

No historical models were retrained for the application.
No independent test evaluation was rerun.

## Research limitations

- Research estimates, not official flood warnings.
- Not a real-time or future storm forecast.
- Rainfall is excluded from the selected model.
- No property-level flood risk assessment.
- Only sampled geographic cells have predictions.
- Sampling coverage is not prediction confidence.
- Negative labels do not establish absence of flooding.
- SHAP contributions are not causal explanations.
- Mean SHAP explains mean raw log-odds, not mean probability.
- General susceptibility predictions are in-sample outputs.
- Historical maps do not represent inundation extent.
- Historical storm samples and coverage differ.
- Development validation informed model selection.
- Study area is not clipped to municipal boundaries.

## Research protection

The preparation script verifies 28 protected source files
against docs/app_source_hashes.csv before and after processing.
The independent test lock remains unchanged.

## Application development

Read model_metadata.json for the complete configuration,
data schema, feature descriptions and limitations.

Application runtime should load precomputed data rather
than importing LightGBM, SHAP or PyTorch.
