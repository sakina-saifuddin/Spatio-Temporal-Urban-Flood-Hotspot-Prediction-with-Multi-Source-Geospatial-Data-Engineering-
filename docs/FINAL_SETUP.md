# Frozen model setup

This setup was calculated from the saved training out-of-fold (OOF)
predictions. For each model, `src.threshold.pick_threshold` selected the
highest threshold from 0.05 to 0.95 (in 0.01 increments) that reached at
least 0.85 recall on the area-group OOF predictions. The same threshold was
then applied to the saved leave-one-storm-out OOF predictions. No models were
retrained.

## Models, features, and cutoffs

| Model | Features used | Cutoff | Area recall | Area precision | Storm recall | Storm precision | Role |
|---|---|---:|---:|---:|---:|---:|---|
| logistic_regression_no_rain | elevation_m, slope_degrees, distance_to_bayou_m, paved_pct, land_cover_type; no rainfall_inches | 0.44 | 0.855132 | 0.677193 | 0.855927 | 0.677333 | final |
| lightgbm_no_rain | elevation_m, slope_degrees, distance_to_bayou_m, paved_pct, land_cover_type; no rainfall_inches | 0.50 | 0.852776 | 0.801173 | 0.884141 | 0.805155 | final |
| mlp_no_rain | elevation_m, slope_degrees, distance_to_bayou_m, paved_pct, land_cover_type; no rainfall_inches | 0.47 | 0.852069 | 0.740820 | 0.847268 | 0.744963 | final |
| cnn_hybrid | elevation_m, slope_degrees, distance_to_bayou_m, paved_pct, land_cover_type, 32x32 relative-elevation terrain tile; no rainfall_inches | 0.47 | 0.852511 | 0.777330 | 0.868267 | 0.766443 | final |
| cnn_tile_only | 32x32 relative-elevation terrain tile; no rainfall_inches | 0.38 | 0.856133 | 0.651808 | 0.906906 | 0.618552 | experiment only |

The table-feature preprocessing scales numeric features and one-hot encodes
`land_cover_type`; it is fitted on each training fold only. CNN hybrid uses
the same table features and the saved terrain tiles. CNN tile-only uses
terrain tiles alone. Terrain values were made from relative elevation in
meters divided by the fixed constant 10.

## Fixed model settings

All models use seed 42. The settings below are the ones used for the saved
cross-validation runs; they were not tuned for this freeze.

| Model | Fixed settings |
|---|---|
| Logistic regression | Maximum 1,000 solver iterations |
| LightGBM | 300 trees; learning rate 0.1 |
| MLP | Hidden layers 64 and 32; dropout 0.2; Adam learning rate 0.001; 25 epochs; batch size 512; CPU |
| CNN hybrid | 16/32/64 convolution channels; Adam learning rate 0.001; 20 epochs; batch size 256; dropout 0.3 |
| CNN tile-only (experiment only) | Same CNN convolution settings; Adam learning rate 0.001; 20 epochs; batch size 256; dropout 0.3 |

## Training-data cleaning

The saved training data was cleaned by removing duplicate
latitude/longitude/storm rows, keeping flooded rows and developed-land
non-flooded rows, removing flooded rows outside developed land, assigning a
grid ID from coordinates rounded to 0.01 degrees, and dropping the excluded
`imperviousness` field. Rainfall was retained in the cleaned data but omitted
from the models listed here.

Frozen before the Imelda test. No changes after seeing Imelda scores.
