# Tree-model comparison record

## What we did

Added Random Forest, XGBoost, and LightGBM with the requested fixed settings and
seed 42. We compared them with the existing dummy and logistic-regression
baselines using five geographic area-group folds and three leave-one-storm-out
folds. Preprocessing was fitted separately inside each training fold. We did not
tune settings or load the held-out 2019 Imelda test data.

The table gives mean cross-validation scores at a 0.5 classification
threshold. ROC-AUC and PR-AUC measure how well each model ranks examples; the
other scores use the threshold to assign flood/non-flood labels.

| Validation split | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Area groups | Dummy | 0.510494 | 0.510494 | 1.000000 | 0.675929 | 0.500000 | 0.510494 |
| Area groups | Logistic regression | 0.723742 | 0.704909 | 0.789326 | 0.744666 | 0.784134 | 0.744711 |
| Area groups | Random Forest | 0.811348 | 0.809886 | 0.823937 | 0.816790 | 0.887299 | 0.866076 |
| Area groups | XGBoost | 0.826037 | 0.812761 | 0.856591 | 0.834053 | 0.899914 | 0.879417 |
| Area groups | LightGBM | 0.827194 | 0.813940 | 0.857554 | 0.835147 | 0.900738 | 0.879999 |
| Leave one storm out | Dummy | 0.510384 | 0.510384 | 1.000000 | 0.675819 | 0.500000 | 0.510384 |
| Leave one storm out | Logistic regression | 0.706823 | 0.682026 | 0.827332 | 0.741026 | 0.771437 | 0.727479 |
| Leave one storm out | Random Forest | 0.686628 | 0.783072 | 0.532818 | 0.608084 | 0.821986 | 0.779616 |
| Leave one storm out | XGBoost | 0.664218 | 0.742301 | 0.476646 | 0.525974 | 0.800202 | 0.760320 |
| Leave one storm out | LightGBM | 0.669270 | 0.777492 | 0.482620 | 0.551157 | 0.828158 | 0.789241 |

## What the results mean

By area-group recall, LightGBM is the best of the tested tree models. Its area
recall is 0.857554 and ROC-AUC is 0.900738. Compared with logistic regression,
that is **0.068228 higher recall** and **0.116604 higher ROC-AUC**. This means
it found more of the positive examples and ranked positive examples more
reliably in the geographic validation folds.

The presentation's approximate benchmark is 0.80 accuracy and 0.83–0.89
recall. On area-group cross-validation, both XGBoost (accuracy 0.826037,
recall 0.856591) and LightGBM (accuracy 0.827194, recall 0.857554) meet both
thresholds. Random Forest meets the accuracy threshold but its recall is
0.823937, below the benchmark range. These are validation comparisons, not
results on the final 2019 test set or proof of real-world flood prediction
performance.

Storm-holdout recall is lower for all three tree models than area-group
recall. This indicates that performance can change when evaluating on a
different storm. The data labels are constructed and balanced, so the scores
should not be interpreted as real-world flood prevalence or calibrated
individual risk.

The notebook also shows Random Forest's five largest built-in feature
importances after fitting it on all cleaned training rows. Those values are a
rough model-specific guide, not a final explanation or evidence of causation.

Validation time for both split types combined on the current environment was
32.6 seconds for Random Forest, 7.3 seconds for XGBoost, and 6.6 seconds for
LightGBM. These timings depend on the machine.

## Threshold selection record

We selected cutoffs using only the saved area-group out-of-fold predictions.
For each model, we chose the highest cutoff from 0.05 to 0.95 (step 0.01)
that kept recall at or above 0.85. A higher cutoff means fewer rows are
classified as flooded, which can reduce false alarms but may also miss floods.
No model was retrained, and the final Imelda test data was not used.

| Model | Area cutoff | Area recall | Area precision | F1 | False alarms | Missed floods |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.44 | 0.854778 | 0.679846 | 0.757342 | 13,668 | 4,931 |
| Random Forest | 0.46 | 0.852363 | 0.796269 | 0.823362 | 7,405 | 5,013 |
| XGBoost | 0.51 | 0.850832 | 0.815181 | 0.832625 | 6,550 | 5,065 |
| LightGBM | 0.51 | 0.852128 | 0.816514 | 0.833941 | 6,502 | 5,021 |

We applied each area-selected cutoff unchanged to the leave-one-storm-out OOF
predictions as an honest check of whether it carries over to another storm.
Overall recall/precision were 0.884082/0.652786 for logistic regression,
0.600825/0.772443 for Random Forest, 0.459638/0.783996 for XGBoost, and
0.464998/0.791270 for LightGBM. Only logistic regression remained at or above
the 0.85 recall target overall; the other models were below it by 0.249175,
0.390362, and 0.385002, respectively.

The per-storm recall also varied. Logistic regression recall was 0.978873 on
Hurricane Harvey, 0.862430 on Memorial Day Flood, and 0.804657 on Tax Day
Flood. For LightGBM, it was 0.117384, 0.707674, and 0.594840 for those storms,
respectively. This shows that its area-selected cutoff did not transfer
consistently across storm holdouts.

For LightGBM, average OOF scores for flooded rows ranged from 0.255739 on
Hurricane Harvey to 0.621376 on Memorial Day Flood. For non-flooded rows, they
ranged from 0.112911 on Hurricane Harvey to 0.232267 on Memorial Day Flood.
Mean rainfall in the cleaned training data was 12.828799 inches for Hurricane
Harvey, 6.549880 for Memorial Day Flood, and 5.718180 for Tax Day Flood. The
average scores differed among storms; these numbers describe the pattern and
do not establish its cause.

These are OOF validation checks, not results from the final held-out Imelda
test. Keep that test untouched until the model and cutoff are fixed.

## Terrain CNN preparation: raster check (Step 1)

The only elevation GeoTIFF found under `data/` and `team/` is
`data/raw/noaa/TX_North2_GCS_3m_NAVDm.tif`. Its metadata reports CRS
EPSG:4269, pixel size 0.0000269494585 by 0.0000269494585 degrees, bounds
(-95.961815417, 28.768234644, -94.352339855, 30.49445526), and nodata -9999.
The file size is 4,228,223,797 bytes. Because this is a geographic CRS, the
pixel size is in degrees, not a fixed number of meters.

The elevation extraction script
`team/marvin/elevation/getElevation.py` reads files from
`data/marvin/houston_dem_reprojected/`; it expects those reprojected 1-meter
tiles to exist. Neither that directory nor the source tile directory it is
supposed to be built from is present in the project. Therefore, the available
NOAA GeoTIFF cannot be confirmed as the raster that produced
`training_clean.csv`'s `elevation_m` values. Point coverage and pixel-value
comparison were not run, and terrain tiles/CNN work must wait until the source
raster used for the elevations is confirmed.

## MLP comparison record

An MLP is a small neural network that learns patterns by passing the feature
values through layers of connected units.

PyTorch CPU 2.15.0 was added to `.venv` and `requirements.txt`. The MLP uses
64 then 32 hidden units, ReLU, dropout 0.2, BCE-with-logits loss, Adam at
0.001, batch size 512, and exactly 25 epochs. Python, NumPy, and PyTorch use
seed 42. The model stores its trained network and per-epoch training loss.
Cross-validation and threshold selection used the existing training folds
only; the final Imelda test data was not used.

The table gives mean OOF cross-validation scores at the default 0.5 cutoff.

| Validation split | Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Area groups | MLP | 0.832390 | 0.764168 | 0.796669 | 0.862315 | 0.841043 |
| Area groups | LightGBM | 0.857554 | 0.813940 | 0.835147 | 0.900738 | 0.879999 |
| Leave one storm out | MLP | 0.548254 | 0.698282 | 0.563699 | 0.805438 | 0.767469 |
| Leave one storm out | LightGBM | 0.482620 | 0.777492 | 0.551157 | 0.828158 | 0.789241 |

On area-group folds, MLP recall and ROC-AUC are below LightGBM. On
leave-one-storm-out folds, MLP recall is higher, but its ROC-AUC is lower.
This means the MLP catches more positive rows at the default cutoff in the
storm holdouts, while LightGBM ranks examples better overall there. The
area-selected MLP cutoff for recall at least 0.85 was 0.47; it produced
0.851892 area recall and 0.562156 overall storm-holdout recall. Its per-storm
recall at that cutoff was 0.106397 for Hurricane Harvey, 0.885161 for
Memorial Day Flood, and 0.727616 for Tax Day Flood.

The MLP validation runs took 140.0 seconds for area-group folds and 69.7
seconds for storm folds on this computer. The fixed, untuned neural-network
settings may not suit every storm equally well; these validation results do
not identify the cause and do not replace the final held-out test.

## Storm and rainfall diagnosis

We compared storm rainfall ranges, then reran LightGBM and logistic-regression
leave-one-storm-out validation with all six table features and with
`rainfall_inches` removed. Every fold fitted its own preprocessing on training
rows only. The additional area-group folds were used only to choose the
0.85-recall cutoff for each model and feature set. We also summarized the
existing LightGBM Harvey OOF scores by rainfall bin.

The rainfall quartiles by storm and flood class were:

| Storm | Flooded label | Min | 25% | Median | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|
| Hurricane Harvey | 0 | 4.985343 | 10.532386 | 11.958555 | 14.081983 | 20.735357 |
| Hurricane Harvey | 1 | 8.108279 | 11.975746 | 12.677602 | 13.608123 | 20.818255 |
| Memorial Day Flood | 0 | 0.462137 | 3.681719 | 5.552015 | 7.550795 | 11.875772 |
| Memorial Day Flood | 1 | 1.015565 | 6.574985 | 7.860905 | 8.416563 | 11.298113 |
| Tax Day Flood | 0 | 0.048510 | 3.212133 | 5.435413 | 9.169340 | 13.216431 |
| Tax Day Flood | 1 | 0.110535 | 4.564173 | 4.974949 | 6.406109 | 12.944935 |

Harvey's maximum rainfall was 20.818255 inches; the maximum among the other
two storms was 13.216431 inches. About 34.412690% of Harvey rows were above
that other-storm maximum. The corresponding share was 0% for each of the
other two storms. Harvey therefore includes rainfall values beyond the
observed range of the other storms.

At the default 0.5 cutoff, the per-storm results (recall / precision /
ROC-AUC) were:

| Model | Features | Storm | Recall | Precision | ROC-AUC |
|---|---|---|---:|---:|---:|
| LightGBM | All | Hurricane Harvey | 0.123384 | 0.741870 | 0.784411 |
| LightGBM | All | Memorial Day Flood | 0.717949 | 0.783644 | 0.847604 |
| LightGBM | All | Tax Day Flood | 0.606526 | 0.806961 | 0.852460 |
| LightGBM | Without rainfall | Hurricane Harvey | 0.878983 | 0.809102 | 0.899057 |
| LightGBM | Without rainfall | Memorial Day Flood | 0.895617 | 0.802706 | 0.906751 |
| LightGBM | Without rainfall | Tax Day Flood | 0.878281 | 0.803454 | 0.898621 |
| Logistic regression | All | Hurricane Harvey | 0.967210 | 0.620763 | 0.738911 |
| Logistic regression | All | Memorial Day Flood | 0.797327 | 0.700232 | 0.788432 |
| Logistic regression | All | Tax Day Flood | 0.717458 | 0.725084 | 0.786968 |
| Logistic regression | Without rainfall | Hurricane Harvey | 0.789572 | 0.710062 | 0.781208 |
| Logistic regression | Without rainfall | Memorial Day Flood | 0.802782 | 0.695690 | 0.785667 |
| Logistic regression | Without rainfall | Tax Day Flood | 0.783891 | 0.699615 | 0.777357 |

The area-selected cutoffs for target recall 0.85 were LightGBM 0.51 with all
features, LightGBM 0.50 without rainfall, and logistic regression 0.44 with
either feature set. At those cutoffs, Harvey recall was 0.117384 and 0.878983
for LightGBM (all features and without rainfall), and 0.978873 and 0.855573
for logistic regression (all features and without rainfall), respectively.

In the existing LightGBM Harvey OOF predictions for flooded rows, mean
predicted probabilities were 0.414247 for rainfall 8-11 inches (1,386 rows),
0.244376 for 11-14 inches (8,327 rows), 0.241187 for 14-17 inches
(1,021 rows), and 0.155458 for 17-21 inches (1,099 rows). There were no
flooded Harvey rows in the 5-8 inch bin. The scores differed across bins and
were lower in the higher listed rainfall bins; this describes the saved
predictions and does not establish a cause.

The feature-removal result is mixed: LightGBM Harvey recall increased when
rainfall was removed, but logistic-regression Harvey recall decreased. The
numbers show that Harvey is wetter than the other storms and that removing
rainfall changed the two models differently. They do not prove that trees
cannot extrapolate to wetter storms or explain why the predictions changed.
No Imelda test data was used.

### CNN notebook observations

Added plain-language observations after each result in `notebooks/10_cnn.ipynb`, covering tile alignment, model scores, per-storm performance, cutoffs, training loss, and run time. These notes describe the saved training-only validation results; no new model run or Imelda test was used.

### Frozen model setup (2026-11-08)

Recomputed the 0.85-recall cutoffs from saved area-group OOF predictions for
logistic regression, LightGBM, MLP, CNN hybrid, and CNN tile-only. Applied
each same cutoff to its saved held-out-storm OOF predictions and saved the
metrics in `reports/final_setup.csv`. Documented the features, thresholds,
fixed training settings, and training-data cleaning steps in
`docs/FINAL_SETUP.md`. CNN tile-only is marked experiment-only; the other
four models are marked final. These calculations used saved training results
only; no models were retrained and no Imelda test data was opened.

### Prepare final-test data (2026-11-08)

Applied the existing duplicate removal and developed-land filtering functions
from `src/data.py` to `data/processed/test_final.csv.csv`, then added the same
grid ID and removed the same excluded field used for training. Saved the
cleaned rows to `data/processed/test_final_clean.csv`. The test cleaning log
was added to `reports/cleaning_log.csv` with a dataset label; existing
training-clean log rows were retained.

The file has 19,164 rows after cleaning. The duplicate step removed 2,663
rows, developed-negative filtering removed 3,008, and non-developed-positive
filtering removed 17. The cleaned columns, their order, and pandas data types
match `training_clean.csv`; no missing values were found in the model feature
columns. No cleaned row shared the training duplicate identity key
(`lat`, `lon`, `storm_name`); 3,800 grid IDs were shared.

Using coordinates and raster bounds only, 19,163 cleaned points fall inside
the 76 already-downloaded terrain tiles and one falls outside. That point is
just north of the downloaded `x28y334` tile; one neighboring USGS tile,
`x28y335`, is needed for coverage. It was not downloaded and no terrain tiles
were built. Median rainfall is 1.480 inches in the Imelda file and 8.017
inches in cleaned training. The frozen final models do not use
`rainfall_inches`. No model was trained, tuned, scored, or evaluated on
Imelda data.

### Build final-test terrain tiles (2026-11-08)

Downloaded the single missing USGS 1-meter tile `USGS_1M_15_x28y335_TX_Houston_B24.tif`
and reprojected it to EPSG:32615 with 1-meter pixels, using the existing
download and raster helpers. The source download was 387,750,713 bytes and
the reprojected raster was 300,879,044 bytes. No other elevation tiles were
downloaded.

Used `src/tiles.py` to make 32x32, 320-meter terrain windows for the 19,164
rows of `test_final_clean.csv`, subtracting center elevation and dividing by
10, with missing pixels filled by zero. The output is
`data/processed/tiles/test_tiles.npy` with row indices in the same cleaned
row order. Its shape is (19,164, 32, 32), dtype float16, minimum -3.378906,
maximum 2.687500, mean -0.006153, and 1,181 tiles (6.163%) had missing pixels.
The row index check passed.

For comparison, training tiles have shape (66,514, 32, 32), minimum
-4.367188, maximum 3.343750, mean -0.005795, and 3,828 tiles (5.755%) with
missing pixels. The Imelda missing-pixel share is 0.408 percentage points
higher. Both new tile files are ignored by the existing `tiles/` Git ignore
rule. This work used coordinates and terrain arrays only; flood labels were
not inspected, and no model was trained, tuned, scored, or evaluated.

## Comparing models without rainfall

Added optional `drop_features` arguments to `get_xy`, `build_preprocessor`,
and `cross_validate`. The default keeps all configured features and preserves
the prior behavior. The logistic-regression area-fold check remained recall
0.789326 and ROC-AUC 0.784134. The notebook evaluates five models with and
without `rainfall_inches` on area folds and held-out-storm folds, using the
same seed and fold-local preprocessing. The comparison below is recall /
precision / F1 / ROC-AUC / PR-AUC; values show with rainfall → without
rainfall.

| Model | Validation | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic regression | Area folds | 0.7893 → 0.7902 | 0.7049 → 0.7019 | 0.7447 → 0.7434 | 0.7841 → 0.7809 | 0.7447 → 0.7431 |
| Logistic regression | Held-out storms | 0.8273 → 0.7921 | 0.6820 → 0.7018 | 0.7410 → 0.7442 | 0.7714 → 0.7814 | 0.7275 → 0.7439 |
| Random forest | Area folds | 0.8239 → 0.7883 | 0.8099 → 0.7944 | 0.8168 → 0.7912 | 0.8873 → 0.8658 | 0.8661 → 0.8397 |
| Random forest | Held-out storms | 0.5328 → 0.8596 | 0.7831 → 0.8011 | 0.6081 → 0.8293 | 0.8220 → 0.8963 | 0.7796 → 0.8833 |
| XGBoost | Area folds | 0.8566 → 0.8445 | 0.8128 → 0.7979 | 0.8341 → 0.8205 | 0.8999 → 0.8869 | 0.8794 → 0.8632 |
| XGBoost | Held-out storms | 0.4766 → 0.8785 | 0.7423 → 0.8017 | 0.5260 → 0.8383 | 0.8002 → 0.8982 | 0.7603 → 0.8736 |
| LightGBM | Area folds | 0.8576 → 0.8528 | 0.8139 → 0.8012 | 0.8351 → 0.8261 | 0.9007 → 0.8913 | 0.8800 → 0.8677 |
| LightGBM | Held-out storms | 0.4826 → 0.8843 | 0.7775 → 0.8051 | 0.5512 → 0.8428 | 0.8282 → 0.9015 | 0.7892 → 0.8772 |
| MLP | Area folds | 0.8324 → 0.8291 | 0.7642 → 0.7518 | 0.7967 → 0.7884 | 0.8623 → 0.8530 | 0.8410 → 0.8316 |
| MLP | Held-out storms | 0.5483 → 0.8248 | 0.6983 → 0.7565 | 0.5637 → 0.7891 | 0.8054 → 0.8541 | 0.7675 → 0.8318 |

Without rainfall, Random Forest, XGBoost, LightGBM, and MLP improved on the
held-out-storm split for recall and ROC-AUC; logistic regression's recall
fell by 0.0353 while its ROC-AUC rose by 0.0100. All five models had lower
area-fold ROC-AUC without rainfall, by 0.0032 to 0.0215. The strongest
no-rain held-out-storm results were LightGBM: recall 0.8843 and ROC-AUC
0.9015, narrowly ahead of the other no-rain models on both measures.

At the default 0.5 cutoff, the MLP's Harvey recall changed from 0.0820 with
rainfall to 0.8233 without it; its ROC-AUC changed from 0.6954 to 0.8508.
This resembles the tree-model change in these saved predictions, but it does
not identify why the change happened. Area-fold thresholds for recall 0.85
were 0.44 for logistic regression, 0.41 for Random Forest, 0.49 for XGBoost,
0.50 for LightGBM, and 0.47 for MLP. On storm OOF predictions, Random Forest
recall stayed at or above 0.8968 on each storm; XGBoost at or above 0.8727;
and LightGBM at or above 0.8783. Logistic regression fell to 0.8466 on Tax
Day Flood, and MLP fell to 0.8354 on Tax Day Flood and 0.8448 on Hurricane
Harvey. These are validation comparisons, not evidence about performance on
an unseen future event; no Imelda test data was used.

## Elevation raster pipeline inspection

Inspected the three scripts under `team/marvin/elevation/`. There is no
README inside that folder; `team/marvin/README.md` contains general pipeline
instructions. The scripts query The National Map API for the USGS “Digital
Elevation Model (DEM) 1 meter” GeoTIFF product using a fixed Houston-area
bounding box. The catalog query returned 271 downloadable files with
93,220,217,608 bytes (93.22 GB) of listed source sizes. The results include
multiple projects, not just one collection.

`getFiles.py` saves those files under `data/marvin/houston_dem_tiles/`.
`mergeFiles.py` does not make one merged raster: it reprojects each input to
EPSG:32615 at 1 meter per pixel, as float32 with DEFLATE compression, and
writes `tile_0000.tif`, `tile_0001.tif`, and so on under
`data/marvin/houston_dem_reprojected/`. The source catalog describes each
tile as 10,000 by 10,000 meters. If all 271 outputs have those dimensions,
their uncompressed float32 size would be up to 108.4 GB; actual compressed
size cannot be known before conversion. `getElevation.py` indexes the
reprojected TIFFs, finds the tile or tiles that intersect each point's small
neighborhood, and merges only those windows in memory. It samples elevation
and calculates slope; it writes `data/marvin/out/elevation_slope.csv`.
Because the file glob is not sorted, the exact `tile_NNNN.tif` number for a
given source tile is not predetermined. No reprojected files currently exist.

The elevation script expects `data/marvin/input.csv` with `lat` and `lon`
columns in decimal degrees (EPSG:4326); extra columns are ignored. The sample
CSV is currently at `team/marvin/data/input.csv`, which is not the path the
script reads. The scripts do not move it automatically. The expected
`data/marvin/` directory does not currently exist. The C: drive, where it
would be created, had 274.34 GB free when checked.

The existing raster is
`data/raw/noaa/TX_North2_GCS_3m_NAVDm.tif`, 4,228,223,797 bytes. Its CRS is
EPSG:4269, its pixel size is about 2.6 by 3.0 meters at Houston's latitude,
and its bounds contain the cleaned training points' coordinate bounds. It
is a NOAA 3-meter product, not the pipeline's USGS 1-meter product; the
geographic coverage overlaps, but source and resolution differ. Its nodata
value is -9999, and point-by-point valid-data coverage and agreement with
the training elevation values have not been tested.

A tighter National Map metadata query using the cleaned training coordinate
bounds returned 191 products from several collections. Filtering to the
Houston B24 collection, 76 tile footprints intersect 320-meter windows
around the 66,514 training points, covering all rows by the catalog
footprints. Their listed download sizes total 29,045,023,814 bytes (29.05
GB). This is a metadata-based estimate, not pixel-level validation. The 76
converted tiles would occupy up to about 30.4 GB uncompressed as 10,000 by
10,000 float32 rasters; compressed size is unknown before conversion. A
float16 array of 66,514 32-by-32 tiles would be 136,220,672 bytes (about
129.9 MiB), excluding its row-index file and example figure.

The proposed low-waste approach is to use only those 76 Houston B24 tiles,
reproject them as the existing pipeline does, and extract the terrain
windows directly without making a full-area mosaic. At 29.05 GB, that is
about 31% of the full query's listed download size. If Van's roughly
two-hour full-pipeline estimate scales linearly with download size, the
download and reprojection subset would take roughly 37 minutes; this is a
projection, not a measured runtime. Tile extraction time is not benchmarked
and should be estimated with a small pilot. The current NOAA raster could
avoid downloading, but it is a different product and should not be treated
as equivalent without validating its point elevations.

## Two-tile elevation pilot

Ran a pilot using only the two highest-row-count Houston B24 catalog tiles:
`USGS_1M_15_x26y330_TX_Houston_B24.tif` and
`USGS_1M_15_x27y330_TX_Houston_B24.tif`. Their catalog footprints contained
5,919 and 4,713 training points, respectively. After reprojection, the
actual raster bounds covered 5,712 and 4,590 rows (10,293 unique rows total).
No held-out Imelda file was opened or used.

Downloaded files are under `data/marvin/pilot/source/`; the two reprojected
rasters and pilot terrain files are under `data/marvin/pilot/`. The source
files total 810,479,961 bytes and took 25.26 seconds to download. Reprojection
used the project helper in `src/tiles.py`, EPSG:32615, 1-meter pixels,
float32, bilinear resampling, and DEFLATE compression, matching the
existing conversion settings. The outputs are each 10,013 by 10,013 pixels
and total 627,866,096 bytes; reprojection took 112.83 seconds.

Compared pixel elevations with `elevation_m` for 2,000 seed-42 random rows
inside these rasters. All 10,293 covered rows had a valid center pixel. The
sample correlation was 0.997142 and mean absolute difference was 0.0651
meters, indicating a close match. The median signed difference was 0.0073
meters; the 95th percentile was 0.1564 meters, with a few larger outliers.

Created 10,293 relative terrain windows, each 320 meters across and 32 by
32 float16 pixels, in `data/marvin/pilot/pilot_tiles.npy`. Their original
training row indices are in `pilot_tiles_row_index.npy`. Extraction took
359.38 seconds; 559 tiles had at least one missing or nodata pixel. The
terrain array is 21,080,192 bytes and its index is 82,472 bytes. The
8-example figure (4 flooded, 4 non-flooded) is
`reports/figures/example_tiles.png`.

Measured pilot storage is about 1.46 GB including the two downloaded
rasters, two converted rasters, tile array, index, and example figure.
Scaling the two-file conversion size ratio to the 76-tile catalog total
gives about 22.50 GB of converted rasters. Together with 29.05 GB of source
files and the approximately 0.137 GB full terrain array plus indices, the
estimated full-run disk requirement is about 51.68 GB (48.13 GiB). The C:
drive had 272.96 GiB free after the pilot.

Linear extrapolation from pilot timings estimates about 15.1 minutes to
download all 76 tiles, 71.5 minutes to reproject them, and 38.7 minutes to
extract all 66,514 terrain windows: about 2 hours 5 minutes total if run
sequentially. These estimates are rough; download speed, raster compression,
tile-boundary missing data, and extraction throughput may differ for the
remaining tiles.

The existing `.gitignore` already contains `data/marvin`, and Git confirms
the pilot rasters are ignored. No `.gitignore` change was made.

## Full 76-tile terrain build

Completed the full elevation run using the 76 USGS Houston B24 tiles. The
resumable runner is `src/run_full_tiles.py`; it reuses the two pilot source
and reprojected files and downloaded/reprojected the other 74. Each tile
status and elapsed time was printed and appended to `logs/tiles_run.log`.
Interrupted source downloads can resume from their partial file, completed
rasters and terrain rows are skipped on restart, and terrain extraction
checkpoints every 500 rows.

The run took 8,131.8 seconds (about 2 hours 15 minutes). The 74 new source
downloads took 977.7 seconds total. Their EPSG:32615, 1-meter reprojected
outputs took 5,395.4 seconds total. Across all 76 tiles, source GeoTIFFs
occupy 29,045,023,814 bytes and converted GeoTIFFs occupy 22,347,169,815
bytes. Two source and projected tiles were reused from the pilot, not
downloaded again.

Saved a float16 array with shape `(66514, 32, 32)` to
`data/processed/tiles/train_tiles.npy` in cleaned-training row order. Its
row-index file is `train_tiles_row_index.npy` and contains the sequential
indices 0 through 66,513. All 66,514 rows have a valid center elevation and
therefore a terrain tile. The tile array occupies 136,220,800 bytes; the
index occupies 532,240 bytes. Two additional masks record rows with valid
center tiles and tiles with any missing pixels. There are 3,828 tiles with
at least one missing or nodata pixel (5.76%); those pixels are filled with
zero after centering and scaling the elevation.

On 2,000 seed-42 random rows sampled across the full training set, the
reprojected center elevations had correlation 0.999589 with `elevation_m`
and mean absolute difference 0.081747 meters. Among rows with valid center
tiles, 5.413% of flooded rows and 6.112% of non-flooded rows had some
missing pixels. Flooded rows made up 48.015% of tiles with missing pixels
and 51.235% of complete tiles; missing pixels were not more concentrated in
flooded tiles in this comparison.

The 8-example image (4 flooded and 4 non-flooded) is saved as
`reports/figures/example_tiles_full.png`. Large elevation rasters and
generated tile arrays remain locally ignored by Git. No CNN had been trained
at the time of this tile-build entry. The held-out Imelda test file was not
opened or used.

## CNN training pilot and full validation

Checked the full training tile array before CNN training. It has shape
`(66514, 32, 32)`, type float16, minimum -4.367188, maximum 3.343750,
mean -0.005795, and 1.4851% exactly-zero pixels. The row-index file contains
the sequential indices 0 through 66,513 and matches the cleaned training
row order. CNNs use the saved values directly: each is relative elevation
in meters divided by the fixed constant 10 meters. No scaling is fitted on
validation data.

Added `src/cnn.py` with the requested tile-only and hybrid architectures,
fixed 20-epoch training, Adam, BCE-with-logits loss, seed 42, CPU/GPU
selection, per-epoch training losses, fold-local preprocessing for the
hybrid, OOF predictions, and fitted modules accessible as `model_`. The
new cross-validation helper saves result and OOF files through the existing
`save_results` helper by default. Existing code was not changed.

Timed one first area-group fold per CNN on CPU. Tile-only took 360.5 seconds
and hybrid took 444.1 seconds. Scaling these measurements by the actual
training-row counts across five area folds and three storm folds estimates
about 30 minutes for the eight tile-only folds and 37 minutes for the eight
hybrid folds, or about 67 minutes combined. This estimate exceeds the
requested 40-minute limit. After approval, the full notebook run took
75.52 minutes for both models and both validation designs: 41.27 minutes
for tile-only (31.82 area, 9.45 storm) and 34.24 minutes for hybrid
(24.90 area, 9.34 storm).

The notebook `notebooks/10_cnn.ipynb` ran all seven code cells successfully.
The table gives recall / precision / F1 / ROC-AUC / PR-AUC:

| Model | Validation | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression without rainfall | Area folds | 0.7902 | 0.7019 | 0.7434 | 0.7809 | 0.7431 |
| LightGBM without rainfall | Area folds | 0.8528 | 0.8012 | 0.8261 | 0.8913 | 0.8677 |
| MLP without rainfall | Area folds | 0.8291 | 0.7518 | 0.7884 | 0.8530 | 0.8316 |
| CNN tile-only | Area folds | 0.7261 | 0.7057 | 0.7154 | 0.7738 | 0.7562 |
| CNN hybrid | Area folds | 0.8318 | 0.7869 | 0.8086 | 0.8797 | 0.8641 |
| Logistic regression without rainfall | Held-out storms | 0.7921 | 0.7018 | 0.7442 | 0.7814 | 0.7439 |
| LightGBM without rainfall | Held-out storms | 0.8843 | 0.8051 | 0.8428 | 0.9015 | 0.8772 |
| MLP without rainfall | Held-out storms | 0.8248 | 0.7565 | 0.7891 | 0.8541 | 0.8318 |
| CNN tile-only | Held-out storms | 0.8163 | 0.6670 | 0.7337 | 0.7692 | 0.7514 |
| CNN hybrid | Held-out storms | 0.8492 | 0.7781 | 0.8118 | 0.8791 | 0.8634 |

Compared with LightGBM without rainfall, CNN hybrid was lower by 0.0210
recall and 0.0116 ROC-AUC on area folds, and lower by 0.0351 recall and
0.0224 ROC-AUC on held-out storms. Compared with logistic regression without
rainfall, tile-only CNN recall was lower by 0.0641 on area folds and higher
by 0.0242 on held-out storms; its ROC-AUC was lower by 0.0072 and 0.0122,
respectively.

The highest area-fold cutoffs reaching recall 0.85 were 0.38 for tile-only
CNN (area recall 0.8561) and 0.47 for hybrid CNN (area recall 0.8525).
Applying the same cutoff to storm OOF
predictions gave tile-only recall 0.9152 for Harvey, 0.9243 for Memorial
Day, and 0.8809 for Tax Day. Hybrid recall was 0.8557, 0.8949, and 0.8554
for those storms, respectively. Both cutoffs reached 0.85 recall on each
storm in these OOF results.

At the default 0.5 cutoff, hybrid CNN's per-storm recall / ROC-AUC were
0.8346 / 0.8758 for Harvey, 0.8783 / 0.8826 for Memorial Day, and
0.8346 / 0.8789 for Tax Day. Tile-only results were 0.8339 / 0.7701,
0.8372 / 0.7754, and 0.7778 / 0.7620, respectively.

The training curves are saved in `reports/figures/cnn_training_loss.png`.
The comparison suggests the terrain image did not add a consistent
performance improvement here: the hybrid CNN did not exceed no-rainfall
LightGBM, and terrain-only had lower ROC-AUC than no-rainfall logistic
regression on both validation designs. These validation results do not prove
that terrain shape has no value in other settings. A 320-meter tile near a
1-kilometer grid-cell border can cross into a neighboring cell, making the
area-fold test slightly less strict for CNNs than for table-feature models.
No Imelda test data was used.

### One-time Imelda final test (2026-11-09)

Added `src/final_test.py` and `notebooks/11_final_test.ipynb`. The runner reads
the frozen models, features, and cutoffs from `reports/final_setup.csv`, fits
each listed model once on all cleaned training rows using the already-fixed
settings, predicts once for the cleaned Imelda rows, and creates
`reports/final_test.lock` to prevent a rerun. It also runs CNN tile-only as
experiment-only and the dummy as an experiment/reference. Predictions are in
`reports/final_test_predictions_<model>.csv`, and both frozen-cutoff and
default-0.5 metrics are in `reports/final_test_results.csv`. The default
threshold is only a comparison; no threshold was picked using Imelda.

At frozen cutoffs, recall was 0.812538 for logistic regression, 0.865483 for
LightGBM, 0.840242 for the MLP, and 0.845885 for CNN hybrid. Among final
models, only LightGBM exceeded the 0.85 target; MLP and CNN hybrid were within
0.01 below it, while logistic regression was 0.037462 below. The
experiment-only tile CNN had recall 0.871434 at its frozen cutoff. The dummy
reference had recall 1.000000 at 0.5 because it predicted every row as
flooded; its precision was 0.508558 and ROC-AUC was 0.500000.

LightGBM's final ROC-AUC / PR-AUC were 0.889570 / 0.860290, highest among the
four final models. The completed one-time run took 929.3 seconds. The
notebook compares these final scores with area/storm validation scores and
notes the shared-grid and rainfall limitations. No tuning, feature
selection, cutoff changes, or rerun was performed.

### Training-only LightGBM SHAP explanations (2026-11-09)

Added `src/explain.py` and `notebooks/12_shap.ipynb`. The analysis fits the
frozen `lightgbm_no_rain` model on all 66,514 rows of
`data/processed/training_clean.csv`, using 300 trees, learning rate 0.1, and
seed 42. Rainfall is excluded; the saved cutoff of 0.50 is used only to label
example predictions. TreeExplainer values are calculated for a fixed random
sample of 5,000 training rows. SHAP 0.54.1 is included in `requirements.txt`.
No test or Imelda file was loaded or used.

Mean absolute SHAP ranked the features as follows:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | elevation_m | 1.329801 |
| 2 | paved_pct | 0.723200 |
| 3 | slope_degrees | 0.653920 |
| 4 | land cover type: Developed, Low Intensity | 0.192739 |
| 5 | distance_to_bayou_m | 0.152670 |
| 6 | land cover type: Developed, Open Space | 0.126350 |
| 7 | land cover type: Developed, Medium Intensity | 0.067090 |
| 8 | land cover type: Developed, High Intensity | 0.058367 |

LightGBM's built-in split-count ranking was similar but not identical: its
top three were `paved_pct`, `elevation_m`, and `distance_to_bayou_m`, versus
`elevation_m`, `paved_pct`, and `slope_degrees` for SHAP. Their rank
correlation was 0.833, with 2 of the top 3 features shared.

Saved the mean-absolute-SHAP bar chart, beeswarm, dependence plots for
elevation, distance to bayou, and slope, plus three example waterfall plots
under `reports/figures/`. Example rows and the feature-importance comparison
are in `reports/shap_examples.csv` and
`reports/shap_feature_importance.csv`. The example rows are in-sample
illustrations, not independent performance estimates.

SHAP shows what the model uses, not what causes flooding. `paved_pct` is
really the share of the tract near major highways.

### Completed-project README (2026-11-09)

Saved the previous project overview as `README_old.md` and rewrote `README.md`
to describe the completed data, models, validation, frozen one-time Imelda
evaluation, SHAP results, limitations, notebook sequence, and team. Corrected
the elevation-source description to the USGS 1-meter DEM obtained through The
National Map, and clarified that `paved_pct` is the tract share near major
highways from OSM rather than NLCD imperviousness.

The README tables and quantitative claims were checked against
`reports/cleaning_log.csv`, `reports/results.csv`,
`reports/final_test_results.csv`, `reports/shap_feature_importance.csv`,
`docs/FINAL_SETUP.md`, and saved notebook outputs. The terrain missing-pixel
share and rainfall medians were taken from this record's earlier documented
results. No Imelda input file was opened, and no model was run or retrained.
No number used in the README was left without a source.

### README results charts (2026-11-09)

Expanded `README.md` with embedded GitHub-rendered figures for the saved
Harvey rainfall diagnostic, validation comparisons, CNN training/terrain
examples, final frozen-cutoff results, and LightGBM SHAP explanations.
Generated `reports/figures/validation_model_comparison.png` from the saved
mean OOF rows in `reports/results.csv`, and
`reports/figures/final_test_model_comparison.png` from the frozen final-model
rows in `reports/final_test_results.csv`. These are static visualizations of
saved results; no model was run or retrained and no Imelda input file was
opened. Captions distinguish training diagnostics from final held-out
results and reiterate that SHAP is not causal evidence.
