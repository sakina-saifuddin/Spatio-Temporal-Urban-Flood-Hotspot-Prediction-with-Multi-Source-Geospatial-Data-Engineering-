# Processed datasets

This directory contains the project training and test tables used by the pipeline. Keep raw source files in `data/raw/`; generate processed tables through reproducible scripts and document feature definitions, row counts, label construction, and versioning.

- `flood_training_set.csv` — development data for 2015–2017 (reported 85,738 rows; verify on load).
- `flood_test_set.csv` — held-out Tropical Storm Imelda 2019 data (reported 24,852 rows; do not use for tuning).

Treat the Imelda set as final holdout. Do not fit preprocessing, tune thresholds, or select features on it.
