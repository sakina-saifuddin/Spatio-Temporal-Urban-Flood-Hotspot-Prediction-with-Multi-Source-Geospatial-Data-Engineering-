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
