# Crop model card

## Crop classifier

**Purpose:** educational crop-class ranking from seven measurements. This
classifier does not estimate yield, profitability, harvest success, or
complete agronomic suitability. Consult local agronomy advice and soil testing
before making high-impact decisions.

**Inputs:** N, P, K, temperature, humidity, pH, rainfall. No region or soil
moisture input exists in the training schema. No new feature was added without
compatible labeled training observations.

**Data:** `data/Crop_recommendation.csv`, 2,200 rows, 22 classes, 100 each.
The 2026-10-16 inventory found no missing features, exact duplicates,
duplicate feature vectors, or conflicting labels for identical feature
vectors. Numeric units, original publisher, license, geography, period,
sampling method and independence are unresolved. The other CSV in `data/` is
byte-identical, not an additional sample. No real rows were added and the
source CSV is unchanged.

**Training/evaluation:** seed 42; stratified 80/20 train/final holdout. Five
fold stratified CV and bounded grid search on the training partition only
select by macro-F1. The final holdout was evaluated once. After evaluation,
the selected configuration was fit on all available rows for serving.
`ml/model_metrics.json` records candidate CV means, holdout baseline and
selected metrics, accuracy Wilson interval, per-class metrics and confusion
matrix. This random split is not a geographic or temporal holdout.

**Observed results:** the CV-selected tuned Random Forest and fixed Random
Forest baseline tied on final holdout accuracy (0.9955), macro-F1 (0.9955),
macro-precision (0.9955), and macro-recall (0.9955). Accuracy's
95% Wilson interval was 0.9836–0.9988 on 440 rows. Two rows were
misclassified (jute as rice, and rice as jute). Training-only CV macro-F1
means were 0.9932 fixed Random Forest, 0.9937 tuned Random Forest, 0.9926
Extra Trees, 0.9880 Histogram Gradient Boosting, and 0.9731 scaled KNN.
Model comparison found
no measured accuracy or macro-F1 gain.
The pre-change model reported the same accuracy and F1, but its holdout
participated in model selection, so it was a development result rather than
an untouched test score.

**Interpretation:** classifier probabilities are not calibrated and must
not be described as confidence of farm suitability or harvest success. The
recommendation explanation now labels them uncalibrated classifier
probabilities. Dataset-profile match values are separate heuristic
similarities, not agronomic validation.

**Limitations:** unknown provenance and units; possible source or sampling
dependence cannot be audited from the CSV; no independent regional/seasonal
validation; no field trials; no yield or farmer-outcome labels; no
agronomist-verified label review; missing region, moisture, soil type,
irrigation, season, pests, market, and resource constraints. The very high
random-split score does not show that recommendations work in farms or unseen
locations.

Train and reproduce: `python ml/train_model.py`. Focused tests:
`python -m pytest tests/test_ml.py -q`. Audit, comparisons and data roadmap:
`docs/crop_recommendation_improvement.md`.

## Other components

Price forecast is conditional on sufficient official market history. The
service requires at least 20 distinct daily observations in a market/variety
series. No live usable history or measured forecast accuracy is established;
forecasting remains unavailable when the required data is absent.
