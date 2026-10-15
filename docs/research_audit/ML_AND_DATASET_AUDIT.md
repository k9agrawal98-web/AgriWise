# Machine-learning and dataset audit

**Evidence basis:** current source, committed data/artifact metadata, existing
project documentation, and focused read-only inventory. Model training was
not run as part of this audit.

## 1. Crop classification

### Dataset inventory

| Property | Verified value |
|---|---|
| Primary path | [`data/Crop_recommendation.csv`](../../data/Crop_recommendation.csv) |
| Rows / classes | 2,200 rows, 22 classes, 100 rows per class |
| Columns | `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label` |
| Crop labels | apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon |
| Duplicate CSV | [`data/recommendationData.csv`](../../data/recommendationData.csv) has the same SHA-256 as the primary file in the audited checkout; it is a duplicate copy, not an independent sample. |
| Provenance | Original provider, license, collection protocol, geography and time period are not established by repository evidence. |

Source references: [`load_and_validate_dataset`](../../services/crop_ml_service.py#L45),
[`train_and_save_crop_models`](../../services/crop_ml_service.py#L105), and
[`ml/train_model.py`](../../ml/train_model.py).
Dataset quality documentation is in
[`docs/data_provenance.md`](../data_provenance.md), which correctly marks
provenance unresolved.

The dataset's balance and schema are inspectable; neither proves the rows are
independent field observations. Do not identify its publisher based on
similarity to a widely circulated crop-recommendation benchmark.

### Training and evaluation path (updated 2026-10-16)

`python ml/train_model.py` validates finite numeric features, nonnegative
N/P/K/rainfall, pH 0–14, humidity 0–100, and normalized labels. It reports
class counts, ranges, exact duplicates, duplicate feature vectors, and
conflicting labels for identical vectors. Training reserves a stratified
80/20 split (seed 42), compares fixed Random Forest, tuned Random Forest,
Extra Trees, Histogram Gradient Boosting, and scaled KNN using stratified
five-fold macro-F1 CV on the training partition, selects only by CV, and
evaluates the final holdout once. It then refits the selected configuration
on all rows for serving and saves per-class metrics, accuracy Wilson
interval, confusion matrix and model metadata.

The 2,200 rows have no missing features, exact duplicates, duplicate feature
vectors, conflicting labels for identical vectors, or invalid rows under
the implemented checks. Units, provenance and sample independence remain
unresolved. Same-vector checks cannot establish independence or rule out
near-similar observations from a shared source batch. The random holdout is
not geographic or temporal; no field evaluation exists.

### Active serving path and parallel implementation

The authenticated `/crop-recommendation` route calls
`services.crop_service.predict_crop` and then
`services.explanation_service.build_explanation`; the saved artifact is
`ml/crop_model.pkl`. The active CLI trainer is instead
`services.crop_ml_service.train_and_save_crop_models`.

`services.crop_service.train_and_save_model` now delegates to this same
audited trainer. The main `/crop-recommendation` handler uses
`services.crop_service.predict_crop`. `predict_crop_ml` remains a parallel
prediction helper and is not called by that handler.

The prediction wrapper reports class probability-derived scores as
uncalibrated classifier probabilities. They are not calibrated field
confidence or probability of successful harvest, yield, profit or agronomic
suitability.

### Recorded model metrics (reproduced 2026-10-16)

The 440-row final holdout (20/class) gave both the CV-selected tuned Random
Forest and fixed Random Forest baseline accuracy, macro-precision,
macro-recall and macro-F1 of 0.9955. Accuracy's 95% Wilson interval is
0.9836–0.9988. Training-only CV macro-F1 means were 0.9932 (fixed RF),
0.9937 (tuned RF), 0.9926 (Extra Trees), 0.9880
(Histogram Gradient Boosting), and 0.9731 (scaled KNN). The final selected
model was refit on all 2,200 valid rows after this holdout evaluation. The
two holdout mistakes were one jute row classified as rice and one rice row
classified as jute. No measured gain over the fixed baseline was
demonstrated. See
`docs/crop_recommendation_improvement.md` and `ml/model_metrics.json`.

The old 0.9955 artifact score used the reported split for model selection;
it was a development score. The new score comes from a holdout not used for
selection. Neither demonstrates farm or regional performance.

### Validity limits

- Dataset origin, license, intended target population, units and label meaning unknown.
- Random split permits geographic/temporal/source-batch overlap.
- No independent external validation, calibration, field trial, yield or
  farmer-outcome measure.
- Seven features omit region, soil moisture, soil type, irrigation, season,
  pests, economics and other local constraints.
- No inference on nationwide performance is justified.

## 2. Conditional market-price regression

[`forecast_commodity_price`](../../services/market_forecast_service.py#L13)
uses the official market adapter's parsed records, not the generated
`data/market_data.json` fixture. It:

- requires at least 20 distinct daily observations in one market/variety
  series;
- regresses elapsed day against modal price;
- holds out the last 20% chronologically for MAE/RMSE, then refits on all
  available observations;
- bounds the point estimate to ±25% of the latest modal price; and
- derives a heuristic range from RMSE and a percentage floor, explicitly not
  a confidence interval.

No saved training data/model artifact or measured forecast error from a live
usable provider history is established. The live data provider has not been
verified to return a usable record set in this environment. Thus this
component is conditional software behavior, not verified operational
forecasting.

## 4. Rules, calculations and configured reference data

| Component | Evidence and classification |
|---|---|
| Expert advisories | [`expert_system.py`](../../services/expert_system.py): explicit conditions/rules; not learned from labeled examples. |
| Daily plan | [`decision_orchestrator.py`](../../services/decision_orchestrator.py): combines active rules, weather, activity state, risk and market context. Deterministic decision composition. |
| Risk/condition scores | [`risk_service.py`](../../services/risk_service.py): weighted threshold arithmetic and evidence/reason text; scores are locally designed indices, not statistically validated probabilities. |
| Crop suitability | [`suitability_service.py`](../../services/suitability_service.py): configured crop ranges/weights and arithmetic. |
| Crop lifecycle/activity | [`lifecycle_service.py`](../../services/lifecycle_service.py), [`activity_service.py`](../../services/activity_service.py): configured stage durations, task templates, dates and status transitions. |
| Fertilizer advice | [`fertilizer_service.py`](../../services/fertilizer_service.py): thresholds and in-code product/nutrient reference entries; not trained and not a substitute for a verified soil test or local extension advice. |
| Crop profiles | `data/crop_knowledge.json`: local configured reference data, documented as lacking precise, traceable references; ranges are unvalidated educational heuristics. |

No evidence was found of an LLM/chatbot, crop-yield model, or automatic
retraining from farmers' activities, reports, expenses or sales.

## 5. Complete data-source inventory

| Source | What it contains | Use/status |
|---|---|---|
| `data/Crop_recommendation.csv` | 2,200 labeled rows / 22 crop classes / seven numeric features. | Active crop-model training input. Provenance unresolved. |
| `data/recommendationData.csv` | Byte-identical copy per SHA-256 comparison in this checkout. | Not independent data; no active Python loader reference established. |
| `data/crop_knowledge.json` | Configured crop attributes/ranges. | Rules/reference data, not model examples; sources are not fully traceable. |
| `data/market_data.json` | Locally generated market sample. | Excluded by active market service; not live prices or forecast evidence. |
| `data/resource_catalog.json` | Locally generated resources/vendors. | Active resource loader returns an empty list rather than presenting generated entries as verified suppliers. |
| SQLite farmer records | Farms, soil inputs, cycles, activities, expenses/sales, reports and more. | User-entered/operational state, not classifier training data. |
| Open-Meteo | Modeled current conditions and forecast. | Weather context, not on-farm sensor measurements or supervised labels. |
| `data.gov.in` / Agmarknet | External government price records. | Configured adapter; requests 10 records at a time; requires API key; a prior check timed out at HTTP response stage after DNS/TCP/TLS. Usable records remain unverified. |
| `ml/crop_model.pkl`, `ml/model_metrics.json`, `ml/confusion_matrix.png` | Derived crop classifier/evaluation artifacts. | Inference and evaluation outputs, not raw data. |

No automated pipeline was found that folds farmer records into the
classification training datasets.

## 6. Safe next research steps

1. Obtain a traceable source citation, license and collection metadata for
   crop training data; use a geographically/time-separated external dataset
   for final evaluation.
2. Reserve an untouched test set before model selection; report per-class
   metrics, confidence intervals where justified, calibration and errors.
3. Evaluate price forecasts only after verified, sufficiently long and
   timestamped official-market series exist; compare with simple baselines
   and report a true temporal protocol.
4. Validate rules and configured agronomic values with qualified local
   experts; distinguish expert-reviewed content from developer-authored
   heuristics.
