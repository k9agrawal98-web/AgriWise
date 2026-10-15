# Crop recommendation reliability audit

**Run date:** 2026-10-16  
**Dataset:** `data/Crop_recommendation.csv` (unchanged)  
**Evaluation seed:** 42

## Audit findings

The active model has seven numeric inputs: N, P, K, temperature, humidity,
pH, and rainfall. The web/API path calls `services.crop_service.predict_crop`;
`ml/train_model.py` writes the model artifact consumed there. The former
project had two different training implementations. The CLI implementation
compared model scores on its held-out test set and selected the winner using
that same set, so its reported test result was not an untouched final test.
The older training function in `crop_service.py` used a separate 75/25
protocol and model set. The model returned forest class probabilities as
"confidence" without calibration.

The primary CSV contains 2,200 valid rows: 100 per each of 22 crop labels.
It has seven features, no missing values, no exact duplicate rows, no
duplicate feature vectors, and no identical feature vectors with conflicting
labels. The loaded data has no invalid rows under the explicit checks
(finite inputs, nonnegative N/P/K/rainfall, pH 0–14, humidity 0–100).
Observed min–max values are recorded in `ml/model_metrics.json`. A nearest
neighbor scan after standardizing each feature found no different-label
nearest pair at Euclidean distance below 0.05. That threshold is a diagnostic,
not a proof of class separation. Numeric units and the dataset's publisher,
license, geography, period, sampling process, and row independence remain
unverified. No class-count balance or random split can resolve those unknowns.

The checked-in pre-change metrics gave 99.55% accuracy and 0.9955 macro-F1
for the selected Random Forest on a 440-row stratified split. Since the old
code selected a model using that split, these values are development results;
they do not constitute an untouched final evaluation or field evidence.

## Strategy comparison

| Action | Expected benefit | Evidence required | Effort | Main risk | Validation |
|---|---|---|---|---|---|
| Fix the evaluation protocol (selected) | Trustworthy within-dataset model comparison; stops test-set winner selection | Stable labels and enough examples per class | Medium | Random split can still share unknown collection batches | Fixed stratified holdout; all selection/tuning in training-only CV |
| Compare and tune classifiers (selected, bounded) | Determine whether a different model improves this dataset benchmark | Reproducible feature/label table | Medium | CV winner may not generalize beyond this source | Identical training split, 5-fold macro-F1, then one holdout evaluation |
| Resolve data quality/provenance | Detect unit/label errors and assess applicability | Original source, units, collection and label definitions | High (external evidence) | Incorrect assumptions could discard valid examples | Provenance review and row-level traceability; no unsupported correction |
| Add real regional observations | Broader local applicability and potential geographic validation | Licensed, traceable field examples with compatible measurement methods and labels | High | Dataset shift and incompatible target definitions | Geographic/temporal external holdout; source-stratified reporting |
| Add region, soil type, moisture, season, irrigation | Model currently cannot condition directly on these factors | Same variables recorded for each labeled training sample and known at prediction time | Very high | Arbitrary feature values, missingness, or future-data leakage | Complete pipeline tests plus grouped external validation |
| Probability calibration / abstention | More interpretable uncertainty on in-distribution examples | Independent validation data with enough samples per class | Medium-high | Calibration on this unverified benchmark can falsely reassure | Reliability diagrams, Brier/log loss, calibration split, external check |
| Farmer outcome and agronomic review | More practically useful recommendations | Local extension expertise, soil tests, yield/outcome and economic context | High | Treating classification labels as farm suitability | Prospective field evaluation and expert review |

## Implemented course

The primary improvement is evaluation integrity, with bounded algorithm
comparison as supporting work. Training reserves a stratified 20% holdout
using seed 42. On the remaining 80%, a stratified five-fold cross-validation
compares a fixed Random Forest baseline, tuned Random Forest, Extra Trees,
Histogram Gradient Boosting, and scaled KNN. Hyperparameter grids and scoring
are defined in `services/crop_ml_service.py`; macro-F1 selects the model
using training data only. The final holdout is evaluated once after selection.
Then the selected configuration is refit on all 2,200 available rows for
serving. A Wilson 95% interval is reported for accuracy. Per-crop precision,
recall, F1, supports, and a confusion matrix are recorded.

The saved artifacts report that the fixed Random Forest baseline and the CV
selected tuned Random Forest both scored **99.55% accuracy and 0.9955
macro-F1** on the same 440-row final holdout (95% Wilson accuracy interval:
98.36%–99.88%). Their identical results show no measured improvement from
the model comparison. Training-only CV macro-F1 means were 0.9932 for fixed
Random Forest, 0.9937 for tuned Random Forest, 0.9926 for Extra Trees, 0.9880
for Histogram Gradient Boosting, and 0.9731 for scaled KNN. Both models'
macro-precision and macro-recall were 0.9955. Two holdout mistakes occurred:
one jute observation was predicted as rice, and one rice observation was
predicted as jute. Per-class metrics are in
`ml/model_metrics.json`.

Thus the defensible result is **more reliable evaluation and safer inference,
not higher measured predictive accuracy**. The old metric's 0.9955 value
matches the new holdout result, but the old workflow selected on that same
holdout; agreement does not retroactively make it independent. The dataset
appears highly separable under a random split, but this does not demonstrate
farm, geographic, future-season, yield, profitability, or harvest-success
performance. There is no verified independent dataset to test against.

Prediction input parsing now rejects missing, non-numeric, infinite values,
negative nutrients/rainfall, pH outside 0–14, humidity outside 0–100, and
temperatures outside the broad physical sanity range −50 to 60 °C.
The seven-feature schema is unchanged. Output probability scores and
explanations explicitly identify them as uncalibrated classifier probabilities;
they are not odds of a successful harvest. The original dataset and its
duplicate copy were not edited, and no synthetic observations were created.

## Reproduce

The recorded run used Python 3.13.7, scikit-learn 1.9.1, pandas 3.0.6, and
numpy 2.4.4. `requirements.txt` pins different library versions, so retrain
the model in the exact deployment environment before deployment; serialized
scikit-learn models are version-sensitive. Install the project's requirements
and then run:

```powershell
python ml/train_model.py
python -m pytest tests/test_ml.py -q
```

Training writes `ml/crop_model.pkl`, `ml/model_metrics.json`, and
`ml/confusion_matrix.png`. The metrics JSON records the fixed seed, split,
candidate CV scores, holdout baseline and selected scores, 95% accuracy
interval, per-crop metrics, confusion matrix, dataset quality counts, and
limitations. Tests use temporary model/data outputs. For a separate test
directory on Windows where the system temp path is unavailable, use:

```powershell
python -m pytest tests/test_ml.py -q --basetemp=.pytest-tmp
```

## Changed files

| File | Change |
|---|---|
| `services/crop_ml_service.py` | Unified quality checks, grouped training-only CV/grid search, fixed holdout evaluation, comparison artifacts, and inference metadata |
| `services/crop_service.py` | Removed divergent trainer; added compatibility delegation, input validation, and optional model path for safe testing |
| `app.py` | Added broad temperature sanity validation at the request boundary |
| `services/explanation_service.py` | Calls probabilities uncalibrated and distinguishes dataset profile similarity from agronomic suitability |
| `locales/en.json`, `locales/hi.json`, `locales/mr.json` | Clarified probability labels and limitations |
| `templates/ml_analytics.html` | Replaced hard-coded/unsupported empirical claims with saved holdout and CV metrics |
| `ml/train_model.py` | Prints training-CV selection and holdout comparisons accurately |
| `ml/crop_model.pkl` | Refit selected Random Forest on all valid rows after final evaluation |
| `ml/model_metrics.json` | Replaced old development metrics with reproducible holdout and data-quality report |
| `ml/confusion_matrix.png` | Now plots the selected model's final holdout confusion matrix |
| `static/img/confusion_matrix.png` | Updated display copy of the final holdout confusion matrix |
| `tests/test_ml.py` | Added preprocessing audit, invalid input, temporary model loading, artifact, and conflicting-label checks |
| `docs/crop_recommendation_improvement.md` | Audit, strategy comparison, results, limitations, roadmap and reproduction commands |
| `docs/data_provenance.md`, `docs/model_cards.md`, `docs/research_audit/ML_AND_DATASET_AUDIT.md` | Updated data/evaluation qualifications and findings |
| `docs/ai_implementation_report.md`, `docs/research_audit/IEEE_PAPER_READINESS.md`, `docs/research_audit/RESEARCH_EVIDENCE_INVENTORY.md`, `docs/research_audit/TESTING_AUDIT.md`, `README.md` | Updated training procedure, metrics, limitations, and test results |

## Prioritized data roadmap and remaining limits

1. Recover the original dataset citation, terms/license, units, measurement
   protocol, location/time coverage, and definition of each crop label. Keep
   source-row identifiers and provenance with every import.
2. Begin a consented, traceable local collection with soil lab results tied to
   N/P/K/pH units, sample depth and date; field moisture sensor method and
   depth; location at an appropriate privacy level; season/sowing date;
   water access; and weather observed or available at recommendation time.
3. Define the target with local agronomists. A label such as "grown here" is
   not automatically a suitable-crop label; record unsuccessful and
   alternative crop outcomes as well as successes.
4. Obtain a compatible licensed regional research/official dataset and
   reconcile units/labels before merging. Keep original sources partitioned
   so one region/source can be held out for external validation.
5. Evaluate top-k usefulness, calibration, subgroup/geographic performance,
   and eventually harvest outcomes prospectively. Do not feed farmer-entered
   outcomes into training without consent, data quality review, and a
   versioned collection pipeline.

Unresolved: the current training data has no verified provenance or
independence; measurement units are unclear; there is no region, moisture,
soil type, season, irrigation, pest, price, or farmer outcome in the model;
crop probability calibration is not established; the random holdout is not
geographic or temporal; no local agronomist reviewed every label; and no
farm trial or real-world accuracy claim is supported.
