import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from services import crop_ml_service
from services.crop_service import predict_crop, train_and_save_model, validate_prediction_features


def test_training_creates_model_and_metrics_without_touching_project_artifacts(tmp_path, monkeypatch):
    rows = []
    for crop, offset in (("rice", 0.0), ("maize", 20.0)):
        for index in range(20):
            rows.append({
                "N": 30 + offset + index * 0.23,
                "P": 20 + offset + index * 0.31,
                "K": 15 + offset + index * 0.17,
                "temperature": 20 + offset / 10 + index * 0.08,
                "humidity": 60 + index * 0.13,
                "ph": 6 + index * 0.01,
                "rainfall": 100 + offset + index * 0.7,
                "label": crop,
            })
    dataset = tmp_path / "Crop_recommendation.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    monkeypatch.setattr(crop_ml_service, "_candidate_models", lambda: {
        "Random Forest baseline": (
            RandomForestClassifier(n_estimators=5, max_depth=16, random_state=42, n_jobs=1), {}
        ),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {}),
    })

    model_path, metrics_path = tmp_path / "model.pkl", tmp_path / "metrics.json"
    metrics = train_and_save_model(dataset, model_path=model_path, metrics_path=metrics_path)
    assert model_path.exists() and metrics_path.exists()
    assert metrics["dataset_size"] == 40
    assert metrics["test_samples"] == 8
    assert metrics["test_set_used_for_selection"] is False
    assert set(metrics["selected_holdout_metrics"]["per_class_metrics"]) == {"rice", "maize"}
    assert len(metrics["selected_holdout_metrics"]["confusion_matrix"]) == 2
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["best_model"] == metrics["best_model"]

    result = predict_crop(
        {"N": 31, "P": 22, "K": 17, "temperature": 20, "humidity": 62, "ph": 6.1, "rainfall": 105},
        model_path=model_path,
    )
    assert result["recommended_crop"] in {"rice", "maize"}
    assert len(result["top_3"]) <= 3
    assert result["confidence_type"] == "uncalibrated classifier probability"


@pytest.mark.parametrize("field,value", [
    ("nitrogen", float("nan")),
    ("ph", 14.1),
    ("humidity", -0.1),
    ("temperature", 61),
    ("rainfall", -1),
])
def test_prediction_rejects_invalid_measurements(field, value):
    inputs = {
        "nitrogen": 50, "phosphorus": 40, "potassium": 30,
        "temperature": 24, "humidity": 65, "ph": 6.5, "rainfall": 110,
    }
    inputs[field] = value
    with pytest.raises(ValueError):
        validate_prediction_features(inputs)


def test_dataset_audit_reports_conflicting_duplicate_features(tmp_path):
    rows = []
    for label, offset in (("rice", 0), ("maize", 20)):
        for i in range(6):
            rows.append([30 + offset + i, 20 + offset + i, 15 + offset + i, 20 + i, 60 + i, 6 + i / 10, 100 + offset + i, label])
    # Same features with two labels are retained and explicitly reported.
    rows[6] = [30, 20, 15, 20, 60, 6, 100, "maize"]
    path = tmp_path / "conflict.csv"
    pd.DataFrame(rows, columns=crop_ml_service.FEATURE_COLUMNS + ["label"]).to_csv(path, index=False)
    _, quality = crop_ml_service.load_and_validate_dataset(path)
    assert quality["conflicting_feature_label_groups"] == 1
    assert quality["duplicate_feature_rows"] == 2
