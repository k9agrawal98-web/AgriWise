from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from services.crop_ml_service import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    FEATURE_INPUT_NAMES,
    MODEL_PATH,
    METRICS_PATH,
    load_and_validate_dataset,
    train_and_save_crop_models,
)


def validate_prediction_features(features: Dict[str, Any]) -> List[float]:
    """Validate the exact seven model inputs before inference."""
    values = []
    for column in FEATURE_COLUMNS:
        alias = FEATURE_INPUT_NAMES[column]
        raw = features.get(alias, features.get(column))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"A valid numeric value is required for {alias}.") from None
        if not np.isfinite(value):
            raise ValueError(f"{alias.capitalize()} must be a finite number.")
        values.append(value)

    vals = dict(zip(FEATURE_COLUMNS, values))
    for col in ("N", "P", "K", "rainfall"):
        if vals[col] < 0:
            raise ValueError(f"{FEATURE_INPUT_NAMES[col].capitalize()} and rainfall must be nonnegative.")
    if not 0 <= vals["ph"] <= 14:
        raise ValueError("pH must be between 0 and 14.")
    if not 0 <= vals["humidity"] <= 100:
        raise ValueError("Humidity must be between 0 and 100.")
    if not -50 <= vals["temperature"] <= 60:
        raise ValueError("Temperature must be between -50 and 60 °C.")
    return values


def train_and_save_model(
    data_path: Path | str = DATASET_PATH,
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
):
    """Compatibility entry point; delegates to the single audited training pipeline."""
    metrics = train_and_save_crop_models(
        data_path=data_path,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        cm_image_path=Path(metrics_path).with_name("confusion_matrix.png"),
    )
    # Retain legacy keys consumed by older scripts while exposing richer metrics.
    metrics["best_f1_score"] = metrics["selected_holdout_metrics"]["macro_f1"]
    metrics["models"] = metrics["models_evaluated"]
    return metrics


def ensure_model_exists():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Crop model unavailable. Run python ml/train_model.py.")
    return MODEL_PATH


def predict_crop(features: Dict[str, Any], model_path: Path | str | None = None) -> Dict[str, Any]:
    model_file = Path(model_path) if model_path is not None else ensure_model_exists()
    if not model_file.exists():
        raise FileNotFoundError(f"Crop model unavailable at {model_file}.")
    model = joblib.load(model_file)
    input_values = validate_prediction_features(features)
    input_frame = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_frame)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)[0]
        class_names = model.classes_
        indices = np.argsort(probabilities)[::-1][:3]
        probability = float(np.max(probabilities)) * 100
        top_3 = [
            {"crop": str(class_names[index]), "confidence": round(float(probabilities[index]) * 100, 2)}
            for index in indices
        ]
    else:
        probability = None
        top_3 = [{"crop": str(prediction), "confidence": None}]

    return {
        "recommended_crop": str(prediction),
        "confidence": round(probability, 2) if probability is not None else None,
        "confidence_type": "uncalibrated classifier probability",
        "top_3": top_3,
    }


def compare_crop_dataset_fit(features: Dict[str, Any], crop_names: List[str] | None = None) -> List[Dict[str, Any]]:
    dataset, _ = load_and_validate_dataset(DATASET_PATH)
    target_name = "label"
    requested = crop_names or sorted(dataset[target_name].astype(str).unique().tolist())
    results = []
    for crop_name in requested:
        profile_rows = dataset[dataset[target_name].astype(str).str.casefold() == crop_name.casefold()]
        if profile_rows.empty:
            continue
        feature_scores = {}
        for column in FEATURE_COLUMNS:
            values = profile_rows[column].astype(float)
            median = float(values.median())
            spread = float(values.quantile(0.75) - values.quantile(0.25))
            if spread <= 0:
                spread = float(dataset[column].std()) or 1.0
            observed = float(features.get(FEATURE_INPUT_NAMES[column], features.get(column, median)))
            match = max(0.0, 100.0 * (1.0 - abs(observed - median) / (2.0 * spread)))
            feature_scores[FEATURE_INPUT_NAMES[column]] = round(match, 1)
        results.append({
            "crop": str(crop_name),
            "dataset_fit": round(sum(feature_scores.values()) / len(feature_scores), 1),
            "feature_scores": feature_scores,
            "sample_count": int(len(profile_rows)),
            "source": DATASET_PATH.name,
        })
    return sorted(results, key=lambda item: item["dataset_fit"], reverse=True)
