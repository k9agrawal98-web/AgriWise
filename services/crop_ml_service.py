from __future__ import annotations

import json
import hashlib
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "ml" / "crop_model.pkl"
METRICS_PATH = BASE_DIR / "ml" / "model_metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "ml" / "confusion_matrix.png"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
FEATURE_INPUT_NAMES = {
    "N": "nitrogen",
    "P": "phosphorus",
    "K": "potassium",
    "temperature": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
}
RANDOM_STATE = 42


def load_and_validate_dataset(path: Path | str = DATASET_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the unchanged source CSV and report transparent row-level quality checks."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    target_col = next((c for c in ("label", "crop", "Crop") if c in df.columns), None)
    if not target_col:
        raise ValueError(f"Dataset missing target column ('label' or 'crop'). Found: {list(df.columns)}")
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required feature columns: {missing_features}")

    original_count = len(df)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    null_counts = df[FEATURE_COLUMNS].isnull().sum().to_dict()
    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_feature_rows = int(df.duplicated(FEATURE_COLUMNS, keep=False).sum())
    conflicting_feature_label_groups = int(
        (df.groupby(FEATURE_COLUMNS, dropna=False)[target_col].nunique() > 1).sum()
    )
    finite = np.isfinite(df[FEATURE_COLUMNS]).all(axis=1)
    physical = (df[["N", "P", "K", "rainfall"]] >= 0).all(axis=1)
    physical &= df["ph"].between(0, 14) & df["humidity"].between(0, 100)
    valid = finite & physical & df[target_col].ne("")
    invalid_rows = int((~valid).sum())
    df = df.loc[valid].drop_duplicates().reset_index(drop=True)
    class_counts = df[target_col].value_counts().to_dict()

    quality_summary = {
        "source_rows": int(original_count),
        "total_rows": int(len(df)),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": target_col,
        "classes_count": int(df[target_col].nunique()),
        "classes": sorted(df[target_col].unique().tolist()),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "null_counts": {str(k): int(v) for k, v in null_counts.items()},
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_feature_rows": duplicate_feature_rows,
        "conflicting_feature_label_groups": conflicting_feature_label_groups,
        "invalid_rows_removed": invalid_rows,
        "feature_ranges": {
            col: {"min": float(df[col].min()), "max": float(df[col].max())}
            for col in FEATURE_COLUMNS
        },
    }
    return df, quality_summary


def _candidate_models() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    """Small, predeclared model comparison; all selection happens inside CV."""
    return {
        "Random Forest baseline": (
            RandomForestClassifier(n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1),
            {},
        ),
        "Random Forest tuned": (
            RandomForestClassifier(n_estimators=160, random_state=RANDOM_STATE, n_jobs=-1),
            {"max_depth": [None, 16], "min_samples_leaf": [1, 2]},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=160, random_state=RANDOM_STATE, n_jobs=-1),
            {"max_depth": [None, 16], "min_samples_leaf": [1, 2]},
        ),
        "Histogram Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"max_iter": [100, 180], "learning_rate": [0.08, 0.12], "l2_regularization": [0.0, 1.0]},
        ),
        "Scaled KNN": (
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1)),
            {"kneighborsclassifier__n_neighbors": [5, 9, 15], "kneighborsclassifier__weights": ["uniform", "distance"]},
        ),
    }


def _accuracy_wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = correct / total
    denom = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denom
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denom
    return [round(float(center - margin), 4), round(float(center + margin), 4)]


def train_and_save_crop_models(
    data_path: Path | str = DATASET_PATH,
    model_output_path: Path | str = MODEL_PATH,
    metrics_output_path: Path | str = METRICS_PATH,
    cm_image_path: Path | str = CONFUSION_MATRIX_PATH,
) -> Dict[str, Any]:
    """Select by training-only stratified CV, evaluate once, then refit for serving."""
    df, quality = load_and_validate_dataset(data_path)
    target_col = quality["target_column"]
    X, y = df[FEATURE_COLUMNS], df[target_col]
    if y.nunique() < 2 or int(y.value_counts().min()) < 6:
        raise ValueError("At least two classes with six or more valid observations each are required.")

    # Identical feature rows are treated as one group so a repeated underlying
    # measurement cannot cross a validation boundary.
    groups = pd.MultiIndex.from_frame(X).factorize()[0]
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer_cv.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups[train_idx]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    comparison = []
    best_name, best_model, best_cv_f1 = None, None, -1.0

    for name, (estimator, param_grid) in _candidate_models().items():
        if param_grid:
            search = GridSearchCV(
                estimator, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1,
                refit=True, return_train_score=False,
            )
            search.fit(X_train, y_train, groups=train_groups)
            candidate = search.best_estimator_
            cv_f1 = float(search.best_score_)
            params = search.best_params_
        else:
            candidate = estimator
            scores = []
            for fit_idx, val_idx in cv.split(X_train, y_train, train_groups):
                candidate_fold = estimator.__class__(**estimator.get_params())
                candidate_fold.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                scores.append(f1_score(y_train.iloc[val_idx], candidate_fold.predict(X_train.iloc[val_idx]), average="macro"))
            cv_f1 = float(np.mean(scores))
            candidate.fit(X_train, y_train)
            params = estimator.get_params(deep=False)

        comparison.append({"model_name": name, "cv_macro_f1_mean": round(cv_f1, 4), "parameters": params})
        if cv_f1 > best_cv_f1:
            best_name, best_model, best_cv_f1 = name, candidate, cv_f1

    if best_model is None:
        raise RuntimeError("No model was successfully selected.")

    baseline = next(item for item in comparison if item["model_name"] == "Random Forest baseline")
    baseline_model = RandomForestClassifier(
        n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    selected_preds = best_model.predict(X_test)

    labels = sorted(y.unique().tolist())
    baseline_report = classification_report(y_test, baseline_preds, labels=labels, output_dict=True, zero_division=0)
    selected_report = classification_report(y_test, selected_preds, labels=labels, output_dict=True, zero_division=0)

    def metrics_for(predictions: np.ndarray, report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "accuracy_95ci": _accuracy_wilson_interval(int((y_test.to_numpy() == predictions).sum()), len(y_test)),
            "macro_precision": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "per_class_metrics": {
                label: {
                    "precision": round(float(report[label]["precision"]), 4),
                    "recall": round(float(report[label]["recall"]), 4),
                    "f1_score": round(float(report[label]["f1-score"]), 4),
                    "support": int(report[label]["support"]),
                } for label in labels
            },
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        }

    baseline_metrics = metrics_for(baseline_preds, baseline_report)
    selected_metrics = metrics_for(selected_preds, selected_report)
    # The selected estimator is refit on all available labeled rows for serving;
    # all reported final metrics above are from the untouched 20% holdout.
    best_model.fit(X, y)

    model_out, metrics_out = Path(model_output_path), Path(metrics_output_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.asarray(selected_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        fig.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels,
               title=f"Untouched holdout confusion matrix ({best_name})",
               ylabel="Actual crop", xlabel="Predicted crop")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()
        cm_path = Path(cm_image_path)
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cm_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        if cm_path.resolve() == CONFUSION_MATRIX_PATH.resolve():
            display_path = BASE_DIR / "static" / "img" / "confusion_matrix.png"
            display_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cm_path, display_path)
        has_cm_plot = True
    except Exception as exc:
        print(f"Warning: Failed to render confusion matrix plot: {exc}")
        has_cm_plot = False

    payload = {
        "best_model": best_name,
        "selection_metric": "mean 5-fold macro-F1 on training split only",
        "best_cv_macro_f1": round(best_cv_f1, 4),
        "models_evaluated": comparison,
        "baseline_model": "Random Forest (n_estimators=100, max_depth=16)",
        "baseline_holdout_metrics": baseline_metrics,
        "selected_holdout_metrics": selected_metrics,
        "dataset_name": Path(data_path).name,
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "training_environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_size": quality["total_rows"],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURE_COLUMNS,
        "crop_classes": labels,
        "random_seed": RANDOM_STATE,
        "evaluation_protocol": "Stratified group 80/20 holdout (seed 42; identical feature rows grouped); stratified group five-fold macro-F1 CV and grid search on training partition only; one final holdout evaluation.",
        "test_set_used_for_selection": False,
        "calibration": "Not calibrated; classifier probabilities are not field success probabilities.",
        "per_class_metrics": selected_metrics["per_class_metrics"],
        "confusion_matrix": {"labels": labels, "values": selected_metrics["confusion_matrix"]},
        "has_confusion_matrix_plot": has_cm_plot,
        "data_quality_summary": quality,
        "limitations": [
            "Dataset provenance, license, units, geography, collection period, and sampling independence are unverified.",
            "Random stratified holdout measures only this dataset distribution, not performance in farms or unseen regions.",
            "The seven input features omit location, soil moisture, irrigation, season, economics, and other local constraints.",
        ],
    }
    with metrics_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def ensure_crop_model_exists() -> Path:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Crop model unavailable. Run python ml/train_model.py after reviewing dataset provenance.")
    return MODEL_PATH


def get_model_evaluation_metrics() -> Dict[str, Any] | None:
    """Load saved model evaluation metadata for the research and reporting views."""
    if not METRICS_PATH.exists():
        return None
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def predict_crop_ml(features: Dict[str, Any]) -> Dict[str, Any]:
    """Return a model ranking; probability scores are uncalibrated classifier outputs."""
    ensure_crop_model_exists()
    model = joblib.load(MODEL_PATH)
    from services.crop_service import validate_prediction_features
    feature_vals = validate_prediction_features(features)
    input_df = pd.DataFrame([feature_vals], columns=FEATURE_COLUMNS)
    raw_pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
    if probs is None:
        top_3 = [{"crop": str(raw_pred), "confidence": None}]
        probabilities_dict = {str(raw_pred): None}
        confidence = None
    else:
        classes = model.classes_
        sorted_indices = np.argsort(probs)[::-1]
        probabilities_dict = {str(classes[i]): round(float(probs[i]) * 100.0, 2) for i in sorted_indices}
        top_3 = [{"crop": str(classes[i]), "confidence": probabilities_dict[str(classes[i])]} for i in sorted_indices[:3]]
        confidence = top_3[0]["confidence"] if top_3 else None
    return {
        "recommended_crop": str(raw_pred),
        "confidence": confidence,
        "confidence_type": "uncalibrated classifier probability",
        "top_3": top_3,
        "all_probabilities": probabilities_dict,
        "input_features": dict(zip(FEATURE_COLUMNS, feature_vals)),
    }
