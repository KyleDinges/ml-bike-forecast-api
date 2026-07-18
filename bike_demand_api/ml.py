"""Feature construction, model training, artifacts, and drift calculations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


NUMERIC_DRIFT_FEATURES = [
    "temperature_normalized",
    "feels_like_temperature_normalized",
    "humidity_normalized",
    "windspeed_normalized",
]
CATEGORICAL_DRIFT_FEATURES = ["weekday", "holiday", "workingday"]
WEATHER_CATEGORICAL_DRIFT_FEATURES = ["weather_condition"]
WEATHER_DRIFT_COHORT_COLUMNS = ["month", "hour"]
NORMALIZED_WEATHER_BIN_EDGES = np.linspace(0, 1, 11)
MODEL_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday",
    "holiday",
    "workingday",
    "weather_condition",
    *NUMERIC_DRIFT_FEATURES,
]


@dataclass
class SeasonalMedianBaseline:
    """A reproducible demand baseline using training-only hour/workday/month medians."""

    medians: dict[str, float]
    global_median: float

    @staticmethod
    def _keys(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["hour"].astype(int).astype(str)
            + "|"
            + frame["workingday"].astype(int).astype(str)
            + "|"
            + frame["month"].astype(int).astype(str)
        )

    @classmethod
    def fit(cls, frame: pd.DataFrame, target: pd.Series) -> "SeasonalMedianBaseline":
        keyed = pd.DataFrame({"key": cls._keys(frame), "target": target.to_numpy()})
        medians = keyed.groupby("key")["target"].median().to_dict()
        return cls({str(key): float(value) for key, value in medians.items()}, float(target.median()))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._keys(frame).map(self.medians).fillna(self.global_median).to_numpy(dtype=float)


def feature_frame_from_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the committed UCI frame into non-leaking model features and target."""
    required = {"dteday", "hr", "holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw bike data is missing columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["dteday"]),
            "hour": raw["hr"].astype(int),
            "holiday": raw["holiday"].astype(int),
            "workingday": raw["workingday"].astype(int),
            "weather_condition": raw["weathersit"].astype(int),
            "temperature_normalized": raw["temp"].astype(float),
            "feels_like_temperature_normalized": raw["atemp"].astype(float),
            "humidity_normalized": raw["hum"].astype(float),
            "windspeed_normalized": raw["windspeed"].astype(float),
        }
    )
    return enrich_features(frame), raw["cnt"].astype(float)


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar and cyclical fields from request-compatible feature inputs."""
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    output["hour"] = output["hour"].astype(int)
    output["month"] = output["date"].dt.month.astype(int)
    output["weekday"] = output["date"].dt.weekday.astype(int)
    output["hour_sin"] = np.sin(2 * np.pi * output["hour"] / 24)
    output["hour_cos"] = np.cos(2 * np.pi * output["hour"] / 24)
    output["month_sin"] = np.sin(2 * np.pi * (output["month"] - 1) / 12)
    output["month_cos"] = np.cos(2 * np.pi * (output["month"] - 1) / 12)
    for column in ("holiday", "workingday", "weather_condition"):
        output[column] = output[column].astype(int)
    return output


def chronological_split(frame: pd.DataFrame, target: pd.Series) -> tuple[tuple[pd.DataFrame, pd.Series], ...]:
    ordered = frame.assign(_timestamp=frame["date"] + pd.to_timedelta(frame["hour"], unit="h")).sort_values("_timestamp")
    ordered_target = target.loc[ordered.index]
    total = len(ordered)
    train_end, validation_end = int(total * 0.6), int(total * 0.8)
    return (
        (ordered.iloc[:train_end].drop(columns="_timestamp"), ordered_target.iloc[:train_end]),
        (ordered.iloc[train_end:validation_end].drop(columns="_timestamp"), ordered_target.iloc[train_end:validation_end]),
        (ordered.iloc[validation_end:].drop(columns="_timestamp"), ordered_target.iloc[validation_end:]),
    )


def _metrics(actual: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
    }


def should_promote_candidate(baseline_mae: float, candidate_mae: float, minimum_improvement: float = 0.01) -> bool:
    """Require a meaningful validation-MAE improvement before promotion."""
    return candidate_mae <= baseline_mae * (1 - minimum_improvement)


def build_drift_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    baseline: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for feature in CATEGORICAL_DRIFT_FEATURES:
        counts = frame[feature].astype(str).value_counts().sort_index()
        baseline["categorical"][feature] = {"proportions": {key: float(value / len(frame)) for key, value in counts.items()}}
    cohorts = {
        _weather_cohort_key(month, hour): _weather_reference(group)
        for (month, hour), group in frame.groupby(WEATHER_DRIFT_COHORT_COLUMNS)
    }
    baseline["weather_conditioned"] = {
        "columns": WEATHER_DRIFT_COHORT_COLUMNS,
        "numeric_edges": NORMALIZED_WEATHER_BIN_EDGES.tolist(),
        "cohorts": cohorts,
        "fallback": _weather_reference(frame),
    }
    return baseline


def _weather_cohort_key(month: int, hour: int) -> str:
    return f"{int(month)}|{int(hour)}"


def _weather_reference(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = {}
    for feature in NUMERIC_DRIFT_FEATURES:
        counts, _ = np.histogram(frame[feature].to_numpy(dtype=float), bins=NORMALIZED_WEATHER_BIN_EDGES)
        numeric[feature] = {"proportions": _proportions(counts).tolist()}
    categorical = {}
    for feature in WEATHER_CATEGORICAL_DRIFT_FEATURES:
        counts = frame[feature].astype(str).value_counts()
        categorical[feature] = {"proportions": {key: float(value / len(frame)) for key, value in counts.items()}}
    return {"numeric": numeric, "categorical": categorical}


def _proportions(counts: np.ndarray) -> np.ndarray:
    smoothed = np.asarray(counts, dtype=float) + 1e-6
    return smoothed / smoothed.sum()


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    actual = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(frame: pd.DataFrame, baseline: dict[str, Any], minimum_sample_size: int, watch: float, alert: float) -> dict[str, Any]:
    if len(frame) < minimum_sample_size:
        return {"status": "insufficient_sample", "sample_size": len(frame), "minimum_sample_size": minimum_sample_size, "features": []}
    features: list[dict[str, Any]] = []
    for feature, reference in baseline["categorical"].items():
        expected_map = reference["proportions"]
        actual_map = frame[feature].astype(str).value_counts(normalize=True).to_dict()
        keys = sorted(set(expected_map) | set(actual_map))
        psi = population_stability_index(
            np.array([expected_map.get(key, 0.0) for key in keys]),
            np.array([actual_map.get(key, 0.0) for key in keys]),
        )
        features.append({"feature": feature, "kind": "categorical", "psi": psi, "status": _drift_status(psi, watch, alert)})
    conditioned = baseline["weather_conditioned"]
    cohort_counts = frame.groupby(WEATHER_DRIFT_COHORT_COLUMNS).size()
    for feature in NUMERIC_DRIFT_FEATURES:
        expected = _conditional_weather_proportions(cohort_counts, conditioned, "numeric", feature)
        actual_counts, _ = np.histogram(frame[feature].to_numpy(dtype=float), bins=np.asarray(conditioned["numeric_edges"], dtype=float))
        psi = population_stability_index(expected, _proportions(actual_counts))
        features.append({"feature": feature, "kind": "numeric", "psi": psi, "status": _drift_status(psi, watch, alert)})
    for feature in WEATHER_CATEGORICAL_DRIFT_FEATURES:
        expected_map = _conditional_weather_category_proportions(cohort_counts, conditioned, feature)
        actual_map = frame[feature].astype(str).value_counts(normalize=True).to_dict()
        keys = sorted(set(expected_map) | set(actual_map))
        psi = population_stability_index(
            np.array([expected_map.get(key, 0.0) for key in keys]),
            np.array([actual_map.get(key, 0.0) for key in keys]),
        )
        features.append({"feature": feature, "kind": "categorical", "psi": psi, "status": _drift_status(psi, watch, alert)})
    statuses = {feature["status"] for feature in features}
    overall = "drifted" if "drifted" in statuses else "watch" if "watch" in statuses else "stable"
    return {"status": overall, "sample_size": len(frame), "minimum_sample_size": minimum_sample_size, "features": features}


def _conditional_weather_proportions(
    cohort_counts: pd.Series, conditioned: dict[str, Any], kind: str, feature: str
) -> np.ndarray:
    expected = np.zeros(len(NORMALIZED_WEATHER_BIN_EDGES) - 1)
    total = cohort_counts.sum()
    for (month, hour), count in cohort_counts.items():
        cohort = conditioned["cohorts"].get(_weather_cohort_key(month, hour), conditioned["fallback"])
        expected += (count / total) * np.asarray(cohort[kind][feature]["proportions"])
    return expected


def _conditional_weather_category_proportions(
    cohort_counts: pd.Series, conditioned: dict[str, Any], feature: str
) -> dict[str, float]:
    expected: dict[str, float] = {}
    total = cohort_counts.sum()
    for (month, hour), count in cohort_counts.items():
        cohort = conditioned["cohorts"].get(_weather_cohort_key(month, hour), conditioned["fallback"])
        for category, proportion in cohort["categorical"][feature]["proportions"].items():
            expected[category] = expected.get(category, 0.0) + (count / total) * proportion
    return expected


def _drift_status(psi: float, watch: float, alert: float) -> str:
    return "drifted" if psi > alert else "watch" if psi >= watch else "stable"


def train_and_save(raw_path: Path, artifact_path: Path, manifest_path: Path, report_path: Path) -> dict[str, Any]:
    raw = pd.read_csv(raw_path)
    features, target = feature_frame_from_raw(raw)
    (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = chronological_split(features, target)
    baseline = SeasonalMedianBaseline.fit(train_x, train_y)
    baseline_validation = baseline.predict(validation_x)
    candidate = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )
    candidate.fit(train_x[MODEL_FEATURE_COLUMNS], train_y)
    candidate_validation = candidate.predict(validation_x[MODEL_FEATURE_COLUMNS])
    baseline_metrics = _metrics(validation_y, baseline_validation)
    candidate_metrics = _metrics(validation_y, candidate_validation)
    improvement = (baseline_metrics["mae"] - candidate_metrics["mae"]) / baseline_metrics["mae"]
    candidate_promoted = should_promote_candidate(baseline_metrics["mae"], candidate_metrics["mae"])
    approved_name = "XGBoost" if candidate_promoted else "SeasonalMedian"
    approved_model: Any = candidate if candidate_promoted else baseline
    approved_validation = candidate_validation if candidate_promoted else baseline_validation
    test_prediction = (
        candidate.predict(test_x[MODEL_FEATURE_COLUMNS]) if candidate_promoted else baseline.predict(test_x)
    )
    residual_radius = float(np.quantile(np.abs(validation_y.to_numpy() - approved_validation), 0.90))
    test_metrics = _metrics(test_y, test_prediction)
    lower = np.maximum(0, test_prediction - residual_radius)
    upper = test_prediction + residual_radius
    test_metrics["interval_coverage"] = float(np.mean((test_y.to_numpy() >= lower) & (test_y.to_numpy() <= upper)))
    data_fingerprint = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    drift_baseline = build_drift_baseline(train_x)
    manifest = {
        "model_version": "1.0.0",
        "feature_schema_version": 1,
        "approved_model": approved_name,
        "baseline_model": "SeasonalMedian",
        "candidate_model": "XGBoost",
        "validation_metrics": {"baseline": baseline_metrics, "candidate": candidate_metrics},
        "promotion_rule": "candidate_validation_mae <= baseline_validation_mae * 0.99",
        "promotion_decision": "candidate_promoted" if candidate_promoted else "baseline_retained",
        "promotion_reason": (
            f"XGBoost improved validation MAE by {improvement:.2%}."
            if candidate_promoted
            else f"XGBoost improvement of {improvement:.2%} did not meet the 1.00% threshold."
        ),
        "test_metrics": test_metrics,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "data_fingerprint_sha256": data_fingerprint,
        "split_periods": {
            "train": [str(train_x["date"].min().date()), str(train_x["date"].max().date())],
            "validation": [str(validation_x["date"].min().date()), str(validation_x["date"].max().date())],
            "test": [str(test_x["date"].min().date()), str(test_x["date"].max().date())],
        },
        "candidate_parameters": candidate.get_params(),
        "interval": {"method": "empirical_validation_absolute_residual_q90", "radius": residual_radius},
        "drift_baseline": drift_baseline,
        "default_drift_configuration": {
            "minimum_batch_size": 30,
            "psi_watch_threshold": 0.10,
            "psi_alert_threshold": 0.25,
            "weather_baseline_conditioning": "month_hour",
        },
    }
    bundle = {"model": approved_model, "approved_model": approved_name, "interval_radius": residual_radius}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
