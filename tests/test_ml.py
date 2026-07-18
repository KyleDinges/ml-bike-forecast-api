from pathlib import Path

import pandas as pd

from bike_demand_api.ml import (
    MODEL_FEATURE_COLUMNS,
    chronological_split,
    feature_frame_from_raw,
    build_drift_baseline,
    drift_report,
    should_promote_candidate,
)


RAW_PATH = Path(__file__).parents[1] / "data" / "raw" / "hour.csv"


def test_feature_pipeline_excludes_target_leakage_and_splits_in_time():
    features, target = feature_frame_from_raw(pd.read_csv(RAW_PATH))
    (train_x, _), (validation_x, _), (test_x, _) = chronological_split(features, target)

    assert "casual" not in MODEL_FEATURE_COLUMNS
    assert "registered" not in MODEL_FEATURE_COLUMNS
    assert "cnt" not in MODEL_FEATURE_COLUMNS
    assert "instant" not in MODEL_FEATURE_COLUMNS
    assert train_x["date"].max() <= validation_x["date"].min()
    assert validation_x["date"].max() <= test_x["date"].min()


def test_promotion_requires_a_meaningful_improvement():
    assert should_promote_candidate(100.0, 98.9)
    assert not should_promote_candidate(100.0, 99.1)
    assert not should_promote_candidate(100.0, 101.0)


def test_drift_reports_insufficient_stable_and_shifted_batches():
    features, _ = feature_frame_from_raw(pd.read_csv(RAW_PATH))
    baseline = build_drift_baseline(features.iloc[:1_000])
    assert baseline["weather_conditioned"]["columns"] == ["month", "hour"]
    assert "1|0" in baseline["weather_conditioned"]["cohorts"]
    insufficient = drift_report(features.iloc[:5], baseline, 30, 0.10, 0.25)
    stable = drift_report(features.iloc[:1_000], baseline, 30, 0.10, 0.25)
    shifted = features.iloc[:1_000].copy()
    shifted["temperature_normalized"] = 1.0
    shifted["weather_condition"] = 4
    shifted_result = drift_report(shifted, baseline, 30, 0.10, 0.25)

    assert insufficient["status"] == "insufficient_sample"
    assert stable["status"] in {"stable", "watch"}
    assert shifted_result["status"] == "drifted"
