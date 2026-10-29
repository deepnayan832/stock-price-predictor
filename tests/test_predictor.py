import numpy as np
import pandas as pd
import pytest

import predictor
from predictor import (
    FEATURE_COLUMNS,
    build_supervised_data,
    clean_close_series,
    evaluate_models,
    fit_and_predict_next,
    make_next_feature_row,
)


def sample_close(n=260):
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.012, n)
    values = 150 * np.cumprod(1 + returns)
    index = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(values, index=index, name="Close")


def test_supervised_features_are_shifted_and_nonempty():
    close = sample_close()
    data = build_supervised_data(close)
    assert "target" in data.columns
    assert data.index[0] > close.index[20]
    assert np.isclose(data.iloc[0]["lag_1"], close.loc[data.index[0] - pd.offsets.BDay(1)])


def test_supervised_features_do_not_use_the_target_day_close():
    close = sample_close()
    data = build_supervised_data(close)
    target_date = data.index[100]
    changed_close = close.copy()
    changed_close.loc[target_date] *= 10

    changed_data = build_supervised_data(changed_close)

    pd.testing.assert_series_equal(
        data.loc[target_date, list(FEATURE_COLUMNS)],
        changed_data.loc[target_date, list(FEATURE_COLUMNS)],
    )
    assert data.loc[target_date, "target"] != changed_data.loc[target_date, "target"]


def test_clean_close_series_drops_invalid_values_and_keeps_last_duplicate():
    close = sample_close(80).astype(object)
    close.iloc[0] = "invalid"
    close.iloc[1] = np.nan
    close.iloc[2] = np.inf
    close.iloc[3] = -np.inf
    close.iloc[4] = 0
    close = pd.concat([close, pd.Series([222.5], index=[close.index[5]])])

    cleaned = clean_close_series(close)

    assert len(cleaned) == 75
    assert cleaned.index.is_unique
    assert cleaned.loc[close.index[5]] == 222.5
    assert np.isfinite(cleaned.to_numpy()).all()
    assert (cleaned > 0).all()


def test_next_row_uses_latest_close_as_lag_one():
    close = sample_close()
    row = make_next_feature_row(close)
    assert row.iloc[0]["lag_1"] == close.iloc[-1]
    assert row.iloc[0]["lag_2"] == close.iloc[-2]


def test_models_evaluate_chronologically_and_predict_positive_value():
    close = sample_close()
    result = evaluate_models(close)
    assert result.metrics.iloc[0]["MAE"] <= result.metrics.iloc[-1]["MAE"]
    assert result.split_date > close.index[0]
    assert result.test_predictions.index.min() == result.split_date
    assert result.test_predictions.index[-1] == close.index[-1]
    assert len(result.test_predictions) >= 10
    assert set(result.metrics["Model"]) == {
        "Linear Regression",
        "Random Forest",
        "Previous Close Baseline",
    }
    assert result.best_model_name in {"Linear Regression", "Random Forest"}
    predicted = fit_and_predict_next(close, result.best_model_name)
    assert np.isfinite(predicted)
    assert predicted >= 0


def test_evaluation_retrains_with_expanding_chronological_windows(monkeypatch):
    fit_end_dates = []

    class RecordingModel:
        def __init__(self, *args, **kwargs):
            self.mean = 0.0

        def fit(self, features, target):
            fit_end_dates.append(features.index[-1])
            self.mean = float(target.mean())
            return self

        def predict(self, features):
            return np.full(len(features), self.mean)

    monkeypatch.setattr(predictor, "LinearRegression", RecordingModel)
    monkeypatch.setattr(predictor, "RandomForestRegressor", RecordingModel)

    predictor.evaluate_models(sample_close())

    distinct_ends = sorted(set(fit_end_dates))
    assert len(fit_end_dates) == 6
    assert len(distinct_ends) == 3
    assert distinct_ends[0] < distinct_ends[1] < distinct_ends[2]
