import numpy as np
import pandas as pd

from predictor import build_supervised_data, make_next_feature_row, evaluate_models, fit_and_predict_next


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
    predicted = fit_and_predict_next(close, result.best_model_name)
    assert np.isfinite(predicted)
    assert predicted >= 0
