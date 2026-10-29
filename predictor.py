"""Core time-series feature engineering and model evaluation for the demo app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

LAGS = (1, 2, 3, 5, 10, 20)
WINDOWS = (5, 10, 20)
FEATURE_COLUMNS = (
    tuple(f"lag_{lag}" for lag in LAGS)
    + tuple(f"mean_{window}" for window in WINDOWS)
    + tuple(f"std_{window}" for window in WINDOWS)
    + tuple(f"momentum_{window}" for window in WINDOWS)
)


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    best_model_name: str
    fitted_test_models: dict[str, Any]
    split_date: pd.Timestamp


def clean_close_series(close: pd.Series) -> pd.Series:
    """Return a sorted, numeric Close series with invalid rows removed."""
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    result = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    result = result[result > 0]
    if len(result) < 60:
        raise ValueError("At least 60 valid daily closing prices are needed; 1–5 years of history is recommended.")
    return result


def build_supervised_data(close: pd.Series) -> pd.DataFrame:
    """Build a one-session-ahead dataset using only information available before each target day."""
    close = clean_close_series(close)
    frame = pd.DataFrame(index=close.index)

    for lag in LAGS:
        frame[f"lag_{lag}"] = close.shift(lag)

    prior_close = close.shift(1)
    for window in WINDOWS:
        frame[f"mean_{window}"] = prior_close.rolling(window).mean()
        frame[f"std_{window}"] = prior_close.rolling(window).std()
        # Momentum is measured from the last known close, never from the target day.
        frame[f"momentum_{window}"] = prior_close / close.shift(window + 1) - 1.0

    frame["target"] = close
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 50:
        raise ValueError("Not enough usable rows after feature engineering. Use a longer history period.")
    return frame


def make_next_feature_row(close: pd.Series) -> pd.DataFrame:
    """Create features for the next trading session from currently known closes."""
    close = clean_close_series(close)
    values: dict[str, float] = {}
    for lag in LAGS:
        values[f"lag_{lag}"] = float(close.iloc[-lag])

    for window in WINDOWS:
        recent = close.iloc[-window:]
        values[f"mean_{window}"] = float(recent.mean())
        values[f"std_{window}"] = float(recent.std(ddof=1))
        values[f"momentum_{window}"] = float(close.iloc[-1] / close.iloc[-(window + 1)] - 1.0)

    return pd.DataFrame([values], columns=FEATURE_COLUMNS, index=[close.index[-1]])


def evaluate_models(close: pd.Series, test_fraction: float = 0.2) -> EvaluationResult:
    """Evaluate models across expanding chronological windows; do not shuffle time."""
    close = clean_close_series(close)
    data = build_supervised_data(close)
    if not 0.10 <= test_fraction <= 0.35:
        raise ValueError("test_fraction must be between 0.10 and 0.35.")

    split_count = 3
    test_size = max(4, int(np.ceil(len(data) * test_fraction / split_count)))
    if len(data) - test_size * split_count < 40:
        raise ValueError("Not enough rows for a reliable expanding walk-forward evaluation.")

    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }
    prediction_frames: list[pd.DataFrame] = []
    splitter = TimeSeriesSplit(n_splits=split_count, test_size=test_size)
    for train_indices, test_indices in splitter.split(data):
        train = data.iloc[train_indices]
        test = data.iloc[test_indices]
        y_test = test["target"]
        previous_close = close.shift(1).reindex(y_test.index)
        fold_predictions = pd.DataFrame(index=y_test.index)
        fold_predictions["Actual"] = y_test
        fold_predictions["Previous Close Baseline"] = previous_close.to_numpy()

        for name, model in models.items():
            model.fit(train[list(FEATURE_COLUMNS)], train["target"])
            fold_predictions[name] = np.asarray(
                model.predict(test[list(FEATURE_COLUMNS)]), dtype=float
            )

        prediction_frames.append(fold_predictions)

    test_predictions = pd.concat(prediction_frames).sort_index()
    rows: list[dict[str, float | str]] = []
    y_test = test_predictions["Actual"].to_numpy(dtype=float)
    previous_close = test_predictions["Previous Close Baseline"].to_numpy(dtype=float)
    for name in models:
        prediction = test_predictions[name].to_numpy(dtype=float)
        true_direction = np.sign(y_test - previous_close)
        predicted_direction = np.sign(prediction - previous_close)
        valid_direction = np.isfinite(true_direction) & np.isfinite(predicted_direction)
        directional_accuracy = (
            float(np.mean(true_direction[valid_direction] == predicted_direction[valid_direction]) * 100.0)
            if valid_direction.any()
            else float("nan")
        )
        rows.append(
            {
                "Model": name,
                "MAE": float(mean_absolute_error(y_test, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
                "R²": float(r2_score(y_test, prediction)),
                "Directional Accuracy (%)": directional_accuracy,
            }
        )

    # Include the persistence forecast (tomorrow equals the latest known close) as a baseline.
    baseline_prediction = previous_close
    baseline_true = y_test
    rows.append(
        {
            "Model": "Previous Close Baseline",
            "MAE": float(mean_absolute_error(baseline_true, baseline_prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(baseline_true, baseline_prediction))),
            "R²": float(r2_score(baseline_true, baseline_prediction)),
            "Directional Accuracy (%)": float("nan"),
        }
    )
    metrics = pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)
    # Select between the trainable models only; the baseline is for comparison.
    learned_rows = [row for row in rows if row["Model"] in models]
    best_model_name = str(min(learned_rows, key=lambda row: float(row["MAE"]))["Model"])
    return EvaluationResult(
        metrics=metrics,
        test_predictions=test_predictions,
        best_model_name=best_model_name,
        fitted_test_models=models,
        split_date=pd.Timestamp(test_predictions.index[0]),
    )


def fit_and_predict_next(close: pd.Series, model_name: str) -> float:
    """Refit the chosen model on all available supervised rows, then forecast one session ahead."""
    close = clean_close_series(close)
    data = build_supervised_data(close)
    if model_name == "Linear Regression":
        model: Any = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(data[list(FEATURE_COLUMNS)], data["target"])
    next_row = make_next_feature_row(close)
    prediction = float(model.predict(next_row[list(FEATURE_COLUMNS)])[0])
    # A model should not produce a negative stock price. Clamp only as a sanity safeguard.
    return max(0.0, prediction)
