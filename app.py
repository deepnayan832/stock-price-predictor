from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import HISTORICAL_PERIODS, fetch_daily_history, validate_ticker_symbol
from predictor import evaluate_models, fit_and_predict_next

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor")
st.caption("Educational forecasting demo • Historical prices do not guarantee future performance.")


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol: str, selected_period: str) -> tuple[pd.DataFrame, datetime]:
    return fetch_daily_history(symbol, selected_period), datetime.now(timezone.utc)


with st.sidebar:
    st.header("Settings")
    with st.form("prediction_settings"):
        ticker = st.text_input("Yahoo Finance ticker", value="RELIANCE.NS", help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL")
        period = st.selectbox("Historical data", HISTORICAL_PERIODS, index=2)
        run_prediction = st.form_submit_button("Run prediction", type="primary", use_container_width=True)
        refresh_data = st.form_submit_button("Refresh data and run", use_container_width=True)
    st.markdown("---")
    st.caption("For NSE-listed Indian shares, Yahoo Finance symbols commonly end in `.NS`. BSE symbols commonly end in `.BO`.")


if refresh_data:
    load_history.clear()

if not run_prediction and not refresh_data:
    st.info("Choose a ticker and history period, then select **Run prediction**.")
    st.markdown("### What this demo does")
    st.markdown("- Downloads historical adjusted daily closing prices.\n- Compares Linear Regression and Random Forest with a previous-close baseline.\n- Evaluates across expanding chronological windows without random shuffling.\n- Selects the lower-MAE machine-learning model, retrains it on available history, and estimates the next session.")
    st.stop()

try:
    symbol = validate_ticker_symbol(ticker)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    with st.spinner(f"Downloading daily data and evaluating models for {symbol}…"):
        history, retrieved_at = load_history(symbol, period)
        close = history["Close"].astype(float)
        evaluation = evaluate_models(close)
        best_name = evaluation.best_model_name
        next_prediction = fit_and_predict_next(close, best_name)
except ValueError as exc:
    st.error(str(exc))
    st.caption("Yahoo Finance data may be delayed, rate limited, or temporarily unavailable. Try another ticker or period, or retry later.")
    st.stop()
except Exception:
    logger.exception("Prediction workflow failed for %s", symbol)
    st.error("Model evaluation or forecasting failed unexpectedly. Please retry later.")
    st.stop()

latest_date = close.index[-1]
latest_close = float(close.iloc[-1])
change = next_prediction - latest_close
change_pct = (change / latest_close * 100.0) if latest_close else 0.0
best_mae = float(evaluation.metrics.loc[evaluation.metrics["Model"] == best_name, "MAE"].iloc[0])
baseline_mae = float(
    evaluation.metrics.loc[evaluation.metrics["Model"] == "Previous Close Baseline", "MAE"].iloc[0]
)
test_start = evaluation.test_predictions.index[0]
test_end = evaluation.test_predictions.index[-1]
test_sample_size = len(evaluation.test_predictions)

st.success("Daily market data loaded; walk-forward evaluation and forecast completed.")
metric_cols = st.columns(4)
metric_cols[0].metric("Latest completed close", f"{latest_close:,.2f}", help=f"Data date: {latest_date:%d %b %Y}")
metric_cols[1].metric("Forecast: next session", f"{next_prediction:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
metric_cols[2].metric("Selected ML model", best_name)
metric_cols[3].metric("Walk-forward MAE", f"{best_mae:,.2f}", help="Average absolute error across the expanding chronological test windows. Lower is better.")

st.caption(
    f"Ticker: {symbol} • Daily data through {latest_date:%d %b %Y} • Retrieved {retrieved_at:%Y-%m-%d %H:%M UTC} "
    f"(cached up to 1 hour) • Walk-forward test: {test_start:%d %b %Y} to {test_end:%d %b %Y} "
    f"({test_sample_size} observations) • Forecast is for the next trading session, not a calendar-day guarantee."
)
if best_mae < baseline_mae:
    st.info(f"{best_name} had lower MAE than the previous-close baseline by {baseline_mae - best_mae:,.2f} on this test period. A single evaluation does not establish repeatable predictive value.")
else:
    st.warning(f"{best_name} did not outperform the previous-close baseline on MAE for this test period.")

left, right = st.columns([1.55, 1])
with left:
    st.subheader("Historical adjusted closing price")
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Adjusted close"))
    fig_history.update_layout(height=420, margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    st.plotly_chart(fig_history, use_container_width=True)

with right:
    st.subheader("Model comparison")
    metric_display = evaluation.metrics.copy()
    for col in ("MAE", "RMSE", "R²", "Directional Accuracy (%)"):
        metric_display[col] = metric_display[col].map(lambda x: f"{x:,.3f}" if pd.notna(x) else "N/A")
    st.dataframe(metric_display, hide_index=True, use_container_width=True)
    st.caption(f"Chronological test segment starts {evaluation.split_date:%d %b %Y}. Lower MAE/RMSE is better; R² can be negative.")

st.subheader("Actual vs predicted values on the walk-forward test")
test = evaluation.test_predictions.tail(150)
fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test.index, y=test["Actual"], mode="lines", name="Actual"))
fig_test.add_trace(go.Scatter(x=test.index, y=test[best_name], mode="lines", name=f"Predicted — {best_name}"))
fig_test.update_layout(height=420, margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
st.plotly_chart(fig_test, use_container_width=True)

with st.expander("How to interpret this result"):
    st.markdown(
        """**MAE** is the average absolute price error on the test dates; it is expressed in the ticker's quoted currency. **RMSE** penalizes larger errors more strongly. **R²** measures fit relative to a constant-mean baseline and can be negative. **Directional accuracy** measures how often the model correctly anticipated whether the close rose or fell versus the previous close.

        The evaluation uses three expanding chronological windows: each model is retrained using earlier rows before the next window. Features for each test day use only prices already observable by that day. The final forecast model is retrained using the full available history. This is a simple baseline, not a professional trading system."""
    )
st.warning("This project is for education only, not a buy/sell signal. It ignores news, results, valuation, liquidity, corporate events, transaction costs, and regime changes. Forecasts can be wrong; do not trade solely on this output.")
st.caption("Data source: Yahoo Finance via yfinance. Current-session daily bars are omitted until the regular session ends; data is not a guaranteed real-time exchange feed. Review the provider's terms before using downloaded data beyond personal research.")
