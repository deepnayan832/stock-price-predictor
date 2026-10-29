from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from predictor import evaluate_models, fit_and_predict_next

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor")
st.caption("Educational forecasting demo • Historical prices do not guarantee future performance.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Yahoo Finance ticker", value="RELIANCE.NS", help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL")
    period = st.selectbox("Historical data", ["1y", "2y", "5y", "10y"], index=2)
    st.markdown("---")
    st.caption("For NSE-listed Indian shares, Yahoo Finance symbols commonly end in `.NS`. BSE symbols commonly end in `.BO`.")


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol: str, selected_period: str) -> pd.DataFrame:
    data = yf.Ticker(symbol.strip().upper()).history(period=selected_period, interval="1d", auto_adjust=True)
    if data is None or data.empty or "Close" not in data.columns:
        raise ValueError("No daily price data was returned. Check the ticker and try again.")
    data = data.loc[:, ["Close"]].dropna().copy()
    data.index = pd.to_datetime(data.index).tz_localize(None) if getattr(data.index, "tz", None) is not None else pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    if len(data) < 60:
        raise ValueError("Fewer than 60 valid daily prices were returned. Try a different ticker or period.")
    return data


if st.button("Run prediction", type="primary", use_container_width=False):
    st.session_state["run_prediction"] = True

if not st.session_state.get("run_prediction", False):
    st.info("Choose a ticker and history period, then select **Run prediction**.")
    st.markdown("### What this demo does")
    st.markdown("- Downloads historical adjusted daily closing prices.\n- Compares Linear Regression with Random Forest.\n- Evaluates on a later chronological holdout (no random shuffling).\n- Selects the lower-MAE model, retrains it on available history, and forecasts the next session.")
    st.stop()

symbol = ticker.strip().upper()
if not symbol:
    st.error("Enter a Yahoo Finance ticker symbol.")
    st.stop()

try:
    with st.spinner(f"Downloading data and evaluating models for {symbol}…"):
        history = load_history(symbol, period)
        close = history["Close"].astype(float)
        evaluation = evaluate_models(close)
        best_name = evaluation.best_model_name
        next_prediction = fit_and_predict_next(close, best_name)
except Exception as exc:
    st.error(f"Could not complete the prediction: {exc}")
    st.caption("Check the ticker, internet connection, and package installation. Yahoo Finance data may occasionally be unavailable.")
    st.stop()

latest_date = close.index[-1]
latest_close = float(close.iloc[-1])
change = next_prediction - latest_close
change_pct = (change / latest_close * 100.0) if latest_close else 0.0

metric_cols = st.columns(4)
metric_cols[0].metric("Latest adjusted close", f"{latest_close:,.2f}", help=f"Data date: {latest_date:%d %b %Y}")
metric_cols[1].metric("Forecast: next session", f"{next_prediction:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
metric_cols[2].metric("Selected model", best_name)
best_mae = float(evaluation.metrics.loc[evaluation.metrics["Model"] == best_name, "MAE"].iloc[0])
metric_cols[3].metric("Holdout MAE", f"{best_mae:,.2f}", help="Average absolute error on the chronological test segment. Lower is better.")

st.caption(f"Ticker: {symbol} • Historical data through {latest_date:%d %b %Y} • Forecast is for the next trading session, not a calendar-day guarantee.")

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

st.subheader("Actual vs predicted values on the test segment")
test = evaluation.test_predictions.tail(150)
fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test.index, y=test["Actual"], mode="lines", name="Actual"))
fig_test.add_trace(go.Scatter(x=test.index, y=test[best_name], mode="lines", name=f"Predicted — {best_name}"))
fig_test.update_layout(height=420, margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
st.plotly_chart(fig_test, use_container_width=True)

with st.expander("How to interpret this result"):
    st.markdown(
        """**MAE** is the average absolute price error on the test dates; it is expressed in the ticker's quoted currency. **RMSE** penalizes larger errors more strongly. **R²** measures fit relative to a constant-mean baseline and can be negative. **Directional accuracy** measures how often the model correctly anticipated whether the close rose or fell versus the previous close.

        The test is chronological: the model is trained on earlier rows and evaluated on later rows. Features for each test day use only prices that would already have been observed by that day. The final forecast model is then retrained using the full available history. This is a simple baseline, not a professional trading system."""
    )
st.warning("This project is for education only, not a buy/sell signal. It ignores news, results, valuation, liquidity, corporate events, transaction costs, and regime changes. Forecasts can be wrong; do not trade solely on this output.")
st.caption("Data source: Yahoo Finance via yfinance. Review the provider's terms before using downloaded data beyond personal research.")
