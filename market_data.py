"""Yahoo Finance input validation and daily price retrieval."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yfinance as yf

HISTORICAL_PERIODS = ("1y", "2y", "5y", "10y")
_TICKER_PATTERN = re.compile(r"[A-Z0-9^][A-Z0-9.^=_-]{0,19}\Z")


def validate_ticker_symbol(symbol: str) -> str:
    """Normalize a Yahoo Finance symbol and reject unsupported input."""
    normalized = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not _TICKER_PATTERN.fullmatch(normalized):
        raise ValueError("Enter a valid Yahoo Finance ticker symbol, such as AAPL or RELIANCE.NS.")
    return normalized


def fetch_daily_history(symbol: str, selected_period: str) -> pd.DataFrame:
    """Fetch and validate adjusted daily closes returned by Yahoo Finance."""
    normalized = validate_ticker_symbol(symbol)
    if selected_period not in HISTORICAL_PERIODS:
        raise ValueError("Choose a supported historical period.")

    try:
        ticker = yf.Ticker(normalized)
        data = ticker.history(
            period=selected_period,
            interval="1d",
            auto_adjust=True,
            timeout=10,
        )
    except Exception as exc:
        raise ValueError("Yahoo Finance is unavailable or rate limited. Please retry later.") from exc

    if not isinstance(data, pd.DataFrame) or data.empty or "Close" not in data.columns:
        raise ValueError("No daily price data was returned. Check the ticker and try again.")
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans:
        raise ValueError("Yahoo Finance returned malformed daily dates. Please retry later.")

    close = data["Close"]
    if not isinstance(close, pd.Series):
        raise ValueError("Yahoo Finance returned malformed daily prices. Please retry later.")

    if data.index.tz is not None:
        latest_date = data.index.max().date()
        exchange_now = pd.Timestamp.now(tz=data.index.tz)
        if latest_date == exchange_now.date():
            metadata = ticker.history_metadata or {}
            current_period = metadata.get("currentTradingPeriod", {})
            regular_period = current_period.get("regular", {})
            session_end = regular_period.get("end")
            if session_end is None:
                session_is_open = True
            else:
                session_end = pd.Timestamp(session_end)
                if session_end.tz is None:
                    session_end = session_end.tz_localize(data.index.tz)
                else:
                    session_end = session_end.tz_convert(data.index.tz)
                session_is_open = exchange_now < session_end
            if session_is_open:
                close = close.loc[close.index.date < latest_date]

    close = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    close = close[close > 0].dropna()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.loc[~close.index.duplicated(keep="last")].sort_index()

    if len(close) < 60:
        raise ValueError("Fewer than 60 valid daily prices were returned. Try a different ticker or a longer period.")

    return close.to_frame(name="Close")
