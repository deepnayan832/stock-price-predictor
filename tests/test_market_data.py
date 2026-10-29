import numpy as np
import pandas as pd
import pytest

import market_data
from market_data import fetch_daily_history, validate_ticker_symbol


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("tcs.ns", "TCS.NS"),
        ("INFY.NS", "INFY.NS"),
        ("AAPL", "AAPL"),
        ("BRK-B", "BRK-B"),
        ("EURUSD=X", "EURUSD=X"),
        ("^NSEI", "^NSEI"),
    ],
)
def test_validate_ticker_symbol_accepts_common_yahoo_symbols(symbol, expected):
    assert validate_ticker_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", ["", "AAPL!", "../AAPL", "AAPL;DROP", "A" * 21])
def test_validate_ticker_symbol_rejects_invalid_values(symbol):
    with pytest.raises(ValueError, match="ticker"):
        validate_ticker_symbol(symbol)


def install_fake_ticker(monkeypatch, data=None, error=None, metadata=None):
    calls = {}

    class FakeTicker:
        def __init__(self, symbol):
            calls["symbol"] = symbol
            self.history_metadata = metadata or {}

        def history(self, **kwargs):
            calls["history_kwargs"] = kwargs
            if error is not None:
                raise error
            return data

    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    return calls


def test_fetch_daily_history_normalizes_provider_observations(monkeypatch):
    index = pd.date_range("2024-01-01", periods=80, tz="America/New_York")
    values = [100.0 + row for row in range(80)]
    values[0] = "not-a-number"
    values[1] = np.nan
    values[2] = np.inf
    values[3] = -np.inf
    values[4] = 0
    data = pd.DataFrame({"Close": values}, index=index)
    data = pd.concat([data, pd.DataFrame({"Close": [222.5]}, index=[index[5]])])
    calls = install_fake_ticker(monkeypatch, data)

    history = fetch_daily_history("tcs.ns", "1y")

    assert calls["symbol"] == "TCS.NS"
    assert calls["history_kwargs"] == {
        "period": "1y",
        "interval": "1d",
        "auto_adjust": True,
        "timeout": 10,
    }
    assert len(history) == 75
    assert history.index.is_unique
    assert history.index.is_monotonic_increasing
    assert history.index.tz is None
    assert history.loc[index[5].tz_localize(None), "Close"] == 222.5
    assert np.isfinite(history["Close"].to_numpy()).all()
    assert (history["Close"] > 0).all()


def test_invalid_ticker_and_period_do_not_call_provider(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda symbol: pytest.fail("provider called"))

    with pytest.raises(ValueError, match="ticker"):
        fetch_daily_history("bad ticker", "1y")
    with pytest.raises(ValueError, match="period"):
        fetch_daily_history("AAPL", "max")


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(),
        pd.DataFrame({"Open": [1.0]}, index=pd.date_range("2024-01-01", periods=1)),
    ],
)
def test_fetch_daily_history_rejects_missing_or_malformed_data(monkeypatch, data):
    install_fake_ticker(monkeypatch, data)

    with pytest.raises(ValueError, match="daily price data"):
        fetch_daily_history("AAPL", "1y")


def test_fetch_daily_history_rejects_insufficient_valid_data(monkeypatch):
    index = pd.date_range("2024-01-01", periods=59)
    install_fake_ticker(monkeypatch, pd.DataFrame({"Close": np.arange(100, 159)}, index=index))

    with pytest.raises(ValueError, match="60 valid"):
        fetch_daily_history("AAPL", "1y")


def test_fetch_daily_history_rejects_malformed_dates(monkeypatch):
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.NaT])
    install_fake_ticker(monkeypatch, pd.DataFrame({"Close": [100.0, 101.0]}, index=index))

    with pytest.raises(ValueError, match="malformed daily dates"):
        fetch_daily_history("AAPL", "1y")


@pytest.mark.parametrize(("end_offset", "expected_count"), [(1, 79), (-1, 80), (None, 79)])
def test_fetch_daily_history_only_uses_completed_current_session(monkeypatch, end_offset, expected_count):
    timezone = "America/New_York"
    now = pd.Timestamp.now(tz=timezone)
    today = now.normalize()
    index = list(pd.bdate_range(end=today - pd.Timedelta(days=1), periods=79, tz=timezone))
    index.append(today)
    data = pd.DataFrame({"Close": np.arange(100.0, 180.0)}, index=index)
    regular_period = {} if end_offset is None else {"end": now + pd.Timedelta(hours=end_offset)}
    metadata = {"currentTradingPeriod": {"regular": regular_period}}
    install_fake_ticker(monkeypatch, data, metadata=metadata)

    history = fetch_daily_history("AAPL", "1y")

    assert len(history) == expected_count
    assert (today.date() in history.index.date) == (expected_count == 80)


def test_fetch_daily_history_sanitizes_provider_failures(monkeypatch):
    install_fake_ticker(monkeypatch, error=RuntimeError("private request details"))

    with pytest.raises(ValueError, match="Yahoo Finance") as error:
        fetch_daily_history("AAPL", "1y")

    assert "private request details" not in str(error.value)
