from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import market_data


def test_dashboard_starts_without_requesting_market_data():
    app_path = Path(__file__).parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    assert any("Run prediction" in item.value for item in app.info)


def test_dashboard_rejects_invalid_ticker_before_fetching_data():
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    app.text_input[0].set_value("bad ticker")
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert any("valid Yahoo Finance ticker" in item.value for item in app.error)


def test_dashboard_renders_forecast_and_refreshes_cached_history(monkeypatch):
    app_path = Path(__file__).parents[1] / "app.py"
    sessions = np.arange(180, dtype=float)
    prices = 100 + sessions * 0.2 + 8 * np.sin(sessions / 5)
    history = pd.DataFrame(
        {"Close": prices},
        index=pd.bdate_range("2024-01-01", periods=len(prices)),
    )
    fetch_calls = []

    def fake_fetch(symbol, period):
        fetch_calls.append((symbol, period))
        return history

    monkeypatch.setattr(market_data, "fetch_daily_history", fake_fetch)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.text_input[0].set_value("TEST.DEMO")
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert any("evaluation and forecast completed" in item.value for item in app.success)
    assert any("observations" in item.value for item in app.caption)
    assert fetch_calls == [("TEST.DEMO", "5y")]

    app.form_submit_button[1].click().run()

    assert not app.exception
    assert fetch_calls == [("TEST.DEMO", "5y"), ("TEST.DEMO", "5y")]
