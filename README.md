# Stock Price Predictor

A beginner-friendly Python machine-learning project that downloads historical stock prices, compares Linear Regression with Random Forest, evaluates both models on later dates, and forecasts the next trading session.

> **Disclaimer:** This is an educational forecasting demo, not a reliable trading signal. Stock prices are noisy, and past performance or model metrics do not guarantee future results.

## 🌐 Live Project Links

- **🚀 Live interactive app (Streamlit):** https://stock-price-predictor-jkqbqgjjy8emsxw4vkhtta.streamlit.app/
- **Project homepage (GitHub Pages):** https://deepnayan832.github.io/stock-price-predictor/
- **GitHub repository:** https://github.com/deepnayan832/stock-price-predictor
- **Source code:** [Browse files on GitHub](https://github.com/deepnayan832/stock-price-predictor/tree/main)

The Streamlit app runs online in your browser. You do not need to install Python or run the app locally to use the published version. GitHub Pages hosts the static project homepage; the interactive Python app is hosted separately on Streamlit Community Cloud.

## Features

- Historical adjusted closing prices from Yahoo Finance through `yfinance`.
- User-selectable ticker and historical lookback period.
- Lagged-price, rolling-average, rolling-standard-deviation, and momentum features.
- Linear Regression and Random Forest models.
- Chronological train/test evaluation without random shuffling.
- MAE, RMSE, R², and directional-accuracy comparison.
- Previous-close baseline for context.
- Next-session point forecast and actual-versus-predicted chart.
- Automated tests for feature construction and model workflow.

## How to use the live app

1. Open the **[Stock Price Predictor app](https://stock-price-predictor-jkqbqgjjy8emsxw4vkhtta.streamlit.app/)**.
2. Enter a supported Yahoo Finance ticker, for example:
   - `RELIANCE.NS` — Reliance Industries (NSE)
   - `TCS.NS` — Tata Consultancy Services (NSE)
   - `INFY.NS` — Infosys (NSE)
   - `AAPL` — Apple (US market)
3. Choose the historical period.
4. Select **Run prediction** and review the forecast, model comparison, historical chart, and test-set predictions.

Symbol availability follows Yahoo Finance conventions. This is not an official exchange feed.

## Run locally (optional)

You only need these steps if you want to develop or run the application on your own computer.

### Requirements

- Python 3.10 or newer recommended.
- Internet connection to retrieve market history.

### Install on Windows

Open PowerShell in the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Start the dashboard

```powershell
python -m streamlit run app.py
```

Streamlit will open the dashboard in your browser.

## Run tests

Install the development dependencies and run pytest:

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

## How the model works

For each target trading day, the feature row uses only previously available closing prices. Features include lags of 1, 2, 3, 5, 10, and 20 sessions; recent averages and standard deviations; and momentum over 5, 10, and 20 sessions. The target is the closing price on the next row/day.

The dataset is sorted chronologically and split without shuffling. Earlier observations train the models, while later observations form the holdout segment. The trainable model with the lower holdout Mean Absolute Error (MAE) is selected. Both models are also compared with a naive previous-close baseline. The chosen model is then retrained on the available history for the next-session forecast.

The holdout test is a rolling one-step-ahead evaluation: previous actual closes are assumed to be known for each test date. It does not demonstrate reliable multi-session forecasts or guarantee that the model can predict future market movements.

## Interpreting the metrics

- **MAE:** Average absolute prediction error in the ticker's quoted currency. Lower is better.
- **RMSE:** Penalizes larger errors more strongly. Lower is better.
- **R²:** Measures fit relative to a constant-mean baseline and can be negative.
- **Directional accuracy:** Percentage of test observations where the predicted up/down direction matches the actual movement relative to the previous close.

Metrics from one holdout period are not proof of profitability. A serious trading research system also needs walk-forward validation across multiple periods, transaction costs, slippage, realistic execution assumptions, and risk management.

## Data source and limitations

The project uses `yfinance`, an open-source interface to Yahoo Finance data. Data may be delayed, missing, adjusted, or temporarily unavailable. The app uses daily historical prices and should **not** be described as a guaranteed real-time exchange feed.

The model does not currently incorporate news, company fundamentals, macroeconomic variables, transaction costs, or all corporate events as predictive signals. Review Yahoo Finance terms and applicable local rules before using downloaded data beyond personal research.

## Project structure

```text
stock-price-predictor/
├── app.py
├── predictor.py
├── requirements.txt
├── requirements-dev.txt
├── tests/
│   └── test_predictor.py
├── .github/workflows/
├── index.html
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Continuous integration

GitHub Actions runs the unit tests on pushes, pull requests, and manual workflow dispatches for Python 3.11 and 3.12. Run the same tests locally with the command above.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
