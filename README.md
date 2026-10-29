# Stock Price Predictor

A beginner-friendly Python project that downloads daily historical stock prices, compares Linear Regression with Random Forest, evaluates the models on later dates, and forecasts the next trading session.

> **Important:** This is an educational baseline, not a reliable trading signal. Stock prices are noisy; past performance and model metrics do not guarantee future results.

## 🌐 Project Links

- **GitHub Repository:** https://github.com/deepnayan832/stock-price-predictor
- **GitHub Pages URL:** https://deepnayan832.github.io/stock-price-predictor/

> **Deployment note:** `index.html` is the static project homepage, and `.github/workflows/pages.yml` deploys it with GitHub Actions. One-time setup may be required: open **Settings → Pages → Build and deployment → Source**, select **GitHub Actions**, then open the Actions tab and check the **Deploy GitHub Pages** workflow. GitHub Pages cannot run the Python/Streamlit forecasting app itself; host that app separately on a Streamlit-compatible service.

## Features

- Historical adjusted close from Yahoo Finance using `yfinance`.
- Configurable ticker and lookback period in a Streamlit web UI.
- Lag, rolling-average, rolling-standard-deviation, and momentum features.
- Two models: Linear Regression and Random Forest.
- Chronological train/test evaluation (no random shuffle).
- MAE, RMSE, R², and directional accuracy comparison, including a previous-close baseline.
- Next-session point forecast and an actual-vs-predicted chart.
- Small unit tests for feature construction and model workflow.

## 1. Requirements

- Python 3.10 or newer recommended.
- Internet connection to download data.

## 2. Install on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and use the environment's executable directly:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

## 3. Run the dashboard

```powershell
python -m streamlit run app.py
```

Streamlit opens a local web app in your browser. Enter a Yahoo Finance ticker such as:

- `RELIANCE.NS` — Reliance Industries, NSE
- `TCS.NS` — TCS, NSE
- `INFY.NS` — Infosys, NSE
- `AAPL` — Apple, US market

Choose a historical period and select **Run prediction**. Ticker suffixes and availability depend on Yahoo Finance's symbol conventions; this is not an official exchange feed.

## 4. Run tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

## How the model works

For each target trading day, the feature row is constructed from previous known daily closes only. Features include lagged closes (1, 2, 3, 5, 10, and 20 sessions), recent averages and standard deviations, and momentum over 5, 10, and 20 sessions. The target is the close on the next row/day.

The data is sorted by date and split without shuffling: earlier observations train the models; later observations form a holdout set. The Linear Regression or Random Forest model with the lower holdout Mean Absolute Error (MAE) is selected, and both are compared with a naive previous-close baseline. The selected trainable model is then refit on all available observations to make the next-session forecast.

This holdout evaluation is a rolling one-step-ahead test: for each test date, the prior actual closes are assumed to be observable, which is realistic for a next-session forecast. It does not mean the model can accurately forecast several sessions into the future.

## Project structure

```text
stock-price-predictor/
├── app.py
├── predictor.py
├── requirements.txt
├── requirements-dev.txt
├── tests/
│   └── test_predictor.py
├── .github/workflows/tests.yml
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Interpreting metrics

- **MAE:** average absolute error, in the ticker's price currency. Lower is better.
- **RMSE:** gives greater weight to large errors. Lower is better.
- **R²:** relative fit score; it can be negative when predictions are worse than a constant-mean baseline.
- **Directional accuracy:** percentage of test dates where the model gets the up/down direction right relative to the previous close.

Compare against a simple baseline before claiming the model has predictive value. Avoid treating a single holdout period or a high R² as evidence of a profitable strategy. A serious trading system also needs walk-forward validation across multiple periods, transaction costs, slippage, survivorship-bias controls, and risk management.

## Data and limitations

The project uses `yfinance`, an open-source interface to Yahoo Finance data. It is intended for personal/research use; review Yahoo Finance terms and your local rules before broader use. Data can be missing, delayed, adjusted, or temporarily unavailable. The model excludes news, fundamentals, macroeconomic variables, dividends as a separate signal, and transaction costs.

## Continuous integration

GitHub Actions runs the unit tests on pushes, pull requests, and manual workflow dispatches for Python 3.11 and 3.12. Run the same tests locally with the command above.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
