# Stock Price Predictor

A beginner-friendly Python project that downloads daily historical stock prices, compares Linear Regression with Random Forest, evaluates the models on later dates, and forecasts the next trading session.

> **Important:** This is an educational baseline, not a reliable trading signal. Stock prices are noisy; past performance and model metrics do not guarantee future results.

## 🌐 Project Links

- **Live interactive app (Streamlit):** https://stock-price-predictor-jkqbqgjjy8emsxw4vkhtta.streamlit.app/
- **GitHub Repository:** https://github.com/deepnayan832/stock-price-predictor
- **Project homepage (GitHub Pages):** https://deepnayan832.github.io/stock-price-predictor/
- **Deploy the interactive online app:** [Open Streamlit Community Cloud deployment](https://share.streamlit.io/deploy?repository=deepnayan832%2Fstock-price-predictor&branch=main&mainModule=app.py)

> **Interactive app:** Open the live Streamlit link to use the predictor in your browser. The one-click deployment link creates or updates an app from this repository, branch `main`, and main file `app.py`. GitHub Pages serves only the static homepage; it cannot run Python server-side code.

The Streamlit URL above was opened and used for a live ticker forecast. The GitHub Pages URL is a separate static landing page.

## Features

- Historical adjusted close from Yahoo Finance using `yfinance`.
- Configurable ticker and lookback period in a Streamlit web UI.
- Lag, rolling-average, rolling-standard-deviation, and momentum features.
- Two models: Linear Regression and Random Forest.
- Three expanding chronological evaluation windows (no random shuffle).
- MAE, RMSE, R², and directional accuracy comparison, including a previous-close baseline.
- Data date, retrieval time, one-hour cache expiry, and a manual refresh action.
- Current-session daily bars are excluded until the regular trading session ends.
- Next-session point forecast and an actual-vs-predicted chart.
- Small unit tests for feature construction and model workflow.

## Deploy or update the interactive app online

1. Open the [one-click Streamlit deployment link](https://share.streamlit.io/deploy?repository=deepnayan832%2Fstock-price-predictor&branch=main&mainModule=app.py).
2. Sign in to Streamlit Community Cloud with GitHub and authorize access to this public repository.
3. Confirm repository `deepnayan832/stock-price-predictor`, branch `main`, and main file `app.py`.
4. Select Python 3.12, click **Deploy**, and wait for the app to start.
5. Open the generated `.streamlit.app` URL and verify a real ticker fetch, evaluation table, and forecast. Update the app buttons in `index.html` to use the verified live app URL, and set the repository's **About → Website** field to that URL. Keep the GitHub Pages URL above as the separate static homepage.
6. The app retrieves completed daily adjusted history from Yahoo Finance via `yfinance`, caches it for up to one hour, and offers a manual refresh. It omits the current-session daily bar until the regular session ends. This is not a guaranteed real-time exchange feed or a profitable-trading guarantee.

Streamlit Community Cloud deployment is documented in the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## 1. Requirements

- Python 3.11 or 3.12; both are tested in GitHub Actions. Choose Python 3.12 in Streamlit Community Cloud.
- Internet connection to download data.

## 2. Install on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and use the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
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

The data is sorted by date and evaluated across three expanding windows without shuffling. The Linear Regression or Random Forest model with the lower aggregate out-of-window Mean Absolute Error (MAE) is selected, and both are compared with a naive previous-close baseline. The app reports the test dates and observation count. The selected trainable model is then refit on all available observations to make the next-session forecast.

This expanding walk-forward evaluation is a rolling one-step-ahead test: for each test date, the prior actual closes are assumed to be observable, which is realistic for a next-session forecast. It does not mean the model can accurately forecast several sessions into the future.

## Project structure

```text
stock-price-predictor/
├── app.py
├── market_data.py
├── predictor.py
├── requirements.txt
├── requirements-dev.txt
├── tests/
│   ├── test_app.py
│   ├── test_market_data.py
│   └── test_predictor.py
├── .github/workflows/
│   ├── pages.yml
│   └── tests.yml
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

Compare against a simple baseline before claiming the model has predictive value. A single walk-forward evaluation does not establish repeatable predictive value. A serious trading system also needs evaluation across market regimes, transaction costs, slippage, survivorship-bias controls, and risk management.

## Data and limitations

The project uses `yfinance`, an open-source interface to Yahoo Finance data. It is intended for personal/research use; review Yahoo Finance terms and your local rules before broader use. Data can be missing, delayed, adjusted, or temporarily unavailable. The model excludes news, fundamentals, macroeconomic variables, dividends as a separate signal, and transaction costs.

## Continuous integration

GitHub Actions runs the unit tests on pushes, pull requests, and manual workflow dispatches for Python 3.11 and 3.12. Run the same tests locally with the command above.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
