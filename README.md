# Stock Price Predictor

A beginner-friendly Python project that downloads daily historical stock prices, compares Linear Regression with Random Forest, evaluates the models on later dates, and forecasts the next trading session.

> **Important:** This is an educational baseline, not a reliable trading signal. Stock prices are noisy; past performance and model metrics do not guarantee future results.

## 🌐 Project Links

- **GitHub Repository:** https://github.com/deepnayan832/stock-price-predictor
- **Project homepage (GitHub Pages):** https://deepnayan832.github.io/stock-price-predictor/
- **Live Streamlit app:** https://deepnayan832-stock-price-predictor.streamlit.app/
- **Deploy the interactive online app:** [Open Streamlit Community Cloud deployment](https://share.streamlit.io/deploy?repository=deepnayan832%2Fstock-price-predictor&branch=main&mainModule=app.py)

> **Interactive deployment:** GitHub Pages serves the static project homepage only. The Python prediction app runs separately on Streamlit Community Cloud. Open the live app above to use it online, or use the one-click deployment link if you want to create your own deployment. GitHub Pages cannot run Python server-side code.

## Features

- Historical adjusted close from Yahoo Finance using `yfinance`.
- Configurable ticker and lookback period in a Streamlit web UI.
- Lag, rolling-average, rolling-standard-deviation, and momentum features.
- Two models: Linear Regression and Random Forest.
- Chronological train/test evaluation (no random shuffle).
- MAE, RMSE, R², and directional accuracy comparison, including a previous-close baseline.
- Next-session point forecast and an actual-vs-predicted chart.
- Small unit tests for feature construction and model workflow.

## Deploy the interactive app yourself

1. Open the [one-click Streamlit deployment link](https://share.streamlit.io/deploy?repository=deepnayan832%2Fstock-price-predictor&branch=main&mainModule=app.py) if you want to create your own deployment. The published app is available at [deepnayan832-stock-price-predictor.streamlit.app](https://deepnayan832-stock-price-predictor.streamlit.app/).
2. Sign in to Streamlit Community Cloud with GitHub and authorize access to this public repository.
3. Confirm repository `deepnayan832/stock-price-predictor`, branch `main`, and main file `app.py`.
4. Click **Deploy**. When the app finishes starting, copy the generated `.streamlit.app` URL and add it to the GitHub repository's **About → Website** field or to this README.
5. The app pulls the latest available daily history from Yahoo Finance when it runs. This is not a guaranteed real-time exchange feed, and it is not a profitable-trading guarantee.

Streamlit Community Cloud deployment is documented in the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

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
