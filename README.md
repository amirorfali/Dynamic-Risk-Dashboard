# Dynamic Risk Dashboard

FastAPI + Streamlit system for classical and quantum risk analysis.

API contract:
- `docs/api_contract.md`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

FastAPI API server:
- `uvicorn app.main:app --reload`
- `http://127.0.0.1:8000/docs` docs provides the interactive Swagger UI for trying endpoints in the browser
- `http://127.0.0.1:8000/redoc` redoc provides the static ReDoc reference docs for the same API schema

Streamlit dashboard:
```bash
streamlit run dashboard/streamlit_app.py
```
Press enter if it asks for email. It starts the Streamlit development server and opens the dashboard defined in `streamlit_app.py` in your browser.

Tests:
```bash
./venv/bin/pytest -q
./venv/bin/ruff check .
./venv/bin/black --check .
```
What they do:
- `pytest`: runs unit tests covering the data pipeline and scenario engine behavior.
- `ruff check .`: static linting for errors, style, and import order.
- `black --check .`: verifies formatting matches the project style.

Makefile shortcuts:
```bash
make setup
make test
make lint
make fmt
make fmt-check
make run-api
make run-dashboard
make dev
```
What they do:
- `make setup`: creates the virtual environment and installs dependencies.
- `make test`: runs the test suite with pytest.
- `make lint`: runs ruff for linting and import order checks.
- `make fmt`: formats code with black.
- `make fmt-check`: verifies formatting without changing files.
- `make run-api`: starts the FastAPI server (uvicorn).
- `make run-dashboard`: starts the Streamlit dashboard.
- `make dev`: starts API + dashboard together via `scripts/dev.sh`.

## Data + calibration

Features:
- CSV price ingestion and cleaning (`load_prices_csv`, `clean_prices`).
- Return computation (`compute_returns`) with log or simple returns.
- Calibration of mean vector and covariance (`calibrate_mu_sigma`).
- PSD covariance fix for near-singular matrices (`ensure_psd`).
- Cached data helper and sample data (`load_cached_prices`, `data/sample_prices.csv`).
- Unit tests for shapes and PSD behavior (`tests/test_data.py`).

Main files:
- `app/core/data.py`
- `data/sample_prices.csv`
- `tests/test_data.py`

## Stress scenario engine

Features:
- Volatility multiplier on covariance (`apply_vol_multiplier`).
- Correlation spike transform (`apply_corr_spike`).
- Mean shock (`apply_mean_shock`).
- Crash mixture scenario with probability, mean shift, vol jump (`apply_crash_mixture`).
- Sanity tests showing `vol↑ ⇒ VaR↑` and `corr↑ ⇒ VaR↑` (`tests/test_scenarios.py`).

Main files:
- `app/core/scenarios.py`
- `tests/test_scenarios.py`
