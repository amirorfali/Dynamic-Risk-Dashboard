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
- `./venv/bin/uvicorn app.main:app --reload`
- `http://127.0.0.1:8000/docs` docs provides the interactive Swagger UI for trying endpoints in the browser
- `http://127.0.0.1:8000/redoc` redoc provides the static ReDoc reference docs for the same API schema

Streamlit dashboard:
```bash
./venv/bin/streamlit run dashboard/streamlit_app.py
```
Press enter if it asks for email. It starts the Streamlit development server and opens the dashboard defined in `streamlit_app.py` in your browser.
Note: the stress sliders are now included in the API request and affect backend results.
When IQAE is selected, the dashboard also shows a qubit slider and uses
`2**qubits` histogram bins for the quantum discretization.

Backend toggle example (classical vs quantum):
```json
{
  "portfolio": { "AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2 },
  "horizon_days": 10,
  "return_model": "normal",
  "backend": "quantum",
  "tail_threshold": 0.02,
  "vol_multiplier": 1.2,
  "corr_spike": 0.25,
  "mean_shock": -0.01,
  "quantum_num_qubits": 5
}
```
Example response fields (truncated):
```json
{
  "var": 0.1095,
  "cvar": 0.1271,
  "tail_threshold": 0.02,
  "histogram": { "bin_edges": [0.0, 0.01], "counts": [12] },
  "backend": { "backend": "quantum", "runtime_ms": 10.8 },
  "quantum": {
    "tail_prob": 0.15,
    "estimate": 0.1490,
    "ci_low": 0.1475,
    "ci_high": 0.1504,
    "diff_abs": 0.0010,
    "diff_rel": 0.0067,
    "bin_qubits": 5,
    "n_bins": 32
  }
}
```

## Quick start (classical → discretization → IQAE)

```bash
./venv/bin/python - <<'PY'
import numpy as np
import pandas as pd

from app.core.classical_mc import simulate_nested_losses
from app.core.discretization import discretize_samples
from app.core.quantum_ae import iqae_from_discretization

mu = pd.Series([0.001, 0.002], index=["A", "B"])
sigma = pd.DataFrame([[0.01, 0.002], [0.002, 0.015]], index=mu.index, columns=mu.index)
weights = np.array([0.6, 0.4], dtype=float)

losses = simulate_nested_losses(mu, sigma, weights, horizon_days=5, n_outer=10, n_inner=200, seed=7)
disc = discretize_samples(values=losses, num_qubits=5, tail_threshold=0.02)
result = iqae_from_discretization(disc, np.array(disc.tail_mask), shots=2000, max_iter=6, seed=11)

print("tail true/est/ci:", result.true_prob, result.estimate, (result.ci_low, result.ci_high))
print("resources:", result.resources)
PY
```

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

## Discretization (bridge to quantum)

Features:
- Convert loss samples to `2**n` bins with edges + probabilities.
- Tail mask generation using a threshold or quantile.
- Optional binning error estimate via bootstrap resampling.
- Unit tests for probability sums and tail-mask correctness.

Main files:
- `app/core/discretization.py`
- `tests/test_discretization.py`

Manual test (discretization):
```bash
./venv/bin/python scripts/manual_test_discretization.py
```
Expected output (example):
```
edges: 9 probs: 1.0 tail: 3
binning_error: {'mean_l1': 0.28181818181818186, 'std_l1': 0.09588646868058631, 'repeats': 5.0}
```

## Quantum IQAE MVP (simulator)

Features:
- Toy histogram path to verify IQAE estimation.
- Discretized histogram path with tail mask + amplitude prep from probabilities.
- IQAE simulator returns estimate + confidence interval.
- Resource report (qubits, depth, 2Q gates, oracle calls, iterations).

Main files:
- `app/core/quantum_ae.py`
- `tests/test_quantum_ae.py`

Manual test (IQAE):
```bash
./venv/bin/python scripts/manual_test_iqae.py
```
Expected output (example):
```
toy true/est/ci: 0.15 0.14896479745088195 (0.14752600415773548, 0.15040359074402843)
disc true/est/ci: 0.25000000000000006 0.25066497747546435 (0.2484089898138266, 0.2529209651371021)
resources: ResourceReport(n_qubits=6, depth=228, two_qubit_gates=164, oracle_calls=7, iterations=7)
```

Grover iteration (intuition):
- State preparation `A` encodes the “good” (tail) probability `p`.
- Grover operator `Q` rotates amplitude toward good states.
- After `m` iterations, good probability is `sin^2((2m+1) * θ)` where `θ = arcsin(sqrt(p))`.
- IQAE probes multiple `m` values to infer `p` efficiently from measurements.

## Experiments (report/demo plots)

Generate all experiment plots (saved to `experiments/plots/`):
```bash
./venv/bin/python experiments/run_experiments.py
```
