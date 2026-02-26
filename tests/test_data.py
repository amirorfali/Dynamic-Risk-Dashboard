import numpy as np
import pandas as pd

# Core data pipeline utilities under test.
from app.core.data import calibrate_mu_sigma, compute_returns, ensure_psd, load_prices_csv


def _write_prices_csv(path):
    # Create a tiny, deterministic price dataset in long format.
    rows = [
        ["date", "ticker", "close"],
        ["2024-01-02", "AAA", "100.0"],
        ["2024-01-02", "BBB", "200.0"],
        ["2024-01-03", "AAA", "101.0"],
        ["2024-01-03", "BBB", "198.0"],
        ["2024-01-04", "AAA", "100.5"],
        ["2024-01-04", "BBB", "202.0"],
    ]
    path.write_text("\n".join([",".join(r) for r in rows]))


def test_load_prices_csv_shapes(tmp_path):
    # Verify that a long-format CSV is pivoted into a (dates x tickers) matrix.
    csv_path = tmp_path / "prices.csv"
    _write_prices_csv(csv_path)
    prices = load_prices_csv(csv_path)
    assert prices.shape == (3, 2)
    assert list(prices.columns) == ["AAA", "BBB"]


def test_returns_and_calibration_shapes(tmp_path):
    # Check that returns, mu, and sigma have expected shapes after calibration.
    csv_path = tmp_path / "prices.csv"
    _write_prices_csv(csv_path)
    prices = load_prices_csv(csv_path)
    returns = compute_returns(prices, method="log")
    result = calibrate_mu_sigma(returns)
    assert result.mu.shape == (2,)
    assert result.sigma.shape == (2, 2)
    assert result.returns.shape[1] == 2


def test_covariance_psd_fix():
    # Build a slightly non-PSD covariance matrix, apply the fix,
    # and assert the eigenvalues are non-negative (within tolerance).
    cov = pd.DataFrame(
        [[1.0, 1.01], [1.01, 1.0]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )
    fixed = ensure_psd(cov, min_eigenvalue=1e-8)
    eigvals = np.linalg.eigvalsh(fixed.values)
    assert eigvals.min() >= -1e-9
