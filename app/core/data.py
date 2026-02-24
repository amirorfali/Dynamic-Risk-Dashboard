from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from app.utils.config import settings


@dataclass(frozen=True)
class CalibrationResult:
    mu: pd.Series
    sigma: pd.DataFrame
    returns: pd.DataFrame


def load_prices_csv(
    path: str | Path,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "close",
) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    for col in (date_col, ticker_col, price_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.sort_values([date_col, ticker_col])
    prices = df.pivot(index=date_col, columns=ticker_col, values=price_col)
    prices = prices.sort_index()
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Prices are empty")
    cleaned = prices.copy()
    cleaned = cleaned.ffill().dropna(how="any")
    if cleaned.empty:
        raise ValueError("Prices are empty after cleaning")
    return cleaned


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    cleaned = clean_prices(prices)
    if method == "log":
        returns = np.log(cleaned / cleaned.shift(1))
    elif method == "simple":
        returns = cleaned.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("Returns are empty")
    return returns


def _clip_eigenvalues(matrix: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, min_eigenvalue)
    return (vecs * vals) @ vecs.T


def ensure_psd(cov: pd.DataFrame, min_eigenvalue: float = 1e-8) -> pd.DataFrame:
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")
    sym = 0.5 * (cov.values + cov.values.T)
    psd = _clip_eigenvalues(sym, min_eigenvalue)
    return pd.DataFrame(psd, index=cov.index, columns=cov.columns)


def calibrate_mu_sigma(
    returns: pd.DataFrame,
    annualization_factor: int | float | None = None,
) -> CalibrationResult:
    if returns.empty:
        raise ValueError("Returns are empty")
    mu = returns.mean()
    sigma = returns.cov()
    sigma = ensure_psd(sigma)

    if annualization_factor is not None:
        mu = mu * annualization_factor
        sigma = sigma * annualization_factor

    return CalibrationResult(mu=mu, sigma=sigma, returns=returns)


def load_cached_prices(
    filename: str = "sample_prices.csv",
) -> pd.DataFrame:
    data_dir = Path(settings.data_dir)
    csv_path = data_dir / filename
    return load_prices_csv(csv_path)


def list_cached_files(extensions: Iterable[str] = (".csv",)) -> list[str]:
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix in extensions
    )


def fetch_stock_data(
    symbol: str,
    period: str = "1y",
    save_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    income_stmt = ticker.financials
    balance_sheet = ticker.balance_sheet

    if save_dir is not None:
        base_dir = Path(save_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(base_dir / f"{symbol}_prices.csv")
        income_stmt.to_csv(base_dir / f"{symbol}_income.csv")
        balance_sheet.to_csv(base_dir / f"{symbol}_balance.csv")

    return {
        "prices": hist,
        "income_statement": income_stmt,
        "balance_sheet": balance_sheet,
    }
