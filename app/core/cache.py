from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.core.data import (
    calibrate_mu_sigma,
    compute_returns,
    fetch_prices_yfinance,
    load_cached_prices,
)
from app.utils.config import settings


@dataclass(frozen=True)
class CachedCalibration:
    mu: pd.Series
    sigma: pd.DataFrame
    cache_hit: bool
    data_source: str


def _cache_key(
    tickers: Iterable[str],
    window_days: int | None,
    horizon_days: int,
) -> str:
    payload = {
        "tickers": sorted(tickers),
        "window_days": window_days,
        "horizon_days": horizon_days,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_path(key: str) -> Path:
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"calibration_{key}.npz"


def _load_from_cache(path: Path) -> tuple[pd.Series, pd.DataFrame] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    tickers = data["tickers"].astype(str).tolist()
    mu = pd.Series(data["mu"], index=tickers)
    sigma = pd.DataFrame(data["sigma"], index=tickers, columns=tickers)
    return mu, sigma


def _save_to_cache(path: Path, mu: pd.Series, sigma: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        tickers=np.array(mu.index, dtype="U"),
        mu=mu.values,
        sigma=sigma.values,
    )


def get_calibration(
    tickers: Iterable[str],
    horizon_days: int,
    window_days: int | None = 252,
) -> CachedCalibration:
    tickers = sorted(tickers)
    key = _cache_key(tickers, window_days, horizon_days)
    path = _cache_path(key)
    cached = _load_from_cache(path)
    if cached is not None:
        mu, sigma = cached
        return CachedCalibration(
            mu=mu,
            sigma=sigma,
            cache_hit=True,
            data_source="cache",
        )

    period = f"{window_days}d" if window_days is not None else "1y"
    data_source = "yfinance"
    try:
        prices = fetch_prices_yfinance(tickers=tickers, period=period)
    except ValueError:
        prices = load_cached_prices()
        missing = [ticker for ticker in tickers if ticker not in prices.columns]
        if missing:
            raise
        prices = prices.loc[:, tickers]
        data_source = "sample_prices.csv"

    returns = compute_returns(prices)
    calibration = calibrate_mu_sigma(returns)
    _save_to_cache(path, calibration.mu, calibration.sigma)
    return CachedCalibration(
        mu=calibration.mu,
        sigma=calibration.sigma,
        cache_hit=False,
        data_source=data_source,
    )
