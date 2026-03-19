from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Histogram:
    bin_edges: list[float]
    counts: list[int]


@dataclass(frozen=True)
class RiskMetrics:
    var: float
    cvar: float
    mean: float
    vol: float
    histogram: Histogram


def _portfolio_losses(
    mu: pd.Series,
    sigma: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scaled_mu = mu.values * horizon_days
    scaled_sigma = sigma.values * horizon_days
    samples = rng.multivariate_normal(scaled_mu, scaled_sigma, size=n_paths)
    portfolio_returns = samples @ weights
    return -portfolio_returns


def compute_risk_metrics(
    mu: pd.Series,
    sigma: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int,
    n_paths: int = 5000,
    histogram_bins: int = 30,
    alpha: float = 0.99,
    tail_threshold: float | None = None,
    seed: int = 42,
) -> RiskMetrics:
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")

    losses = _portfolio_losses(
        mu=mu,
        sigma=sigma,
        weights=weights,
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
    )

    var = float(np.quantile(losses, alpha))
    threshold = var if tail_threshold is None else float(tail_threshold)
    tail_losses = losses[losses >= threshold]
    cvar = float(tail_losses.mean()) if tail_losses.size else var

    histogram = np.histogram(losses, bins=histogram_bins)
    hist = Histogram(
        bin_edges=histogram[1].astype(float).tolist(),
        counts=histogram[0].astype(int).tolist(),
    )

    return RiskMetrics(
        var=var,
        cvar=cvar,
        mean=float(losses.mean()),
        vol=float(losses.std(ddof=1)),
        histogram=hist,
    )
