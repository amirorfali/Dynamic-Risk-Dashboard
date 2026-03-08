from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NestedRiskMetrics:
    """Container for portfolio loss risk outputs from nested Monte Carlo."""

    # Value-at-Risk at confidence level alpha.
    var: float
    # Conditional Value-at-Risk (expected tail loss above VaR).
    cvar: float
    # Mean of simulated portfolio losses.
    mean: float
    # Standard deviation of simulated portfolio losses.
    vol: float
    # Average of VaR values computed inside each outer scenario.
    conditional_var_mean: float
    # Average of CVaR values computed inside each outer scenario.
    conditional_cvar_mean: float


def _validate_inputs(
    mu: pd.Series,
    sigma: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int,
) -> None:
    """Validate dimensions and consistency of model inputs."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if mu.empty:
        raise ValueError("mu is empty")
    if sigma.empty or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a non-empty square matrix")
    if len(weights) != len(mu):
        raise ValueError("weights must have the same length as mu")
    if not mu.index.equals(sigma.index) or not sigma.index.equals(sigma.columns):
        raise ValueError("mu and sigma indices must align")


def _draw_outer_scenarios(
    rng: np.random.Generator,
    n_outer: int,
    n_assets: int,
    drift_uncertainty: float,
    vol_of_vol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample outer-layer regime variables for nested Monte Carlo.

    Returns:
        vol_multipliers: (n_outer,) volatility scaling factors for covariance.
        drift_shocks: (n_outer, n_assets) additive shocks to asset drifts.
    """
    # Keep E[multiplier] near 1 while allowing stochastic volatility.
    vol_multipliers = rng.lognormal(
        mean=-0.5 * vol_of_vol * vol_of_vol,
        sigma=vol_of_vol,
        size=n_outer,
    )
    drift_shocks = rng.normal(
        loc=0.0,
        scale=drift_uncertainty,
        size=(n_outer, n_assets),
    )
    return vol_multipliers, drift_shocks


def compute_nested_risk_metrics(
    mu: pd.Series,
    sigma: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int,
    #These will control how many sims we run
    n_outer: int = 500, 
    n_inner: int = 1000,
    #These are market conditions we should pay around with and perhaps let the user choose
    alpha: float = 0.99,
    drift_uncertainty: float = 0.002,
    vol_of_vol: float = 0.20,
    seed: int = 42,
) -> NestedRiskMetrics:
    """Run two-layer Monte Carlo to estimate portfolio loss risk metrics.

    Outer layer:
        Samples uncertainty in regime (drift and volatility level).
    Inner layer:
        Samples return paths conditional on each outer regime.
    """
    if n_outer <= 0 or n_inner <= 0:
        raise ValueError("n_outer and n_inner must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    _validate_inputs(mu=mu, sigma=sigma, weights=weights, horizon_days=horizon_days)

    rng = np.random.default_rng(seed)
    # Number of assets in the portfolio model.
    n_assets = len(mu)
    # Drift and covariance scaled from 1-day estimates to the requested horizon.
    scaled_mu = mu.values * horizon_days
    scaled_sigma = sigma.values * horizon_days

    vol_multipliers, drift_shocks = _draw_outer_scenarios(
        rng=rng,
        n_outer=n_outer,
        n_assets=n_assets,
        drift_uncertainty=drift_uncertainty,
        vol_of_vol=vol_of_vol,
    )

    all_losses: list[np.ndarray] = []
    # Scenario-level VaR/CVaR from each outer regime.
    conditional_var: list[float] = []
    conditional_cvar: list[float] = []

    for i in range(n_outer):
        # Regime-specific drift and covariance used by the inner simulation.
        mu_i = scaled_mu + drift_shocks[i] * horizon_days
        sigma_i = scaled_sigma * (vol_multipliers[i] ** 2)

        # Inner Monte Carlo returns and portfolio losses for this outer regime.
        inner_samples = rng.multivariate_normal(mu_i, sigma_i, size=n_inner)
        inner_returns = inner_samples @ weights
        losses_i = -inner_returns
        all_losses.append(losses_i)

        # Scenario-conditional tail metrics.
        var_i = float(np.quantile(losses_i, alpha))
        tail_i = losses_i[losses_i >= var_i]
        cvar_i = float(tail_i.mean()) if tail_i.size else var_i
        conditional_var.append(var_i)
        conditional_cvar.append(cvar_i)

    # Aggregate losses from all outer regimes for portfolio-level metrics.
    losses = np.concatenate(all_losses)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var

    return NestedRiskMetrics(
        var=var,
        cvar=cvar,
        mean=float(losses.mean()),
        vol=float(losses.std(ddof=1)),
        conditional_var_mean=float(np.mean(conditional_var)),
        conditional_cvar_mean=float(np.mean(conditional_cvar)),
    )
