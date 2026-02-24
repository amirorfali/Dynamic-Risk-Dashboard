from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    mu: pd.Series
    sigma: pd.DataFrame


def apply_vol_multiplier(mu: pd.Series, sigma: pd.DataFrame, m: float) -> ScenarioResult:
    if m <= 0:
        raise ValueError("Vol multiplier must be positive")
    scaled = sigma * (m**2)
    return ScenarioResult(mu=mu.copy(), sigma=scaled)


def apply_mean_shock(mu: pd.Series, sigma: pd.DataFrame, delta_mu: pd.Series) -> ScenarioResult:
    if not mu.index.equals(delta_mu.index):
        raise ValueError("delta_mu must share the same index as mu")
    shocked = mu + delta_mu
    return ScenarioResult(mu=shocked, sigma=sigma.copy())


def apply_corr_spike(
    mu: pd.Series,
    sigma: pd.DataFrame,
    alpha: float = 0.5,
    high_corr: float = 0.9,
) -> ScenarioResult:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if not (-1.0 < high_corr < 1.0):
        raise ValueError("high_corr must be in (-1, 1)")

    std = np.sqrt(np.diag(sigma.values))
    if np.any(std <= 0):
        raise ValueError("Sigma must have positive diagonal entries")

    corr = sigma.values / np.outer(std, std)
    corr = 0.5 * (corr + corr.T)
    target = np.full_like(corr, high_corr)
    np.fill_diagonal(target, 1.0)

    blended = (1.0 - alpha) * corr + alpha * target
    blended = 0.5 * (blended + blended.T)
    np.fill_diagonal(blended, 1.0)

    spiked = blended * np.outer(std, std)
    return ScenarioResult(mu=mu.copy(), sigma=pd.DataFrame(spiked, index=sigma.index, columns=sigma.columns))


def apply_crash_mixture(
    mu: pd.Series,
    sigma: pd.DataFrame,
    pc: float,
    mean_shift: pd.Series,
    vol_jump: float,
) -> ScenarioResult:
    if not (0.0 <= pc <= 1.0):
        raise ValueError("pc must be in [0, 1]")
    if vol_jump <= 0:
        raise ValueError("vol_jump must be positive")
    if not mu.index.equals(mean_shift.index):
        raise ValueError("mean_shift must share the same index as mu")

    base_mu = mu
    crash_mu = mu + mean_shift

    mixed_mu = (1.0 - pc) * base_mu + pc * crash_mu

    base_sigma = sigma.values
    crash_sigma = sigma.values * (vol_jump**2)

    mean_diff = (crash_mu - base_mu).values.reshape(-1, 1)
    mixture_extra = pc * (1.0 - pc) * (mean_diff @ mean_diff.T)

    mixed_sigma = (1.0 - pc) * base_sigma + pc * crash_sigma + mixture_extra

    return ScenarioResult(
        mu=mixed_mu,
        sigma=pd.DataFrame(mixed_sigma, index=sigma.index, columns=sigma.columns),
    )
