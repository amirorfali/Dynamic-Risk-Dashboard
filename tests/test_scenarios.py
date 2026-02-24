import math

import numpy as np
import pandas as pd

from app.core.scenarios import apply_corr_spike, apply_vol_multiplier


def _norm_ppf(p: float) -> float:
    # Acklam approximation
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _portfolio_var(mu: pd.Series, sigma: pd.DataFrame, w: np.ndarray, alpha: float) -> float:
    z = _norm_ppf(alpha)
    port_var = float(w.T @ sigma.values @ w)
    port_mu = float(w.T @ mu.values)
    return max(0.0, -(port_mu + z * math.sqrt(port_var)))


def test_vol_multiplier_increases_var():
    mu = pd.Series([0.0, 0.0], index=["AAA", "BBB"])
    sigma = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=mu.index,
        columns=mu.index,
    )
    w = np.array([0.5, 0.5])

    base_var = _portfolio_var(mu, sigma, w, alpha=0.05)
    shocked = apply_vol_multiplier(mu, sigma, m=1.5)
    shocked_var = _portfolio_var(shocked.mu, shocked.sigma, w, alpha=0.05)

    assert shocked_var > base_var


def test_corr_spike_increases_var():
    mu = pd.Series([0.0, 0.0], index=["AAA", "BBB"])
    sigma = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=mu.index,
        columns=mu.index,
    )
    w = np.array([0.5, 0.5])

    base_var = _portfolio_var(mu, sigma, w, alpha=0.05)
    spiked = apply_corr_spike(mu, sigma, alpha=1.0, high_corr=0.9)
    spiked_var = _portfolio_var(spiked.mu, spiked.sigma, w, alpha=0.05)

    assert spiked_var > base_var
