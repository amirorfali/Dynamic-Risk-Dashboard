from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Iterable

import numpy as np

from app.core.discretization import QubitDiscretization


@dataclass(frozen=True)
class ResourceReport:
    """Rough resource accounting for the IQAE simulator."""

    n_qubits: int
    depth: int
    two_qubit_gates: int
    oracle_calls: int
    iterations: int


@dataclass(frozen=True)
class IQAEResult:
    estimate: float
    ci_low: float
    ci_high: float
    true_prob: float | None
    shots_per_round: int
    m_schedule: list[int]
    success_counts: list[int]
    trials: list[int]
    resources: ResourceReport


def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF of standard normal.

    Uses a rational approximation (Acklam). Accurate to ~1e-9.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")

    # Coefficients in rational approximations.
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
        q = sqrt(-2 * np.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
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


def _amplitude_from_probs(probs: np.ndarray) -> np.ndarray:
    """Return amplitude vector (sqrt of probabilities)."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be 1-D")
    if probs.size == 0:
        raise ValueError("probs must not be empty")
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("probs must sum to a positive value")
    probs = probs / total
    return np.sqrt(probs)


def tail_mask_from_threshold(
    bin_centers: Iterable[float],
    threshold: float,
    side: str = "right",
) -> np.ndarray:
    """Return boolean mask for tail bins based on a threshold."""
    centers = np.asarray(list(bin_centers), dtype=float)
    if side == "right":
        return centers >= threshold
    if side == "left":
        return centers <= threshold
    raise ValueError("side must be 'right' or 'left'")


def _default_m_schedule(max_iter: int) -> list[int]:
    """Powers of two for Grover iterations, bounded by max_iter."""
    schedule: list[int] = []
    m = 0
    power = 1
    while m <= max_iter:
        schedule.append(m)
        m = power
        power *= 2
    return schedule


def _log_likelihood(theta: np.ndarray, m: np.ndarray, counts: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """Log-likelihood for observing counts given Grover powers m."""
    probs = np.sin((2 * m + 1)[:, None] * theta[None, :]) ** 2
    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)
    return (counts[:, None] * np.log(probs) + (shots[:, None] - counts[:, None]) * np.log(1 - probs)).sum(axis=0)


def _mle_with_ci(
    m_schedule: list[int],
    counts: list[int],
    shots: list[int],
    alpha: float,
    grid_size: int = 4096,
) -> tuple[float, float, float]:
    """Return MLE for p and an approximate CI using curvature of log-likelihood."""
    m = np.asarray(m_schedule, dtype=int)
    c = np.asarray(counts, dtype=float)
    n = np.asarray(shots, dtype=float)

    theta_grid = np.linspace(1e-6, np.pi / 2 - 1e-6, grid_size)
    ll = _log_likelihood(theta_grid, m, c, n)
    idx = int(np.argmax(ll))
    theta_hat = theta_grid[idx]

    # Second-derivative approximation for CI.
    h = theta_grid[1] - theta_grid[0]
    if 0 < idx < len(theta_grid) - 1:
        ll_left = ll[idx - 1]
        ll_right = ll[idx + 1]
        ll_center = ll[idx]
        second = (ll_left - 2 * ll_center + ll_right) / (h * h)
    else:
        second = -1.0

    if second >= 0:
        second = -1.0

    var_theta = -1.0 / second
    dp_dtheta = np.sin(2 * theta_hat)
    var_p = (dp_dtheta * dp_dtheta) * var_theta

    z = _norm_ppf(1 - alpha / 2)
    se_p = sqrt(max(var_p, 0.0))
    p_hat = float(np.sin(theta_hat) ** 2)
    ci_low = max(0.0, p_hat - z * se_p)
    ci_high = min(1.0, p_hat + z * se_p)
    return p_hat, ci_low, ci_high


def iqae_simulator_from_probs(
    probs: np.ndarray,
    tail_mask: np.ndarray,
    *,
    shots: int = 1000,
    max_iter: int = 8,
    alpha: float = 0.05,
    seed: int = 7,
    m_schedule: list[int] | None = None,
    prep_depth: int = 8,
    prep_2q: int = 6,
    oracle_depth: int = 6,
    oracle_2q: int = 4,
    diffusion_depth: int = 6,
    diffusion_2q: int = 4,
) -> IQAEResult:
    """Simulate IQAE on a probability distribution.

    The simulator uses the exact Grover-amplified probabilities to generate
    Bernoulli samples, then performs MLE over theta to estimate p.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs must be 1-D")
    if probs.size == 0:
        raise ValueError("probs must not be empty")
    if tail_mask.shape != probs.shape:
        raise ValueError("tail_mask must match probs shape")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if max_iter < 0:
        raise ValueError("max_iter must be >= 0")

    amplitudes = _amplitude_from_probs(probs)
    tail_prob = float((amplitudes[tail_mask] ** 2).sum())

    theta = np.arcsin(sqrt(tail_prob))
    rng = np.random.default_rng(seed)

    if m_schedule is None:
        m_schedule = _default_m_schedule(max_iter)

    counts: list[int] = []
    trials: list[int] = []
    for m in m_schedule:
        prob_m = float(np.sin((2 * m + 1) * theta) ** 2)
        counts.append(int(rng.binomial(shots, prob_m)))
        trials.append(int(shots))

    estimate, ci_low, ci_high = _mle_with_ci(m_schedule, counts, trials, alpha)

    n_bins = probs.size
    n_bins_bits = int(np.log2(n_bins))
    if 2**n_bins_bits != n_bins:
        raise ValueError("probs length must be a power of two")
    n_qubits = n_bins_bits + 1  # bin register + 1 ancilla for good/bad

    q_depth = oracle_depth + diffusion_depth + 2 * prep_depth
    q_2q = oracle_2q + diffusion_2q + 2 * prep_2q

    total_depth = 0
    total_2q = 0
    total_oracle_calls = 0
    total_iterations = 0
    for m in m_schedule:
        total_depth += prep_depth + m * q_depth
        total_2q += prep_2q + m * q_2q
        total_oracle_calls += m
        total_iterations += m

    resources = ResourceReport(
        n_qubits=n_qubits,
        depth=int(total_depth),
        two_qubit_gates=int(total_2q),
        oracle_calls=int(total_oracle_calls),
        iterations=int(total_iterations),
    )

    return IQAEResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        true_prob=tail_prob,
        shots_per_round=shots,
        m_schedule=m_schedule,
        success_counts=counts,
        trials=trials,
        resources=resources,
    )


def iqae_from_discretization(
    discretization: QubitDiscretization,
    tail_mask: np.ndarray,
    *,
    shots: int = 1000,
    max_iter: int = 8,
    alpha: float = 0.05,
    seed: int = 7,
) -> IQAEResult:
    """Run the IQAE simulator using a discretized histogram."""
    probs = np.asarray(discretization.probabilities, dtype=float)
    return iqae_simulator_from_probs(
        probs=probs,
        tail_mask=tail_mask,
        shots=shots,
        max_iter=max_iter,
        alpha=alpha,
        seed=seed,
    )


def toy_histogram() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values, probs, tail_mask) for a tiny toy distribution."""
    values = np.linspace(-2.0, 2.0, 8)
    probs = np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05])
    tail_mask = values >= 1.0
    return values, probs, tail_mask
