from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.classical_mc import simulate_nested_losses


@dataclass(frozen=True)
class QubitDiscretization:
    """Discrete approximation of a continuous distribution using 2**n bins."""

    num_qubits: int
    num_bins: int
    bin_edges: list[float]
    bin_centers: list[float]
    counts: list[int]
    probabilities: list[float]
    basis_states: list[str]
    samples: int


def _resolve_bin_range(
    values: np.ndarray,
    value_range: tuple[float, float] | None,
) -> tuple[float, float]:
    if value_range is None:
        lower = float(np.min(values))
        upper = float(np.max(values))
    else:
        lower = float(value_range[0])
        upper = float(value_range[1])

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bin range must be finite")
    if upper < lower:
        raise ValueError("bin range upper bound must be >= lower bound")
    if upper == lower:
        padding = max(abs(lower) * 1e-9, 1e-9)
        lower -= padding
        upper += padding

    return lower, upper


def discretize_samples(
    values: np.ndarray,
    num_qubits: int,
    value_range: tuple[float, float] | None = None,
) -> QubitDiscretization:
    """Bin continuous samples into 2**num_qubits intervals.

    This is useful for mapping a classical distribution onto quantum basis states.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    samples = np.asarray(values, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("values must not be empty")
    if np.any(~np.isfinite(samples)):
        raise ValueError("values must be finite")

    num_bins = 2**num_qubits
    lower, upper = _resolve_bin_range(samples, value_range)
    counts, bin_edges = np.histogram(samples, bins=num_bins, range=(lower, upper))
    total = counts.sum()
    probabilities = counts / total if total else np.zeros(num_bins, dtype=float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    basis_states = [format(i, f"0{num_qubits}b") for i in range(num_bins)]

    return QubitDiscretization(
        num_qubits=num_qubits,
        num_bins=num_bins,
        bin_edges=bin_edges.astype(float).tolist(),
        bin_centers=bin_centers.astype(float).tolist(),
        counts=counts.astype(int).tolist(),
        probabilities=probabilities.astype(float).tolist(),
        basis_states=basis_states,
        samples=int(samples.size),
    )


def discretize_nested_losses(
    mu: pd.Series,
    sigma: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int,
    num_qubits: int,
    n_outer: int = 500,
    n_inner: int = 1000,
    drift_uncertainty: float = 0.002,
    vol_of_vol: float = 0.20,
    seed: int = 42,
    value_range: tuple[float, float] | None = None,
) -> QubitDiscretization:
    """Run nested Monte Carlo and discretize the resulting loss distribution."""
    losses = simulate_nested_losses(
        mu=mu,
        sigma=sigma,
        weights=weights,
        horizon_days=horizon_days,
        n_outer=n_outer,
        n_inner=n_inner,
        drift_uncertainty=drift_uncertainty,
        vol_of_vol=vol_of_vol,
        seed=seed,
    )
    return discretize_samples(
        values=losses,
        num_qubits=num_qubits,
        value_range=value_range,
    )
