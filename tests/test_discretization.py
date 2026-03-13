import numpy as np
import pandas as pd

from app.core.classical_mc import simulate_nested_losses
from app.core.discretization import (
    discretize_nested_losses,
    discretize_samples,
    estimate_binning_error,
)


def test_discretize_samples_uses_power_of_two_bins():
    values = np.linspace(-1.0, 1.0, 16)

    result = discretize_samples(values=values, num_qubits=3)

    assert result.num_bins == 8
    assert len(result.counts) == 8
    assert len(result.probabilities) == 8
    assert result.basis_states[0] == "000"
    assert result.basis_states[-1] == "111"
    assert sum(result.counts) == 16
    assert np.isclose(sum(result.probabilities), 1.0)
    assert len(result.tail_mask) == 8


def test_discretize_nested_losses_matches_simulated_sample_count():
    mu = pd.Series([0.001, 0.002], index=["A", "B"])
    sigma = pd.DataFrame(
        [[0.01, 0.002], [0.002, 0.015]],
        index=mu.index,
        columns=mu.index,
    )
    weights = np.array([0.6, 0.4], dtype=float)

    losses = simulate_nested_losses(
        mu=mu,
        sigma=sigma,
        weights=weights,
        horizon_days=5,
        n_outer=4,
        n_inner=10,
        seed=7,
    )
    result = discretize_nested_losses(
        mu=mu,
        sigma=sigma,
        weights=weights,
        horizon_days=5,
        num_qubits=2,
        n_outer=4,
        n_inner=10,
        seed=7,
    )

    assert result.samples == losses.size
    assert sum(result.counts) == losses.size
    assert result.num_bins == 4
    assert np.isclose(sum(result.probabilities), 1.0)


def test_tail_mask_marks_right_tail():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = discretize_samples(
        values=values,
        num_qubits=2,
        tail_threshold=0.5,
        tail_side="right",
    )

    centers = np.asarray(result.bin_centers)
    mask = np.asarray(result.tail_mask)
    assert mask.dtype == bool
    assert np.all(mask == (centers >= 0.5))


def test_tail_mask_marks_left_tail():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = discretize_samples(
        values=values,
        num_qubits=2,
        tail_threshold=-0.5,
        tail_side="left",
    )

    centers = np.asarray(result.bin_centers)
    mask = np.asarray(result.tail_mask)
    assert np.all(mask == (centers <= -0.5))


def test_estimate_binning_error_returns_non_negative_stats():
    values = np.linspace(-1.0, 1.0, 100)
    stats = estimate_binning_error(values, num_qubits=3, repeats=5, sample_frac=0.7, seed=1)

    assert stats["mean_l1"] >= 0.0
    assert stats["std_l1"] >= 0.0
