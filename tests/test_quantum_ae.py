import numpy as np

from app.core.discretization import discretize_samples
from app.core.quantum_ae import (
    iqae_from_discretization,
    iqae_simulator_from_probs,
    toy_histogram,
)


def test_iqae_toy_histogram_estimate_close_to_true():
    values, probs, tail_mask = toy_histogram()

    result = iqae_simulator_from_probs(
        probs=probs,
        tail_mask=tail_mask,
        shots=3000,
        max_iter=6,
        alpha=0.05,
        seed=11,
    )

    assert result.true_prob is not None
    assert abs(result.estimate - result.true_prob) < 0.05
    assert 0.0 <= result.ci_low <= result.estimate <= result.ci_high <= 1.0


def test_iqae_from_discretization_returns_consistent_outputs():
    values = np.linspace(-2.0, 2.0, 64)
    disc = discretize_samples(values=values, num_qubits=5, tail_threshold=1.0)

    result = iqae_from_discretization(
        discretization=disc,
        tail_mask=np.array(disc.tail_mask),
        shots=2000,
        max_iter=5,
        seed=9,
    )

    assert result.true_prob is not None
    assert 0.0 <= result.estimate <= 1.0
    assert 0.0 <= result.ci_low <= result.ci_high <= 1.0
    assert result.resources.n_qubits == 6
