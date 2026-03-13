from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.core.discretization import discretize_samples
from app.core.quantum_ae import iqae_from_discretization, iqae_simulator_from_probs, toy_histogram


def main() -> None:
    values, probs, tail_mask = toy_histogram()
    toy = iqae_simulator_from_probs(probs=probs, tail_mask=tail_mask, shots=2000, max_iter=6, seed=11)
    print("toy true/est/ci:", toy.true_prob, toy.estimate, (toy.ci_low, toy.ci_high))

    disc = discretize_samples(values=np.linspace(-2.0, 2.0, 64), num_qubits=5, tail_threshold=1.0)
    res = iqae_from_discretization(disc, np.array(disc.tail_mask), shots=2000, max_iter=5, seed=9)
    print("disc true/est/ci:", res.true_prob, res.estimate, (res.ci_low, res.ci_high))
    print("resources:", res.resources)


if __name__ == "__main__":
    main()
