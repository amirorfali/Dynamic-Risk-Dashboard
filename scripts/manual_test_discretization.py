from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.core.discretization import discretize_samples, estimate_binning_error


def main() -> None:
    values = np.linspace(-2.0, 2.0, 64)
    disc = discretize_samples(values=values, num_qubits=3, tail_threshold=0.5)
    print("edges:", len(disc.bin_edges), "probs:", sum(disc.probabilities), "tail:", sum(disc.tail_mask))

    stats = estimate_binning_error(values, num_qubits=3, repeats=5, sample_frac=0.7, seed=1)
    print("binning_error:", stats)


if __name__ == "__main__":
    main()
