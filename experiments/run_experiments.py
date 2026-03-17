from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR))

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    }
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.core.discretization import discretize_samples
from app.core.engine import compute_risk_metrics
from app.core.quantum_ae import iqae_simulator_from_probs


@dataclass(frozen=True)
class DistSpec:
    mu: float
    sigma: float
    alpha: float
    tail_threshold: float
    tail_prob: float


def _norm_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")

    # Acklam rational approximation (matches core IQAE helper).
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _base_distribution(alpha: float = 0.99) -> DistSpec:
    mu = 0.0
    sigma = 1.0
    z = _norm_ppf(alpha)
    tail_threshold = z
    tail_prob = 1.0 - alpha
    return DistSpec(mu=mu, sigma=sigma, alpha=alpha, tail_threshold=tail_threshold, tail_prob=tail_prob)


def classical_error_vs_paths(spec: DistSpec) -> None:
    mu = pd.Series([spec.mu], index=["asset"])
    sigma = pd.DataFrame([[spec.sigma ** 2]], index=["asset"], columns=["asset"])
    weights = np.array([1.0])

    n_paths_list = [200, 500, 1000, 2000, 5000, 10000, 20000]
    errors = []
    for n_paths in n_paths_list:
        trial_errors = []
        for seed in range(5):
            metrics = compute_risk_metrics(
                mu=mu,
                sigma=sigma,
                weights=weights,
                horizon_days=1,
                n_paths=n_paths,
                alpha=spec.alpha,
                seed=seed,
            )
            trial_errors.append(abs(metrics.var - spec.tail_threshold))
        errors.append(float(np.mean(trial_errors)))

    plt.figure(figsize=(6, 4))
    plt.plot(n_paths_list, errors, marker="o", color="#2563eb")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("# Paths (log)")
    plt.ylabel("Abs VaR Error (log)")
    plt.title("Classical Error vs # Paths")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "classical_error_vs_paths.png", dpi=200)
    plt.close()


def _sample_losses(seed: int = 0, n: int = 80_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=n)


def quantum_error_vs_oracle_calls(spec: DistSpec) -> None:
    samples = _sample_losses(seed=1)
    disc = discretize_samples(samples, num_qubits=6, tail_threshold=spec.tail_threshold)
    probs = np.asarray(disc.probabilities, dtype=float)
    tail_mask = np.asarray(disc.tail_mask, dtype=bool)
    true_tail = float(probs[tail_mask].sum())

    max_iters = list(range(0, 9))
    oracle_calls = []
    errors = []
    for max_iter in max_iters:
        result = iqae_simulator_from_probs(
            probs=probs,
            tail_mask=tail_mask,
            shots=2000,
            max_iter=max_iter,
            alpha=0.05,
            seed=11,
        )
        oracle_calls.append(result.resources.oracle_calls)
        errors.append(abs(result.estimate - true_tail))

    plt.figure(figsize=(6, 4))
    plt.plot(oracle_calls, errors, marker="o", color="#16a34a")
    plt.xlabel("Oracle Calls")
    plt.ylabel("Abs Tail Prob Error")
    plt.title("Quantum Error vs Oracle Calls")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "quantum_error_vs_oracle_calls.png", dpi=200)
    plt.close()


def error_vs_discretization_bits(spec: DistSpec) -> None:
    samples = _sample_losses(seed=2)
    bits = list(range(3, 9))
    errors = []
    for b in bits:
        disc = discretize_samples(samples, num_qubits=b, tail_threshold=spec.tail_threshold)
        probs = np.asarray(disc.probabilities, dtype=float)
        tail_mask = np.asarray(disc.tail_mask, dtype=bool)
        tail_prob = float(probs[tail_mask].sum())
        errors.append(abs(tail_prob - spec.tail_prob))

    plt.figure(figsize=(6, 4))
    plt.plot(bits, errors, marker="o", color="#f97316")
    plt.xlabel("Discretization Bits (n)")
    plt.ylabel("Abs Tail Prob Error")
    plt.title("Error vs Discretization Bits")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_vs_discretization_bits.png", dpi=200)
    plt.close()


def error_vs_noise_level(spec: DistSpec) -> None:
    samples = _sample_losses(seed=3)
    disc = discretize_samples(samples, num_qubits=6, tail_threshold=spec.tail_threshold)
    base_probs = np.asarray(disc.probabilities, dtype=float)
    tail_mask = np.asarray(disc.tail_mask, dtype=bool)
    true_tail = float(base_probs[tail_mask].sum())

    rng = np.random.default_rng(4)
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    errors = []
    for level in noise_levels:
        trial_errors = []
        for _ in range(15):
            noise = rng.normal(0.0, level, size=base_probs.shape)
            noisy = np.clip(base_probs + noise, 0.0, None)
            total = noisy.sum()
            if total <= 0:
                noisy = base_probs.copy()
            else:
                noisy = noisy / total
            result = iqae_simulator_from_probs(
                probs=noisy,
                tail_mask=tail_mask,
                shots=1500,
                max_iter=6,
                alpha=0.05,
                seed=int(rng.integers(0, 10_000)),
            )
            trial_errors.append(abs(result.estimate - true_tail))
        errors.append(float(np.mean(trial_errors)))

    plt.figure(figsize=(6, 4))
    plt.plot(noise_levels, errors, marker="o", color="#a855f7")
    plt.xlabel("Noise Level (std on probs)")
    plt.ylabel("Abs Tail Prob Error")
    plt.title("Error vs Noise Level")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_vs_noise_level.png", dpi=200)
    plt.close()


def feasibility_boundary(spec: DistSpec) -> None:
    samples = _sample_losses(seed=5)

    bits = list(range(3, 10))
    max_iters = list(range(0, 9))

    status = np.zeros((len(bits), len(max_iters)), dtype=int)

    # Heuristic thresholds for feasibility bands (composite score).
    green_limits = {"qubits": 7, "depth": 180, "oracle": 12}
    yellow_multiplier = 1.25

    for i, b in enumerate(bits):
        disc = discretize_samples(samples, num_qubits=b, tail_threshold=spec.tail_threshold)
        probs = np.asarray(disc.probabilities, dtype=float)
        tail_mask = np.asarray(disc.tail_mask, dtype=bool)
        for j, max_iter in enumerate(max_iters):
            result = iqae_simulator_from_probs(
                probs=probs,
                tail_mask=tail_mask,
                shots=500,
                max_iter=max_iter,
                alpha=0.05,
                seed=13,
            )
            r = result.resources
            score = (
                r.n_qubits / green_limits["qubits"]
                + r.depth / green_limits["depth"]
                + r.oracle_calls / green_limits["oracle"]
            ) / 3.0
            if score <= 1.0:
                status[i, j] = 0
            elif score <= yellow_multiplier:
                status[i, j] = 1
            else:
                status[i, j] = 2

    cmap = plt.cm.get_cmap("RdYlGn_r", 3)
    plt.figure(figsize=(7, 4.5))
    im = plt.imshow(status, cmap=cmap, aspect="auto", origin="lower")
    plt.colorbar(im, ticks=[0, 1, 2], label="Feasibility")
    plt.yticks(range(len(bits)), bits)
    plt.xticks(range(len(max_iters)), max_iters)
    plt.xlabel("Max Grover Iterations")
    plt.ylabel("Discretization Bits (n)")
    plt.title("Feasibility Boundary (Green/Yellow/Red)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feasibility_boundary.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    spec = _base_distribution(alpha=0.99)
    classical_error_vs_paths(spec)
    quantum_error_vs_oracle_calls(spec)
    error_vs_discretization_bits(spec)
    error_vs_noise_level(spec)
    feasibility_boundary(spec)
    print(f"Saved plots to {PLOTS_DIR}")
