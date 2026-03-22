import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_experiments import (
    _base_distribution,
    compute_risk_metrics,
    discretize_samples,
    iqae_simulator_from_probs,
    _sample_losses,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public" / "experiments" / "animated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_svg_with_animation(fig, out_path: Path, stroke_hex: str, dur: str = "1.2s") -> None:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", dpi=200)
    svg = buf.getvalue()
    buf.close()

    # Find the first path with the desired stroke color.
    target = f"stroke:{stroke_hex}"
    idx = svg.find(target)
    if idx == -1:
        out_path.write_text(svg, encoding="utf-8")
        return

    # Find the start of the path tag containing the stroke.
    path_start = svg.rfind("<path", 0, idx)
    path_end = svg.find("/>", idx)
    if path_start == -1 or path_end == -1:
        out_path.write_text(svg, encoding="utf-8")
        return

    path_tag = svg[path_start:path_end + 2]
    if "stroke-dasharray" in path_tag:
        out_path.write_text(svg, encoding="utf-8")
        return

    animated_path = (
        path_tag[:-2]
        + f' stroke-dasharray="1000" stroke-dashoffset="1000">'
        + f'<animate attributeName="stroke-dashoffset" from="1000" to="0" dur="{dur}" fill="freeze" />'
        + "</path>"
    )

    svg = svg[:path_start] + animated_path + svg[path_end + 2:]
    out_path.write_text(svg, encoding="utf-8")


spec = _base_distribution(alpha=0.99)

# Classical error vs MC paths
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

fig = plt.figure(figsize=(6, 4))
plt.plot(n_paths_list, errors, marker="o", color="#2563eb")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("MC Paths (log)")
plt.ylabel("Abs VaR Error (log)")
plt.title("Classical Error vs MC Paths")
plt.grid(True, alpha=0.3)
plt.tight_layout()
_save_svg_with_animation(fig, OUT_DIR / "classical_error_vs_paths.svg", "#2563eb")
plt.close(fig)

# Quantum error vs oracle calls
samples = _sample_losses(seed=1)
disc = discretize_samples(samples, num_qubits=6, tail_threshold=spec.tail_threshold)
probs = np.asarray(disc.probabilities, dtype=float)
tail_mask = np.asarray(disc.tail_mask, dtype=bool)
true_tail = float(probs[tail_mask].sum())

max_iters = list(range(0, 9))
oracle_calls = []
q_errors = []
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
    q_errors.append(abs(result.estimate - true_tail))

fig = plt.figure(figsize=(6, 4))
plt.plot(oracle_calls, q_errors, marker="o", color="#16a34a")
plt.xlabel("Oracle Calls")
plt.ylabel("Abs Tail Prob Error")
plt.title("Quantum Error vs Oracle Calls")
plt.grid(True, alpha=0.3)
plt.tight_layout()
_save_svg_with_animation(fig, OUT_DIR / "quantum_error_vs_oracle_calls.svg", "#16a34a")
plt.close(fig)

# Error vs discretization bits
samples = _sample_losses(seed=2)
bits = list(range(3, 9))
errors = []
for b in bits:
    disc = discretize_samples(samples, num_qubits=b, tail_threshold=spec.tail_threshold)
    probs = np.asarray(disc.probabilities, dtype=float)
    tail_mask = np.asarray(disc.tail_mask, dtype=bool)
    tail_prob = float(probs[tail_mask].sum())
    errors.append(abs(tail_prob - spec.tail_prob))

fig = plt.figure(figsize=(6, 4))
plt.plot(bits, errors, marker="o", color="#f97316")
plt.xlabel("Discretization Bits (n)")
plt.ylabel("Abs Tail Prob Error")
plt.title("Error vs Discretization Bits")
plt.grid(True, alpha=0.3)
plt.tight_layout()
_save_svg_with_animation(fig, OUT_DIR / "error_vs_discretization_bits.svg", "#f97316")
plt.close(fig)

# Error vs noise level
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

fig = plt.figure(figsize=(6, 4))
plt.plot(noise_levels, errors, marker="o", color="#a855f7")
plt.xlabel("Noise Level (std on probs)")
plt.ylabel("Abs Tail Prob Error")
plt.title("Error vs Noise Level")
plt.grid(True, alpha=0.3)
plt.tight_layout()
_save_svg_with_animation(fig, OUT_DIR / "error_vs_noise_level.svg", "#a855f7")
plt.close(fig)

print(f"Wrote animated SVGs to {OUT_DIR}")
