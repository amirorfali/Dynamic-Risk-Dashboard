from time import perf_counter

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.api.schemas import RiskRequest, RiskResponse
from app.core.cache import get_calibration
from app.core.engine import compute_risk_metrics
from app.core.quantum_ae import iqae_simulator_from_probs
from app.core.scenarios import (
    apply_corr_spike,
    apply_crash_mixture,
    apply_mean_shock,
    apply_vol_multiplier,
)

router = APIRouter()


@router.post("/risk", response_model=RiskResponse)
def compute_risk(payload: RiskRequest) -> RiskResponse:
    if not payload.portfolio:
        raise HTTPException(status_code=400, detail="Portfolio must not be empty")

    tickers = list(payload.portfolio.keys())
    weights = np.array([payload.portfolio[t] for t in tickers], dtype=float)
    if np.any(~np.isfinite(weights)):
        raise HTTPException(status_code=400, detail="Portfolio weights must be finite")

    start = perf_counter()
    try:
        cached = get_calibration(
            tickers=tickers,
            horizon_days=payload.horizon_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mu = cached.mu
    sigma = cached.sigma
    weights = pd.Series(weights, index=tickers).reindex(mu.index).to_numpy()

    if payload.vol_multiplier != 1.0:
        shocked = apply_vol_multiplier(mu=mu, sigma=sigma, m=payload.vol_multiplier)
        mu = shocked.mu
        sigma = shocked.sigma

    if payload.corr_spike != 0.0:
        shocked = apply_corr_spike(mu=mu, sigma=sigma, alpha=payload.corr_spike)
        mu = shocked.mu
        sigma = shocked.sigma

    if payload.mean_shock != 0.0:
        shocked = apply_mean_shock(
            mu=mu,
            sigma=sigma,
            delta_mu=pd.Series(payload.mean_shock, index=mu.index, dtype=float),
        )
        mu = shocked.mu
        sigma = shocked.sigma

    model = payload.return_model
    crash_params = None
    if model == "normal_crash_mixture":
        crash_params = {
            "pc": payload.crash_pc,
            "mean_shift": payload.crash_mean_shift,
            "vol_jump": payload.crash_vol_jump,
        }
        crash_shift = pd.Series(crash_params["mean_shift"], index=mu.index)
        mixed = apply_crash_mixture(
            mu=mu,
            sigma=sigma,
            pc=crash_params["pc"],
            mean_shift=crash_shift,
            vol_jump=crash_params["vol_jump"],
        )
        mu = mixed.mu
        sigma = mixed.sigma

    histogram_bins = (
        2**payload.quantum_num_qubits if payload.backend == "quantum" else 30
    )

    metrics = compute_risk_metrics(
        mu=mu,
        sigma=sigma,
        weights=weights,
        horizon_days=payload.horizon_days,
        histogram_bins=histogram_bins,
        tail_threshold=payload.tail_threshold,
    )
    runtime_ms = (perf_counter() - start) * 1000.0
    tail_threshold = (
        metrics.var
        if payload.tail_threshold is None
        else float(payload.tail_threshold)
    )

    histogram = {
        "bin_edges": metrics.histogram.bin_edges,
        "counts": metrics.histogram.counts,
    }

    quantum_payload = None
    if payload.backend == "quantum":
        edges = np.asarray(metrics.histogram.bin_edges, dtype=float)
        counts = np.asarray(metrics.histogram.counts, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:]) if edges.size >= 2 else np.array([])
        total = float(counts.sum())
        tail_mask = (
            centers >= tail_threshold
            if centers.size
            else np.array([], dtype=bool)
        )
        tail_prob = float(counts[tail_mask].sum() / total) if total > 0 else None

        probs = counts / total if total > 0 else counts
        n_bins = int(probs.size)
        expected_bins = 2**payload.quantum_num_qubits
        if n_bins != expected_bins:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Quantum discretization mismatch: "
                    f"expected {expected_bins} bins but received {n_bins}"
                ),
            )

        iqae = None
        if probs.size > 0 and probs.sum() > 0:
            iqae = iqae_simulator_from_probs(
                probs=probs,
                tail_mask=tail_mask.astype(bool),
                shots=2000,
                max_iter=6,
                alpha=0.05,
                seed=11,
            )

        estimate = iqae.estimate if iqae else None
        ci_low = iqae.ci_low if iqae else None
        ci_high = iqae.ci_high if iqae else None
        diff_abs = (
            abs(estimate - tail_prob)
            if estimate is not None and tail_prob is not None
            else None
        )
        diff_rel = (
            (diff_abs / tail_prob)
            if diff_abs is not None and tail_prob not in (None, 0.0)
            else None
        )

        quantum_payload = {
            "tail_prob": tail_prob,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "diff_abs": diff_abs,
            "diff_rel": diff_rel,
            "bin_qubits": payload.quantum_num_qubits,
            "n_bins": n_bins,
            "padded_bins": n_bins,
            "shots": 2000,
            "max_iter": 6,
        }

    return RiskResponse(
        var=metrics.var,
        cvar=metrics.cvar,
        mean=metrics.mean,
        vol=metrics.vol,
        tail_threshold=tail_threshold,
        histogram=histogram,
        backend={
            "runtime_ms": runtime_ms,
            "n_paths": 5000,
            "model": model,
            "backend": payload.backend,
            "cache_hit": cached.cache_hit,
            "data_source": "yfinance",
            "window_days": 252,
            "crash_params": crash_params,
        },
        quantum=quantum_payload,
    )
