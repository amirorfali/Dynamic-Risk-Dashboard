from time import perf_counter

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.api.schemas import RiskRequest, RiskResponse
from app.core.cache import get_calibration
from app.core.engine import compute_risk_metrics
from app.core.scenarios import apply_crash_mixture

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

    model = payload.return_model
    crash_params = None
    if model == "normal_crash_mixture":
        crash_params = {
            "pc": 0.05,
            "mean_shift": -0.02,
            "vol_jump": 2.0,
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

    metrics = compute_risk_metrics(
        mu=mu,
        sigma=sigma,
        weights=weights,
        horizon_days=payload.horizon_days,
        tail_threshold=payload.tail_threshold,
    )
    runtime_ms = (perf_counter() - start) * 1000.0

    return RiskResponse(
        var=metrics.var,
        cvar=metrics.cvar,
        mean=metrics.mean,
        vol=metrics.vol,
        histogram={
            "bin_edges": metrics.histogram.bin_edges,
            "counts": metrics.histogram.counts,
        },
        backend={
            "runtime_ms": runtime_ms,
            "n_paths": 5000,
            "model": model,
            "cache_hit": cached.cache_hit,
            "data_source": "yfinance",
            "window_days": 252,
            "crash_params": crash_params,
        },
    )
