from typing import Literal

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    portfolio: dict
    horizon_days: int = Field(ge=1)
    return_model: Literal["normal", "normal_crash_mixture"] = "normal"
    backend: Literal["classical", "quantum"] = "classical"
    tail_threshold: float | None = Field(
        default=None,
        description="Loss threshold ℓ; defaults to VaR at 99% if omitted.",
    )
    vol_multiplier: float = Field(default=1.0, gt=0.0)
    corr_spike: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_shock: float = 0.0
    crash_pc: float = Field(default=0.05, ge=0.0, le=1.0)
    crash_mean_shift: float = 0.0
    crash_vol_jump: float = Field(default=2.0, gt=0.0)
    quantum_num_qubits: int = Field(default=5, ge=1, le=12)


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class BackendStats(BaseModel):
    runtime_ms: float
    n_paths: int
    model: str
    backend: str
    cache_hit: bool
    data_source: str
    window_days: int | None
    crash_params: dict | None = None


class QuantumEstimate(BaseModel):
    tail_prob: float | None
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    diff_abs: float | None
    diff_rel: float | None
    bin_qubits: int | None
    n_bins: int | None
    padded_bins: int | None
    shots: int | None
    max_iter: int | None


class RiskResponse(BaseModel):
    var: float
    cvar: float
    mean: float
    vol: float
    tail_threshold: float
    histogram: Histogram
    backend: BackendStats
    quantum: QuantumEstimate | None = None
