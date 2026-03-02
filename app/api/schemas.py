from typing import Literal

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    portfolio: dict
    horizon_days: int = Field(ge=1)
    return_model: Literal["normal", "normal_crash_mixture"] = "normal"
    tail_threshold: float | None = Field(
        default=None,
        description="Loss threshold ℓ; defaults to VaR at 99% if omitted.",
    )


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class BackendStats(BaseModel):
    runtime_ms: float
    n_paths: int
    model: str
    cache_hit: bool
    data_source: str
    window_days: int | None
    crash_params: dict | None = None


class RiskResponse(BaseModel):
    var: float
    cvar: float
    mean: float
    vol: float
    histogram: Histogram
    backend: BackendStats
