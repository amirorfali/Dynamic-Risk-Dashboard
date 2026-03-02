from typing import Literal

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    portfolio: dict
    horizon_days: int
    return_model: Literal["normal", "normal_crash_mixture"] = "normal"
    tail_threshold: float | None = Field(
        default=None,
        description="Loss threshold ℓ; defaults to VaR at 99% if omitted.",
    )


class RiskResponse(BaseModel):
    var: float
    cvar: float
