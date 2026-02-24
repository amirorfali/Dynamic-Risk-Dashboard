from pydantic import BaseModel


class RiskRequest(BaseModel):
    portfolio: dict
    horizon_days: int


class RiskResponse(BaseModel):
    var: float
    cvar: float