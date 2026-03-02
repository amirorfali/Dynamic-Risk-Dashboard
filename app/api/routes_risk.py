from fastapi import APIRouter

from app.api.schemas import RiskRequest, RiskResponse

router = APIRouter()


@router.post("/risk", response_model=RiskResponse)
def compute_risk(payload: RiskRequest) -> RiskResponse:
    # TODO: wire in actual risk computation once model inputs are finalized
    return RiskResponse(var=0.0, cvar=0.0)
