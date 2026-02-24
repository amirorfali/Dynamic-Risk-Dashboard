from fastapi import APIRouter

router = APIRouter()


@router.post("/risk")
def compute_risk():
    return {"status": "ok", "message": "Risk endpoint stub"}