from fastapi import APIRouter

router = APIRouter()


@router.get("/experiments")
def list_experiments():
    return {"status": "ok", "message": "Experiments endpoint stub"}