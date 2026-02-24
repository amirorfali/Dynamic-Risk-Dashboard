from fastapi import FastAPI

from app.api.routes_risk import router as risk_router
from app.api.routes_experiments import router as exp_router


def create_app() -> FastAPI:
    app = FastAPI(title="Dynamic Risk Dashboard")

    app.include_router(risk_router, prefix="/api")
    app.include_router(exp_router, prefix="/api")

    return app


app = create_app()