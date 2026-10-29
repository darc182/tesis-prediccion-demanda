from fastapi import FastAPI

from backend.app.api.datasets import router as datasets_router
from backend.app.api.pronosticos import router as pronosticos_router


app = FastAPI()
app.include_router(datasets_router)
app.include_router(pronosticos_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
