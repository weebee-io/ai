from pydantic import BaseModel

class PredictionOut(BaseModel):
    cluster: int
    proba: float | None = None
