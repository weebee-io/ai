from pydantic import BaseModel

class PredictionOut(BaseModel):
    lit_level: str
    proba: float | None = None