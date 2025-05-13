from pydantic import BaseModel

class PredictionOut(BaseModel):
    user_rank: str
    proba: float | None = None