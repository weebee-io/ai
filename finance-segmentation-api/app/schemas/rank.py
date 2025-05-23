from pydantic import BaseModel
from typing import Optional

class PredictionOut(BaseModel):
    lit_level: str
    proba: Optional[float] = None