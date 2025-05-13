from pydantic import BaseModel, Field

class InputData(BaseModel):
    age: int = Field(..., ge=0)
    essential_ratio: float
    discretionary_ratio: float
    complex_product_flag: int
    digital_engagement: int
    sav_inv_ratio: float
    spend_volatility: float
    risk_profile_score: float
