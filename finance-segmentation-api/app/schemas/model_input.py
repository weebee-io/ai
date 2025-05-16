from pydantic import BaseModel

class InputData(BaseModel):
    risk_profile_score: int
    complex_product_flag: int
    is_married: int
    essential_pct: int
    discretionary_pct: int
    sav_inv_ratio: int
    spend_volatility: int
    digital_engagement: int