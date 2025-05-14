from pydantic import BaseModel, Field

class InputData(BaseModel):
    risk_profile_score: float
    complex_product_flag: float
    is_married: float
    essential_ratio: float
    discretionary_ratio: float
    sav_inv_ratio: float
    spend_volatility: float
    digital_engagement: float
    quiz_score: float