from pydantic import BaseModel, Field

class InputData(BaseModel):
    AGE: int = Field(..., ge=0)
    SEX_CD: int
    INVEST_TYPE: int
    RESOURCE_USED: int
    INVEST_RATIO: float
    CREDIT_SCORE: float
    DELINQUENT_COUNT: int
    DEBT_RATIO: float
    Q1_SCORE: int
    Q2_SCORE: int
    Q3_SCORE: int
    CONSUMPTION_SCORE: int
    FIN_KNOW_SCORE: int
    DIGITAL_FRIENDLY: int
