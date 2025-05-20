from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MODEL_PATH: str = "models/v1/rf_model.joblib"
    MODEL_PATH: str = "models/v3/centroid_weighted_model.joblib"
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
