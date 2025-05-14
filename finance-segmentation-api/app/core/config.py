from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MODEL_PATH: str = "models/v1/rf_model.joblib"
    MODEL_PATH: str = "models/v2/stack_model_56.joblib"
    SCALER_PATH: str = "models/v2/scaler.joblib"  # Add scaler path

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
