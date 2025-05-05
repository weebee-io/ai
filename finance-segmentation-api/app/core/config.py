from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/test/rf_model.joblib"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
