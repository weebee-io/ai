import joblib
from functools import lru_cache
from sklearn.base import BaseEstimator
from app.core.config import get_settings

@lru_cache
def load_model() -> BaseEstimator:
    settings = get_settings()
    model: BaseEstimator = joblib.load(settings.MODEL_PATH)
    return model
