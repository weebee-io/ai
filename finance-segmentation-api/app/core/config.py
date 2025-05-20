import os
from functools import lru_cache
from pydantic_settings import BaseSettings

# 프로젝트 루트 디렉토리 경로 계산 (config.py 파일이 위치한 디렉토리에서 두 단계 위)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    # 상대 경로 대신 절대 경로 사용
    MODEL_PATH: str = os.path.join(ROOT_DIR, "models", "v3", "centroid_weighted_model.joblib")
    
    # 디버깅용 설정 추가
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
