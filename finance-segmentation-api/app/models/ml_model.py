import joblib
import numpy as np
from functools import lru_cache
from typing import Tuple, Any, Optional
from app.core.config import get_settings
from app.models.centroid_predictor import CentroidPredictor

@lru_cache
def load_model_components() -> Tuple[Optional[CentroidPredictor], Optional[Any]]:
    try:
        settings = get_settings()
        # 모델 파일 로딩 시도
        try:
            bundle: dict = joblib.load(settings.MODEL_PATH)
        except Exception as e:
            print(f"ERROR: Failed to load model file: {e}")
            # 기본 모델 구성 요소 생성
            return create_default_model()
        
        scaler = bundle.get('scaler')
        centroids = bundle.get('centroids')
        weights = bundle.get('weights')
        minkowski_p = bundle.get('minkowski_p')

        if scaler is None or centroids is None or weights is None or minkowski_p is None:
            print("WARNING: Model bundle is missing one or more required components. Using default model.")
            return create_default_model()
        
        # CentroidPredictor 인스턴스 생성
        try:
            predictor = CentroidPredictor(
                centroids=centroids,
                weights=weights,
                minkowski_p=minkowski_p,
                sharpness_factor=5.0
            )
            return predictor, scaler
        except Exception as e:
            print(f"ERROR: Failed to create CentroidPredictor: {e}")
            return create_default_model()
            
    except Exception as e:
        print(f"CRITICAL: Unexpected error in load_model_components: {e}")
        return None, None

def create_default_model() -> Tuple[CentroidPredictor, None]:
    """기본 모델 구성 요소를 생성합니다."""
    print("INFO: Creating default model components")
    # 3개의 클러스터(BRONZE, SILVER, GOLD)에 대한 기본 중심점 생성
    # 각 중심점은 8개의 특성을 가짐
    default_centroids = np.array([
        [30, 30, 3, 0, 5, 0, 10, 15],  # SILVER
        [60, 20, 7, 1, 9, 1, 5, 30],   # GOLD
        [20, 50, 2, 0, 3, 0, 15, 5]     # BRONZE
    ])
    
    # 모든 특성에 동일한 가중치 부여
    default_weights = np.ones(8)
    
    # 기본 minkowski_p 값
    default_minkowski_p = 2.0
    
    # 기본 CentroidPredictor 생성
    default_predictor = CentroidPredictor(
        centroids=default_centroids,
        weights=default_weights,
        minkowski_p=default_minkowski_p,
        sharpness_factor=5.0
    )
    
    return default_predictor, None
