import joblib
from functools import lru_cache
from typing import Tuple, Any # Any 대신 더 구체적인 타입 사용 가능
from app.core.config import get_settings
from app.models.centroid_predictor import CentroidPredictor # 새로 추가

@lru_cache
def load_model_components() -> Tuple[CentroidPredictor, Any]: # 반환 타입을 튜플로 변경, 함수 이름 변경
    settings = get_settings()
    # 모델 파일은 'scaler', 'centroids', 'weights', 'minkowski_p'를 포함하는 딕셔너리
    bundle: dict = joblib.load(settings.MODEL_PATH)
    
    scaler = bundle.get('scaler') # Use .get for safety, though it should be there
    centroids = bundle.get('centroids')
    weights = bundle.get('weights')
    minkowski_p = bundle.get('minkowski_p')

    if scaler is None or centroids is None or weights is None or minkowski_p is None:
        # 실제 운영 환경에서는 더 구체적인 예외 처리나 로깅이 필요할 수 있습니다.
        raise ValueError("Model bundle is missing one or more required components: 'scaler', 'centroids', 'weights', 'minkowski_p'.")
    
    # CentroidPredictor 인스턴스 생성
    predictor = CentroidPredictor(
        centroids=centroids,
        weights=weights,
        minkowski_p=minkowski_p,
        sharpness_factor=5.0  # Further increased sharpness_factor
    )
    
    return predictor, scaler # 예측기와 스케일러를 튜플로 반환
