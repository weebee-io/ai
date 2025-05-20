from fastapi import APIRouter, BackgroundTasks, Depends, Path
from app.controllers.predict_controller import (
    predict_controller, 
    start_kafka_consumer, 
    stop_kafka_consumer, 
    get_prediction_result,
    get_latest_prediction_result,
    get_kafka_service
)
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut

router = APIRouter(prefix="/predict", tags=["Predict"])

# 기존 HTTP 예측 엔드포인트
router.post("/", response_model=PredictionOut)(predict_controller)

# Kafka 소비자 시작/중지 엔드포인트
router.post("/kafka/start")(start_kafka_consumer)
router.post("/kafka/stop")(stop_kafka_consumer)

# 최근 처리된 예측 결과 자동 조회 엔드포인트 - 구체적인 경로를 먼저 정의
router.get("/results/latest", response_model=dict)(get_latest_prediction_result)

# 사용자 ID별 예측 결과 조회 엔드포인트 - 파라미터가 있는 경로는 나중에 정의
router.get("/results/{user_id}", response_model=PredictionOut)(get_prediction_result)
