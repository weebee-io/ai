import json
import threading
import time
from typing import Dict, Optional, Any, Union

from fastapi import Depends, HTTPException, BackgroundTasks, status
from app.schemas.model_input import InputData
from app.schemas.rank import PredictionOut
from app.services.predict_service import PredictService

# Kafka 모듈 임포트 시 예외 처리
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: Kafka module not available. Kafka functionality will be disabled.")


# Kafka 설정
KAFKA_BOOTSTRAP_SERVERS = '52.78.4.114:9092'
INPUT_TOPIC = 'clustering_userRank'
OUTPUT_TOPIC = 'clustering_results'

# 결과 저장 딕셔너리 - 사용자 ID별 예측 결과 캐싱
prediction_results: Dict[str, PredictionOut] = {}

# 가장 최근 처리된 user_id 저장 변수
latest_processed_user_id: Optional[str] = None

# Kafka 사용 가능 여부에 따라 서비스 구현 결정
if KAFKA_AVAILABLE:
    # Kafka Consumer 서비스 클래스
    class KafkaConsumerService:
        def __init__(self, predict_service: PredictService):
            self.predict_service = predict_service
            self.consumer = None
            self.producer = None
            self.running = False
            self.thread = None
        
        def start(self):
            """Kafka 소비자와 생산자를 초기화하고 백그라운드 스레드에서 메시지 처리를 시작합니다."""
            if self.running:
                return {"status": "already_running"}
                
            try:
                # Kafka Consumer 초기화
                self.consumer = KafkaConsumer(
                    INPUT_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    auto_offset_reset='earliest',
                    group_id='finance-segmentation-consumer',
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    key_deserializer=lambda x: x.decode('utf-8') if x else None
                )
                
                # Kafka Producer 초기화
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    key_serializer=lambda x: x.encode('utf-8') if x else None
                )
                
                # 상태 업데이트 및 스레드 시작
                self.running = True
                self.thread = threading.Thread(target=self._consume_messages)
                self.thread.daemon = True
                self.thread.start()
                
                return {"status": "started"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        def stop(self):
            """Kafka 소비자 및 생산자를 중지합니다."""
            if not self.running:
                return {"status": "not_running"}
                
            self.running = False
            if self.consumer:
                self.consumer.close()
            if self.thread:
                self.thread.join(timeout=5.0)
                
            return {"status": "stopped"}
        
        def _consume_messages(self):
            """백그라운드 스레드에서 실행되는 메시지 소비 및 처리 로직"""
            try:
                while self.running:
                    # 메시지 배치 가져오기 (timeout으로 블로킹 방지)
                    message_batch = self.consumer.poll(timeout_ms=1000)
                    
                    for tp, messages in message_batch.items():
                        for message in messages:
                            try:
                                # 키(userid)와 데이터(JSON) 추출
                                user_id = message.key
                                input_data = message.value
                                
                                # InputData 형식으로 변환
                                payload = InputData(**input_data)
                                
                                # 예측 수행
                                result = self.predict_service.predict(payload)
                                
                                # 결과 캐싱
                                prediction_results[user_id] = result
                                
                                # 최근 처리된 user_id 업데이트
                                global latest_processed_user_id
                                latest_processed_user_id = user_id
                                
                                # 결과를 새 토픽으로 전송
                                self._send_prediction_result(user_id, result)
                                
                                print(f"Processed prediction for user_id: {user_id}, result: {result}")
                            except Exception as e:
                                print(f"Error processing message: {e}")
                    
                    # CPU 사용률 절약을 위한 짧은 대기
                    time.sleep(0.1)
            except Exception as e:
                print(f"Kafka consumer error: {e}")
                self.running = False
        
        def _send_prediction_result(self, user_id: str, result: PredictionOut):
            """예측 결과를 Kafka 토픽으로 전송합니다."""
            try:
                # PredictionOut을 딕셔너리로 변환
                result_dict = result.model_dump()
                
                # 결과 전송 (키: 원래 메시지의 userid, 값: 예측 결과)
                self.producer.send(OUTPUT_TOPIC, key=user_id, value=result_dict)
                self.producer.flush()  # 즉시 전송 보장
            except Exception as e:
                print(f"Error sending prediction result to Kafka: {e}")

    # KafkaConsumerService 인스턴스 - FastAPI 시작 시 생성됨
    kafka_service = None

    def get_kafka_service(predict_service: PredictService = Depends(PredictService)):
        """KafkaConsumerService 인스턴스를 가져오거나 생성합니다."""
        global kafka_service
        if kafka_service is None:
            kafka_service = KafkaConsumerService(predict_service)
        return kafka_service

else:
    # Kafka를 사용할 수 없을 때의 대체 구현
    class KafkaConsumerService:
        def __init__(self, predict_service: PredictService):
            self.predict_service = predict_service
            self.running = False
        
        def start(self):
            return {"status": "error", "message": "Kafka is not available. Please install kafka-python package."}
        
        def stop(self):
            return {"status": "error", "message": "Kafka is not available. Please install kafka-python package."}
    
    # 대체 서비스 인스턴스
    kafka_service = None
    
    def get_kafka_service(predict_service: PredictService = Depends(PredictService)):
        global kafka_service
        if kafka_service is None:
            kafka_service = KafkaConsumerService(predict_service)
        return kafka_service

# 기존 예측 컨트롤러 - HTTP 요청용
def predict_controller(
    payload: InputData,
    service: PredictService = Depends(PredictService),
) -> PredictionOut:
    try:
        return service.predict(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )

# Kafka 소비자 시작/중지 컨트롤러 추가
def start_kafka_consumer(
    background_tasks: BackgroundTasks,
    kafka_service: KafkaConsumerService = Depends(get_kafka_service)
):
    """Kafka 소비자 서비스를 시작합니다."""
    return kafka_service.start()

def stop_kafka_consumer(
    kafka_service: KafkaConsumerService = Depends(get_kafka_service)
):
    """Kafka 소비자 서비스를 중지합니다."""
    return kafka_service.stop()

# 특정 사용자 ID에 대한 예측 결과 조회
def get_prediction_result(user_id: str):
    """캐시된 예측 결과를 사용자 ID로 조회합니다."""
    result = prediction_results.get(user_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction result for user_id '{user_id}' not found"
        )
    return result

# 최근 처리된 예측 결과 조회
def get_latest_prediction_result():
    """가장 최근에 처리된 예측 결과를 자동으로 조회합니다."""
    if latest_processed_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prediction results available yet"
        )
    
    result = prediction_results.get(latest_processed_user_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Latest prediction result not found"
        )
    
    return {
        "user_id": latest_processed_user_id,
        "result": result
    }
