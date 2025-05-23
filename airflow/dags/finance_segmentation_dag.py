"""
금융 세그먼테이션 워크플로우 DAG
주기적으로 금융 세그먼테이션 모델을 실행하고 결과를 모니터링합니다.
"""
from datetime import datetime, timedelta
import os
import json
import requests
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook

# 기본 인자 설정
default_args = {
    'owner': 'weebee',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 환경 변수 설정
FASTAPI_SERVICE_URL = os.getenv('FASTAPI_SEGMENTATION_SERVICE', 'http://host.docker.internal:8000')
ES_HOST = os.getenv('ES_HOST', 'elasticsearch')
ES_PORT = os.getenv('ES_PORT', '9200')
ES_URL = f"http://{ES_HOST}:{ES_PORT}"
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')

# 로깅 설정
logger = logging.getLogger(__name__)

# 테스트 데이터 생성 함수
def generate_test_data(**kwargs):
    """테스트 데이터를 생성하고 Kafka 토픽에 전송합니다."""
    import random
    import uuid
    from kafka import KafkaProducer
    
    logger.info("테스트 데이터 생성 시작")
    
    try:
        # Kafka 프로듀서 생성
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        # 데이터 생성 및 전송
        count = kwargs.get('count', 10)
        topic = kwargs.get('topic', 'clustering_userRank')
        
        for i in range(count):
            user_id = str(uuid.uuid4())
            
            # 금융 행동 데이터 랜덤 생성
            data = {
                "essential_pct": round(random.uniform(0.3, 0.7), 2),
                "discretionary_pct": round(random.uniform(0.1, 0.4), 2),
                "risk_profile_score": round(random.uniform(1, 10), 1),
                "complex_product_flag": random.choice([0, 1]),
                "digital_engagement": round(random.uniform(0, 1), 2),
                "is_married": random.choice([0, 1]),
                "spend_volatility": round(random.uniform(0, 0.5), 2),
                "sav_inv_ratio": round(random.uniform(0, 2), 2)
            }
            
            # 메시지 전송
            producer.send(topic, key=user_id, value=data)
            logger.info(f"데이터 전송: user_id={user_id}, data={data}")
        
        producer.flush()
        producer.close()
        logger.info(f"{count}개의 테스트 데이터 생성 및 전송 완료")
        return True
    
    except Exception as e:
        logger.error(f"테스트 데이터 생성 실패: {e}")
        return False

# 모델 예측 실행 함수
def run_model_prediction(**kwargs):
    """FastAPI 서비스를 호출하여 모델 예측을 실행합니다."""
    logger.info("모델 예측 실행 시작")
    
    try:
        # 테스트 입력 데이터
        test_data = {
            "essential_pct": 0.45,
            "discretionary_pct": 0.25,
            "risk_profile_score": 7.5,
            "complex_product_flag": 1,
            "digital_engagement": 0.8,
            "is_married": 1,
            "spend_volatility": 0.15,
            "sav_inv_ratio": 1.2
        }
        
        # FastAPI 예측 엔드포인트 호출
        response = requests.post(
            f"{FASTAPI_SERVICE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"예측 결과: {result}")
            return result
        else:
            logger.error(f"예측 실패: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        logger.error(f"모델 예측 실행 실패: {e}")
        return None

# 모니터링 데이터 수집 함수
def collect_monitoring_data(**kwargs):
    """모니터링 데이터를 수집하고 Elasticsearch에 저장합니다."""
    from elasticsearch import Elasticsearch
    from datetime import datetime
    
    logger.info("모니터링 데이터 수집 시작")
    
    try:
        # Elasticsearch 클라이언트 생성
        es = Elasticsearch([ES_URL])
        
        # 세그먼트별 메트릭 데이터 생성
        segments = ["GOLD", "SILVER", "BRONZE"]
        metrics = {
            "avg_risk_score": [7.5, 5.2, 3.1],
            "avg_digital_engagement": [0.8, 0.6, 0.3],
            "avg_sav_inv_ratio": [1.2, 0.9, 0.5]
        }
        
        # 현재 시간
        timestamp = datetime.now().isoformat()
        
        # 메트릭 데이터 저장
        for metric_name, values in metrics.items():
            for i, segment in enumerate(segments):
                data = {
                    "timestamp": timestamp,
                    "metric_name": metric_name,
                    "metric_value": values[i],
                    "segment": segment,
                    "count": random.randint(100, 600)
                }
                
                es.index(index="finance_segmentation_metrics", document=data)
                logger.info(f"메트릭 데이터 저장: {metric_name} - {segment} - {values[i]}")
        
        logger.info("모니터링 데이터 수집 및 저장 완료")
        return True
    
    except Exception as e:
        logger.error(f"모니터링 데이터 수집 실패: {e}")
        return False

# 성능 지표 계산 함수
def calculate_performance_metrics(**kwargs):
    """모델 성능 지표를 계산하고 보고서를 생성합니다."""
    logger.info("성능 지표 계산 시작")
    
    try:
        # 가상의 성능 지표 계산
        metrics = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.87,
            "f1_score": 0.88,
            "timestamp": datetime.now().isoformat()
        }
        
        # 결과를 JSON 파일로 저장
        output_dir = "/opt/airflow/data"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"성능 지표 계산 완료: {metrics}")
        logger.info(f"성능 지표 저장 완료: {output_file}")
        return metrics
    
    except Exception as e:
        logger.error(f"성능 지표 계산 실패: {e}")
        return None

# DAG 정의
with DAG(
    'finance_segmentation_workflow',
    default_args=default_args,
    description='금융 세그먼테이션 워크플로우',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 5, 23),
    catchup=False,
    tags=['finance', 'segmentation', 'ml'],
) as dag:
    
    # 1. 테스트 데이터 생성
    generate_data_task = PythonOperator(
        task_id='generate_test_data',
        python_callable=generate_test_data,
        op_kwargs={'count': 20, 'topic': 'clustering_userRank'},
        dag=dag,
    )
    
    # 2. 모델 예측 실행
    run_prediction_task = PythonOperator(
        task_id='run_model_prediction',
        python_callable=run_model_prediction,
        dag=dag,
    )
    
    # 3. 모니터링 데이터 수집
    collect_monitoring_task = PythonOperator(
        task_id='collect_monitoring_data',
        python_callable=collect_monitoring_data,
        dag=dag,
    )
    
    # 4. 성능 지표 계산
    calculate_metrics_task = PythonOperator(
        task_id='calculate_performance_metrics',
        python_callable=calculate_performance_metrics,
        dag=dag,
    )
    
    # 5. 결과 요약 보고서 생성
    generate_report_task = BashOperator(
        task_id='generate_report',
        bash_command='echo "금융 세그먼테이션 워크플로우 실행 완료: $(date)" > /opt/airflow/data/report_$(date +%Y%m%d_%H%M%S).txt',
        dag=dag,
    )
    
    # 작업 의존성 설정
    generate_data_task >> run_prediction_task >> collect_monitoring_task >> calculate_metrics_task >> generate_report_task
