#!/usr/bin/env python
import json
import os
from elasticsearch import Elasticsearch
from datetime import datetime

# Elasticsearch 설정
# Docker 내부에서는 'elasticsearch', 로컬에서는 'localhost' 사용
ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = os.getenv('ES_PORT', '9200')
ES_URL = f"http://{ES_HOST}:{ES_PORT}"

# 인덱스 이름
PREDICTION_INDEX = 'finance_segmentation_predictions'
METRICS_INDEX = 'finance_segmentation_metrics'

def create_elasticsearch_client():
    """Elasticsearch 클라이언트 생성"""
    try:
        es = Elasticsearch([ES_URL])
        print(f"Elasticsearch 연결 성공: {ES_URL}")
        return es
    except Exception as e:
        print(f"Elasticsearch 연결 실패: {e}")
        return None

def create_prediction_index(es):
    """예측 결과 저장용 인덱스 생성"""
    if not es:
        return False
    
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=PREDICTION_INDEX):
        print(f"인덱스 '{PREDICTION_INDEX}'가 이미 존재합니다.")
        return True
    
    # 인덱스 매핑 정의
    mappings = {
        "mappings": {
            "properties": {
                "timestamp": {"type": "date"},
                "user_id": {"type": "keyword"},
                "lit_level": {"type": "keyword"},
                "proba": {"type": "float"},
                "input_data": {"type": "object", "dynamic": True}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    try:
        es.indices.create(index=PREDICTION_INDEX, body=mappings)
        print(f"인덱스 '{PREDICTION_INDEX}' 생성 성공")
        return True
    except Exception as e:
        print(f"인덱스 생성 실패: {e}")
        return False

def create_metrics_index(es):
    """메트릭 데이터 저장용 인덱스 생성"""
    if not es:
        return False
    
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=METRICS_INDEX):
        print(f"인덱스 '{METRICS_INDEX}'가 이미 존재합니다.")
        return True
    
    # 인덱스 매핑 정의
    mappings = {
        "mappings": {
            "properties": {
                "timestamp": {"type": "date"},
                "metric_name": {"type": "keyword"},
                "metric_value": {"type": "float"},
                "segment": {"type": "keyword"},
                "count": {"type": "integer"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    try:
        es.indices.create(index=METRICS_INDEX, body=mappings)
        print(f"인덱스 '{METRICS_INDEX}' 생성 성공")
        return True
    except Exception as e:
        print(f"인덱스 생성 실패: {e}")
        return False

def insert_sample_data(es):
    """샘플 데이터 삽입"""
    if not es:
        return False
    
    # 샘플 예측 데이터
    prediction_samples = [
        {
            "timestamp": datetime.now().isoformat(),
            "user_id": "user123",
            "lit_level": "GOLD",
            "proba": 0.85,
            "input_data": {
                "essential_pct": 0.45,
                "discretionary_pct": 0.25,
                "risk_profile_score": 7.5,
                "complex_product_flag": 1,
                "digital_engagement": 0.8,
                "is_married": 1,
                "spend_volatility": 0.15,
                "sav_inv_ratio": 1.2
            }
        },
        {
            "timestamp": datetime.now().isoformat(),
            "user_id": "user456",
            "lit_level": "SILVER",
            "proba": 0.72,
            "input_data": {
                "essential_pct": 0.52,
                "discretionary_pct": 0.18,
                "risk_profile_score": 5.2,
                "complex_product_flag": 0,
                "digital_engagement": 0.6,
                "is_married": 1,
                "spend_volatility": 0.22,
                "sav_inv_ratio": 0.9
            }
        },
        {
            "timestamp": datetime.now().isoformat(),
            "user_id": "user789",
            "lit_level": "BRONZE",
            "proba": 0.64,
            "input_data": {
                "essential_pct": 0.65,
                "discretionary_pct": 0.12,
                "risk_profile_score": 3.1,
                "complex_product_flag": 0,
                "digital_engagement": 0.3,
                "is_married": 0,
                "spend_volatility": 0.35,
                "sav_inv_ratio": 0.5
            }
        }
    ]
    
    # 샘플 메트릭 데이터
    metric_samples = [
        {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "avg_risk_score",
            "metric_value": 7.5,
            "segment": "GOLD",
            "count": 120
        },
        {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "avg_risk_score",
            "metric_value": 5.2,
            "segment": "SILVER",
            "count": 350
        },
        {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "avg_risk_score",
            "metric_value": 3.1,
            "segment": "BRONZE",
            "count": 530
        }
    ]
    
    # 예측 데이터 삽입
    for sample in prediction_samples:
        try:
            es.index(index=PREDICTION_INDEX, document=sample)
            print(f"예측 샘플 데이터 삽입 성공: {sample['user_id']}")
        except Exception as e:
            print(f"데이터 삽입 실패: {e}")
    
    # 메트릭 데이터 삽입
    for sample in metric_samples:
        try:
            es.index(index=METRICS_INDEX, document=sample)
            print(f"메트릭 샘플 데이터 삽입 성공: {sample['metric_name']} - {sample['segment']}")
        except Exception as e:
            print(f"데이터 삽입 실패: {e}")
    
    return True

def print_kibana_setup_instructions():
    """Kibana 대시보드 설정 안내"""
    print("\n=== Kibana 대시보드 설정 안내 ===")
    print("1. Kibana에 접속: http://localhost:5601")
    print("2. 'Stack Management' > 'Index Patterns' 메뉴로 이동")
    print(f"3. '{PREDICTION_INDEX}' 인덱스 패턴 생성 (timestamp 필드를 시간 필드로 설정)")
    print(f"4. '{METRICS_INDEX}' 인덱스 패턴 생성 (timestamp 필드를 시간 필드로 설정)")
    print("5. 'Dashboard' 메뉴에서 새 대시보드 생성")
    print("6. 다음 시각화 추가:")
    print("   - 세그먼트별 사용자 분포 (파이 차트)")
    print("   - 시간별 세그먼트 변화 추이 (라인 차트)")
    print("   - 세그먼트별 평균 위험 점수 (막대 차트)")
    print("   - 디지털 참여도와 위험 점수 관계 (히트맵)")

if __name__ == "__main__":
    es = create_elasticsearch_client()
    if es:
        create_prediction_index(es)
        create_metrics_index(es)
        insert_sample_data(es)
        print_kibana_setup_instructions()
    else:
        print("Elasticsearch 연결에 실패하여 설정을 중단합니다.")
