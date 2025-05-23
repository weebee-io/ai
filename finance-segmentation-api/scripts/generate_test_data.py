#!/usr/bin/env python
import json
import random
import time
import uuid
from kafka import KafkaProducer

# Kafka 설정
KAFKA_BOOTSTRAP_SERVERS = 'kafka:29092'
INPUT_TOPIC = 'clustering_userRank'

def create_producer():
    """Kafka 프로듀서 생성"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        print(f"Kafka 프로듀서 생성 성공: {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        print(f"Kafka 프로듀서 생성 실패: {e}")
        return None

def generate_random_user_data():
    """랜덤 사용자 데이터 생성"""
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
    
    return user_id, data

def send_test_data(producer, count=10, interval=1):
    """테스트 데이터 전송"""
    if not producer:
        print("프로듀서가 없어 데이터를 전송할 수 없습니다.")
        return
    
    print(f"{count}개의 테스트 데이터를 {interval}초 간격으로 전송합니다...")
    
    for i in range(count):
        user_id, data = generate_random_user_data()
        try:
            # 메시지 전송 (키: user_id, 값: 금융 행동 데이터)
            future = producer.send(INPUT_TOPIC, key=user_id, value=data)
            record_metadata = future.get(timeout=10)
            
            print(f"데이터 전송 성공 ({i+1}/{count}): user_id={user_id}")
            print(f"  - 토픽: {record_metadata.topic}")
            print(f"  - 파티션: {record_metadata.partition}")
            print(f"  - 오프셋: {record_metadata.offset}")
            print(f"  - 데이터: {data}")
            
            # 간격 대기
            time.sleep(interval)
            
        except Exception as e:
            print(f"데이터 전송 실패 ({i+1}/{count}): {e}")
    
    # 모든 메시지가 전송되도록 보장
    producer.flush()
    print("모든 테스트 데이터 전송 완료")

if __name__ == "__main__":
    producer = create_producer()
    if producer:
        try:
            send_test_data(producer, count=20, interval=2)
        finally:
            producer.close()
    else:
        print("프로듀서 생성에 실패하여 테스트를 중단합니다.")
