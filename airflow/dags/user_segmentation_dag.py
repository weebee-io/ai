from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import json
import requests
# Try to import confluent_kafka, but fall back to kafka-python if not available
try:
    from confluent_kafka import Consumer
    USING_CONFLUENT = True
except ImportError:
    from kafka import KafkaConsumer
    USING_CONFLUENT = False
from elasticsearch import Elasticsearch

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')  # Docker 환경에서는 kafka:29092 사용
KAFKA_TOPIC = 'new-user-topic'
# FastAPI 서비스 URL 수정 - Docker 네트워크에서 접근할 수 있도록 finance-segmentation-api 컨테이너 이름 사용
FASTAPI_SEGMENTATION_SERVICE = os.getenv('FASTAPI_SEGMENTATION_SERVICE', 'http://finance-segmentation-api:8000')
ES_HOST = os.getenv('ES_HOST', 'elasticsearch')
ES_PORT = os.getenv('ES_PORT', '9200')

# 개발/디버깅 모드 설정
DEBUG_MODE = False  # 디버그 모드 비활성화

# 시작 시 설정 정보 출력
print(f"[CONFIG] Kafka Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
print(f"[CONFIG] Kafka Topic: {KAFKA_TOPIC}")
print(f"[CONFIG] FastAPI Service: {FASTAPI_SEGMENTATION_SERVICE}")
print(f"[CONFIG] Elasticsearch: {ES_HOST}:{ES_PORT}")
print(f"[CONFIG] Debug Mode: {DEBUG_MODE}")

# Consumer configuration
consumer_config = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'group.id': f'user-segmentation-group-{datetime.now().strftime("%Y%m%d%H%M%S")}',  # 유니크한 그룹 ID 생성
    'auto.offset.reset': 'earliest',  # 가장 오래된 메시지부터 가져오도록 변경
    'enable.auto.commit': False,
}

def consume_new_user_event(**kwargs):
    """
    Consumes new user event from Kafka topic
    """
    print(f"Starting to consume messages from {KAFKA_TOPIC} using bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Consumer config: {consumer_config}")
    
    # 디버그 정보 출력
    import socket
    try:
        kafka_host = KAFKA_BOOTSTRAP_SERVERS.split(':')[0]
        kafka_port = int(KAFKA_BOOTSTRAP_SERVERS.split(':')[1])
        print(f"Attempting to connect to Kafka at {kafka_host}:{kafka_port}...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex((kafka_host, kafka_port))
        if result == 0:
            print(f"Successfully connected to Kafka at {kafka_host}:{kafka_port}")
        else:
            print(f"Failed to connect to Kafka at {kafka_host}:{kafka_port}, error code: {result}")
        s.close()
    except Exception as e:
        print(f"Error testing Kafka connection: {e}")
    
    # 디버그 모드일 경우 테스트 데이터 사용
    if DEBUG_MODE:
        test_user_data = {
            "userId": 85,
            "userLoginId": "henryh3nry",
            "nickname": "Henry",
            "name": "정창훈",
            "gender": "미정",
            "age": 28,
            "userRank": "string"
        }
        
        print(f"[DEBUG MODE] Using test user data: {test_user_data}")
        return [test_user_data]
    
    # 최대 10개 메시지 처리 또는 10초 타임아웃
    max_messages = 10
    processed_messages = 0
    user_events = []
    
    try:
        if USING_CONFLUENT:
            # Confluent Kafka 사용
            print(f"Using confluent-kafka with config: {consumer_config}")
            consumer = Consumer(consumer_config)
            consumer.subscribe([KAFKA_TOPIC])
            
            print("Polling for messages from Kafka...")
            while processed_messages < max_messages:
                msg = consumer.poll(timeout=10.0)
                if msg is None:
                    print("No message received, continuing...")
                    break
                
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue
                    
                try:
                    # Spring Boot에서 보낸 메시지 처리
                    message_str = msg.value().decode('utf-8')
                    print(f"Raw message: {message_str}")
                    user_data = json.loads(message_str)
                    
                    # _Type_Id_ 필드가 있는지 확인하고 제거
                    if "_Type_Id_" in user_data:
                        print(f"Found _Type_Id_: {user_data['_Type_Id_']}")
                        del user_data["_Type_Id_"]
                    
                    print(f"Processed message: {user_data}")
                    user_events.append(user_data)
                    processed_messages += 1
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
                consumer.commit(msg)
            
            consumer.close()
        else:
            # Kafka-Python 사용
            print("Using kafka-python")
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id='user-segmentation-group-new',  # 컨스머 그룹 ID 변경
                auto_offset_reset='latest',  # 최신 메시지부터 가져오도록 변경
                enable_auto_commit=False,
                consumer_timeout_ms=10000  # 10초 타임아웃
            )
            
            print("Polling for messages from Kafka...")
            for msg in consumer:
                try:
                    # Spring Boot에서 보낸 메시지 처리
                    message_str = msg.value.decode('utf-8')
                    print(f"Raw message: {message_str}")
                    user_data = json.loads(message_str)
                    
                    # _Type_Id_ 필드가 있는지 확인하고 제거
                    if "_Type_Id_" in user_data:
                        print(f"Found _Type_Id_: {user_data['_Type_Id_']}")
                        del user_data["_Type_Id_"]
                    
                    print(f"Processed message: {user_data}")
                    user_events.append(user_data)
                    processed_messages += 1
                    consumer.commit()
                    
                    if processed_messages >= max_messages:
                        break
                except Exception as e:
                    print(f"Error processing message: {e}")
            
            consumer.close()
    except Exception as e:
        print(f"Error consuming from Kafka: {e}")
        # 오류 발생 시 디버그 모드에서는 테스트 데이터 사용
        if DEBUG_MODE:
            print("Error occurred, using debug data instead")
            user_events = [{
                "userId": 85,
                "userLoginId": "henryh3nry",
                "nickname": "Henry",
                "name": "정창훈",
                "gender": "미정",
                "age": 28,
                "userRank": "string"
            }]
    
    # 메시지가 없으면 디버그용 데이터 사용
    if not user_events and DEBUG_MODE:
        print("No messages received from Kafka, using debug data")
        user_events = [{
            "userId": 85,
            "userLoginId": "henryh3nry",
            "nickname": "Henry",
            "name": "정창훈",
            "gender": "미정",
            "age": 28,
            "userRank": "string"
        }]
        
    # 처리된 사용자 이벤트 반환
    return user_events

def process_user_segmentation(ti, **kwargs):
    """
    Process user data and make prediction request to FastAPI segmentation service
    """
    user_events = ti.xcom_pull(task_ids='consume_new_user_event')
    
    if not user_events:
        print("No user events to process")
        return []
        
    segmentation_results = []
    
    for user_event in user_events:
        try:
            # 사용자 특성에서 필요한 정보 추출
            # 나이와 성별에 따라 간단한 변환 규칙 적용
            age = user_event.get('age', 30)
            gender = user_event.get('gender', '미정')
            
            # 나이를 기반으로 간단한 위험 프로필 점수 계산 (예시)
            risk_profile_score = min(10, max(1, int(age / 10))) if age else 5
            
            # 성별에 따른 간단한 규칙 (예시)
            is_married = 1 if age > 30 else 0  # 단순 추정
            
            # 실제 데이터가 없으므로 사용자 ID 기반의 일관된 가상 데이터 생성
            # 실제로는 사용자 실제 금융 데이터를 사용해야 함
            user_id = user_event.get('userId', 0)
            seed = user_id % 100 if user_id else 50
            
            model_input_payload = {
                "risk_profile_score": risk_profile_score,
                "complex_product_flag": 1 if age > 40 else 0,
                "is_married": is_married,
                "essential_pct": 40 + (seed % 20),
                "discretionary_pct": 30 + (seed % 15),
                "sav_inv_ratio": 20 + (seed % 10),
                "spend_volatility": 5 + (seed % 10),
                "digital_engagement": 7 + (seed % 3)
            }
            
            print(f"Sending prediction request for user {user_event.get('userId')}: {model_input_payload}")
            
            # FastAPI 서비스에 예측 요청
            response = requests.post(
                f"{FASTAPI_SEGMENTATION_SERVICE}/predict/", 
                json=model_input_payload
            )
            
            if response.status_code == 200:
                prediction_result = response.json()
                
                # 원본 사용자 이벤트 데이터와 예측 결과 결합
                segmented_user = {
                    **user_event,  # 모든 원본 사용자 이벤트 필드 포함
                    'prediction': prediction_result,
                    'processed_at': datetime.now().isoformat(),
                    'input_features': model_input_payload  # 입력 특성도 저장하여 추적 가능하게
                }
                
                segmentation_results.append(segmented_user)
                print(f"성공적으로 사용자 {user_event.get('userId')}의 세그먼테이션 처리 완료")
            else:
                print(f"사용자 {user_event.get('userId')}의 예측 획득 실패: {response.text}")
                
        except Exception as e:
            print(f"사용자 세그먼테이션 처리 중 오류 발생: {e}")
    
    return segmentation_results

def store_user_segments(ti, **kwargs):
    """
    Store user segmentation results in Elasticsearch
    """
    segmentation_results = ti.xcom_pull(task_ids='process_user_segmentation')
    
    if not segmentation_results:
        print("No segmentation results to store")
        return
        
    try:
        # Elasticsearch 연결
        print(f"Connecting to Elasticsearch at http://{ES_HOST}:{ES_PORT}")
        es = Elasticsearch([f'http://{ES_HOST}:{ES_PORT}'])
        
        # 인덱스 존재 확인, 없으면 생성
        index_name = 'user-segments'
        
        # Elasticsearch 연결 확인
        if not es.ping():
            print("Elasticsearch에 연결할 수 없습니다. 설정을 확인하세요.")
            if DEBUG_MODE:
                print("디버그 모드: Elasticsearch 저장 단계를 건너뜁니다.")
                return
            raise Exception("Elasticsearch connection failed")
            
        # 인덱스 확인 및 생성
        if not es.indices.exists(index=index_name):
            print(f"Creating new index: {index_name}")
            index_settings = {
                'mappings': {
                    'properties': {
                        'userId': {'type': 'integer'},
                        'userLoginId': {'type': 'keyword'},
                        'nickname': {'type': 'keyword'},
                        'name': {'type': 'text'},
                        'gender': {'type': 'keyword'},
                        'age': {'type': 'integer'},
                        'userRank': {'type': 'keyword'},
                        'prediction': {
                            'type': 'object',
                            'properties': {
                                'segment': {'type': 'keyword'},
                                'segment_description': {'type': 'text'}
                            }
                        },
                        'input_features': {
                            'type': 'object'
                        },
                        'processed_at': {'type': 'date'}
                    }
                }
            }
            es.indices.create(index=index_name, body=index_settings)
        
        # 세그먼테이션 결과 인덱싱
        for result in segmentation_results:
            try:
                # userId를 문서 ID로 사용하여 업데이트 허용
                user_id = result.get('userId')
                if user_id is None:
                    print(f"Warning: Missing userId in result, skipping: {result}")
                    continue
                    
                print(f"Indexing document for user {user_id}")
                es.index(
                    index=index_name,
                    id=user_id,
                    document=result
                )
                print(f"사용자 {user_id}의 세그먼테이션 결과를 Elasticsearch에 저장 완료")
            except Exception as e:
                print(f"Elasticsearch에 사용자 세그먼트 저장 중 오류: {e}")
    except Exception as e:
        print(f"Elasticsearch 처리 중 오류 발생: {e}")
        if DEBUG_MODE:
            print("디버그 모드: 오류가 발생했지만 계속 진행합니다.")

# Create the DAG
with DAG(
    'user_segmentation_dag',
    default_args=default_args,
    description='A DAG to process new user events and perform segmentation',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2025, 5, 1),
    catchup=False,
    tags=['user', 'segmentation', 'kafka'],
) as dag:
    
    # Task to consume new user events from Kafka
    consume_task = PythonOperator(
        task_id='consume_new_user_event',
        python_callable=consume_new_user_event,
        provide_context=True,
    )
    
    # Task to process user data and get segmentation
    segment_task = PythonOperator(
        task_id='process_user_segmentation',
        python_callable=process_user_segmentation,
        provide_context=True,
    )
    
    # Task to store user segmentation results in Elasticsearch
    store_task = PythonOperator(
        task_id='store_user_segments',
        python_callable=store_user_segments,
        provide_context=True,
    )
    
    # Set task dependencies
    consume_task >> segment_task >> store_task
