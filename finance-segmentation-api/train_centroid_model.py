import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
import os
import json
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# --- 환경 변수 로드 ---
load_dotenv()  # .env 파일에서 환경 변수 로드

# --- MySQL 연결 정보 ---
DB_CONFIG = {
    'host': os.getenv('DB_IP'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USERNAME'),
    'password': os.getenv('DB_PASSWORD')
}

# --- MLflow 설정 및 기본 파라미터 ---
MLFLOW_EXPERIMENT_NAME = "Centroid Model Training"

# 학습된 모델 및 아티팩트 저장 경로
BASE_OUTPUT_DIR = "models/mlflow_trained"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)  # 디렉토리가 없으면 생성
OUTPUT_MODEL_FILENAME = "centroid_weighted_model.joblib"
OUTPUT_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, OUTPUT_MODEL_FILENAME)

# 학습 파라미터
TEST_SIZE = 0.3
RANDOM_STATE = 42
KMEANS_N_CLUSTERS = 3
KMEANS_N_INIT = 'auto'
KMEANS_MAX_ITER = 100

# 피처 순서 (중요: 이 순서를 반드시 지켜야 함)
FEATURE_NAMES = [
    'essential_pct', 'discretionary_pct', 'risk_profile_score',
    'complex_product_flag', 'digital_engagement', 'is_married',
    'spend_volatility', 'sav_inv_ratio'
]

def load_data_from_mysql(query):
    """MySQL 서버에서 데이터를 로드하는 함수"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("MySQL 서버에 연결되었습니다.")
        
        # 쿼리 실행 및 결과를 DataFrame으로 변환
        df = pd.read_sql(query, conn)
        print(f"데이터 로드 완료. 총 {len(df)}개의 레코드가 로드되었습니다.")
        return df
        
    except Error as e:
        print(f"MySQL 오류 발생: {e}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            print("MySQL 연결이 종료되었습니다.")

def train_centroid_model():
    # MLflow 실험 설정
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # 1. 데이터 로드
        print("데이터 로드 중...")
        query = """
        SELECT 
            essential_pct, discretionary_pct, risk_profile_score,
            complex_product_flag, digital_engagement, is_married,
            spend_volatility, sav_inv_ratio
        FROM your_table_name  # 실제 테이블 이름으로 변경 필요
        """
        
        data = load_data_from_mysql(query)
        
        if data is None or data.empty:
            print("데이터를 로드하는 데 실패하거나 데이터가 비어 있습니다.")
            return
        
        # 2. 데이터 전처리
        print("데이터 전처리 중...")
        # 필요한 컬럼만 선택 (중복 확인)
        data = data[FEATURE_NAMES].copy()
        
        # 결측치 처리
        if data.isnull().any().any():
            print("결측치가 발견되어 0으로 대체합니다.")
            data = data.fillna(0)
        
        # 3. 데이터 분할
        X_train, X_test = train_test_split(
            data, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # 4. 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. K-means 모델 학습
        print("K-means 모델 학습 중...")
        kmeans = KMeans(
            n_clusters=KMEANS_N_CLUSTERS,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            random_state=RANDOM_STATE
        )
        kmeans.fit(X_train_scaled)
        
        # 6. 모델 평가
        train_score = kmeans.score(X_train_scaled)
        test_score = kmeans.score(X_test_scaled)
        
        # 7. 모델 저장
        model_bundle = {
            'centroids': kmeans.cluster_centers_,
            'weights': np.ones(len(FEATURE_NAMES)),  # 가중치 (필요에 따라 조정)
            'scaler': scaler,
            'feature_names': FEATURE_NAMES,
            'class_mapping': {0: 'SILVER', 1: 'GOLD', 2: 'BRONZE'}
        }
        
        # 모델 저장 디렉토리 생성
        os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
        joblib.dump(model_bundle, OUTPUT_MODEL_PATH)
        print(f"모델이 {OUTPUT_MODEL_PATH}에 저장되었습니다.")
        
        # 8. MLflow 로깅
        print("MLflow에 로깅 중...")
        # 파라미터 로깅
        mlflow.log_param("model_type", "WeightedCentroid")
        mlflow.log_param("n_clusters", KMEANS_N_CLUSTERS)
        mlflow.log_param("n_init", KMEANS_N_INIT)
        mlflow.log_param("max_iter", KMEANS_MAX_ITER)
        mlflow.log_param("test_size", TEST_SIZE)
        
        # 메트릭 로깅
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        
        # 모델 아티팩트 저장
        mlflow.log_artifact(OUTPUT_MODEL_PATH)
        
        # 피처 중요도 시각화 (선택사항)
        # ...
        
        print("학습 및 로깅이 완료되었습니다.")

if __name__ == "__main__":
    train_centroid_model()