import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import mlflow
import mlflow.pyfunc
import joblib
import os
import json
from mlflow.models.signature import infer_signature
import matplotlib
matplotlib.use('Agg') # Ensure this is called before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from dotenv import load_dotenv

# --- MLflow 설정 및 기본 파라미터 ---
MLFLOW_EXPERIMENT_NAME = "Centroid Model Training (CSV Input)"
MLFLOW_MODEL_ARTIFACT_PATH = "kmeans_pyfunc_model_csv"

# 로컬 임시 모델 번들 저장 경로
BASE_OUTPUT_DIR = "models/mlflow_trained"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
OUTPUT_MODEL_BUNDLE_FILENAME = "centroid_model_bundle_csv.joblib"
LOCAL_MODEL_BUNDLE_PATH = os.path.join(BASE_OUTPUT_DIR, OUTPUT_MODEL_BUNDLE_FILENAME)
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

# CSV 파일 경로 (Airflow DAG의 CWD를 기준으로 상대 경로 또는 절대 경로)
# 현재는 train_centroid_model.py와 같은 디렉토리 또는 프로젝트 루트에 있다고 가정
CSV_FILE_PATH = '../masterpiece.csv' # 경로 수정: 상위 폴더의 masterpiece.csv"

# 학습 파라미터
TEST_SIZE = 0.3
RANDOM_STATE = 42
KMEANS_N_CLUSTERS = 3 # 'SILVER', 'GOLD', 'BRONZE'
KMEANS_N_INIT = 'auto'
KMEANS_MAX_ITER = 300

# 피처 이름 리스트 (MySQL survey 테이블의 컬럼명과 일치해야 함)
FEATURE_NAMES = [
    'essential_pct',        # 필수 소비 비율
    'discretionary_pct',    # 재량 소비 비율
    'risk_profile_score',   # 위험 프로파일 점수 (1-10)
    'complex_product_flag', # 복합 상품 보유 여부 (0 또는 1)
    'digital_engagement',   # 디지털 참여도 (1-5)
    'is_married',           # 결혼 여부 (0 또는 1)
    'spend_volatility',     # 소비 변동성 (0-1)
    'sav_inv_ratio'        # 저축 대비 투자 비율
]

LABEL_COLUMN_NAME = 'user_rank' # 실제 레이블이 있는 컬럼명 (user 테이블)

# KMeans 클러스터 인덱스와 실제 세그먼트 이름 간의 고정 매핑
# Memory eb6d10fe: 0 -> 'SILVER', 1 -> 'GOLD', 2 -> 'BRONZE'
# 이 순서는 K-means 클러스터 중심점의 순서와 대응된다고 가정합니다.
# 실제로는 클러스터 특성 분석 후 이 매핑을 검증/결정해야 합니다.
FIXED_CLASS_MAPPING_INDICES_TO_NAMES = {0: 'SILVER', 1: 'GOLD', 2: 'BRONZE'} # 예시: 실제 값으로 변경
FIXED_CLASS_MAPPING_NAMES_TO_INDICES = {name: idx for idx, name in FIXED_CLASS_MAPPING_INDICES_TO_NAMES.items()}

class KMeansPyfuncWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_bundle_path = context.artifacts["model_bundle"]
        bundle = joblib.load(model_bundle_path)
        self.scaler = bundle['scaler']
        self.kmeans_model = bundle['kmeans_model']
        self.feature_names = bundle['feature_names']
        self.class_mapping_from_indices = bundle['class_mapping_from_indices']

    def _predict_cluster_indices(self, model_input_df: pd.DataFrame) -> np.ndarray:
        # 입력 DataFrame 컬럼 순서 확인 및 재정렬
        try:
            model_input_df_reordered = model_input_df[self.feature_names]
        except KeyError as e:
            raise ValueError(f"Input DataFrame is missing one or more expected columns. Expected: {self.feature_names}, Got: {model_input_df.columns.tolist()}. Error: {e}")
        
        scaled_input = self.scaler.transform(model_input_df_reordered)
        cluster_indices = self.kmeans_model.predict(scaled_input)
        return cluster_indices

    def predict(self, context, model_input):
        if isinstance(model_input, list) and len(model_input) > 0 and isinstance(model_input[0], dict):
            model_input_df = pd.DataFrame(model_input)
        elif not isinstance(model_input, pd.DataFrame):
            # MLflow 서빙 시 입력 형식을 고려하여 DataFrame으로 변환
            try:
                model_input_df = pd.DataFrame(model_input, columns=self.feature_names)
            except Exception as e:
                 raise ValueError(f"Could not convert model_input to DataFrame. Input type: {type(model_input)}, Error: {e}")
        else:
            model_input_df = model_input

        cluster_indices = self._predict_cluster_indices(model_input_df)
        # class_mapping_from_indices의 키가 정수인지 확인 (JSON 저장/로드 시 문자열로 변형 가능성 방지)
        mapping_int_keys = {int(k): v for k, v in self.class_mapping_from_indices.items()}
        predicted_classes = [mapping_int_keys.get(idx, f"Unmapped_Cluster_{idx}") for idx in cluster_indices]
        return predicted_classes


def load_data_from_mysql(feature_cols, label_col):
    load_dotenv() # .env 파일에서 환경 변수 로드

    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    db_port = os.getenv("DB_PORT", "3306") # 기본 MySQL 포트

    if not all([db_host, db_user, db_password, db_name]):
        print("오류: 데이터베이스 연결 정보가 .env 파일에 올바르게 설정되지 않았습니다.")
        print("필수 환경 변수: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME")
        return None, None

    conn = None
    try:
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=db_port
        )
        print(f"MySQL 데이터베이스 '{db_name}'에 성공적으로 연결되었습니다.")

        # DB의 컬럼명이 feature_cols, label_col과 일치해야 함
        # feature_cols는 survey 테이블에서, label_col은 user 테이블에서 가져옴
        survey_feature_cols_str = ", ".join([f"s.`{col}`" for col in feature_cols])
        user_label_col_str = f"u.`{label_col}`"

        query = f"""
        SELECT 
            {survey_feature_cols_str}, 
            {user_label_col_str}
        FROM 
            survey s
        JOIN 
            users u ON s.user_id = u.user_id
        """
        print(f"Executing query: {query}")
        df = pd.read_sql(query, conn)
        print(f"데이터베이스에서 {len(df)}개의 레코드를 로드했습니다.")

        # 필수 컬럼들이 DataFrame에 모두 있는지 확인
        # feature_cols는 s.col 형태로, label_col은 u.col 형태로 가져오므로, df에는 원래 컬럼명으로 존재
        all_expected_columns = feature_cols + [label_col]
        missing_cols_in_df = [col for col in all_expected_columns if col not in df.columns]
        if missing_cols_in_df:
            print(f"오류: 데이터베이스에서 다음 필수 컬럼을 가져오지 못했습니다: {missing_cols_in_df}")
            return None, None
            
        X = df[feature_cols].copy()
        # 레이블 컬럼(user_rank)의 문자열 값을 정수 인덱스로 변환
        if not df[label_col].isin(FIXED_CLASS_MAPPING_NAMES_TO_INDICES.keys()).all():
            unknown_labels = df[~df[label_col].isin(FIXED_CLASS_MAPPING_NAMES_TO_INDICES.keys())][label_col].unique()
            print(f"오류: '{label_col}' 컬럼에 알 수 없는 레이블 값이 포함되어 있습니다: {unknown_labels}")
            print(f"기대하는 레이블 값: {list(FIXED_CLASS_MAPPING_NAMES_TO_INDICES.keys())}")
            return None, None
            
        y_true = df[label_col].map(FIXED_CLASS_MAPPING_NAMES_TO_INDICES).astype(int).copy()
        print(f"레이블 컬럼 '{label_col}'을 성공적으로 정수 인덱스로 변환했습니다.")
        return X, y_true

    except mysql.connector.Error as err:
        print(f"MySQL 연결 또는 쿼리 실행 중 오류 발생: {err}")
        # mlflow.log_param("data_loading_error", f"MySQL error: {err}")
        return None, None
    except Exception as e:
        print(f"데이터베이스 데이터 로드 중 일반 오류 발생: {e}")
        # mlflow.log_param("data_loading_error", f"Generic DB load error: {e}")
        return None, None
    finally:
        if conn and conn.is_connected():
            conn.close()
            print("MySQL 연결이 닫혔습니다.")


def map_kmeans_clusters_to_true_labels(cluster_assignments, y_true, class_mapping_indices_to_names):
    """KMeans 클러스터 인덱스를 class_mapping_indices_to_names에 따라 실제 레이블 이름으로 변환합니다."""
    # 클러스터 인덱스가 정수형인지 확인
    mapped_predictions = pd.Series(cluster_assignments, index=y_true.index).map(class_mapping_indices_to_names)
    # 매핑되지 않은 경우 (예: class_mapping_indices_to_names에 없는 클러스터 인덱스) NaN이 될 수 있음
    # 이를 'Unknown_Cluster_X' 등으로 처리할 수 있으나, 여기서는 고정 매핑이므로 모든 클러스터가 커버된다고 가정.
    # 만약 NaN이 있다면, 정확도 계산 전 처리 필요.
    if mapped_predictions.isnull().any():
        print("Warning: Some cluster assignments could not be mapped to true label names. Check FIXED_CLASS_MAPPING_INDICES_TO_NAMES.")
        # Fallback for unmapped clusters if any (should ideally not happen with fixed mapping)
        mapped_predictions = mapped_predictions.fillna('Unmapped_Cluster')
    return mapped_predictions


def train_centroid_model():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"KMeans Training Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") as run:
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("num_features", len(FEATURE_NAMES))
        mlflow.log_param("features", FEATURE_NAMES)
        mlflow.log_param("label_column_name_in_source", LABEL_COLUMN_NAME)
        mlflow.log_param("fixed_class_mapping_indices_to_names", str(FIXED_CLASS_MAPPING_INDICES_TO_NAMES))
        mlflow.log_param("fixed_class_mapping_names_to_indices", str(FIXED_CLASS_MAPPING_NAMES_TO_INDICES))

        # 데이터 로드
        print(f"'{MLFLOW_EXPERIMENT_NAME}' 실험 하에 다음 소스에서 데이터 로드 중: MySQL (survey & user 테이블 조인)")
        X, y_true = load_data_from_mysql(FEATURE_NAMES, LABEL_COLUMN_NAME)
        mlflow.log_param("data_source", "mysql_survey_user_join")

        if X is None or y_true is None:
            print("데이터 로드 실패. 학습을 중단합니다.")
            mlflow.log_param("data_loading_status", "failed_db_join_or_mapping_issue")
            return
        
        mlflow.log_param("data_loading_status", "success_db")
        mlflow.log_metric("loaded_records_count", len(X))

        print("데이터 전처리 (결측치 처리)...")
        if X.isnull().values.any():
            print("피처 데이터에 결측치가 발견되어 0으로 대체합니다.")
            missing_counts = X.isnull().sum()
            print("결측치 수:\n", missing_counts[missing_counts > 0])
            for col, count in missing_counts.items():
                if count > 0: mlflow.log_metric(f"missing_{col}_count_before_fill", count)
            X = X.fillna(0)
            mlflow.log_param("missing_value_handling", "filled_with_zero")
        else:
            print("피처 데이터에 결측치가 없습니다.")
            mlflow.log_param("missing_value_handling", "none_needed")
        
        # 데이터 분할 (정답 레이블 y_true를 사용하여 계층적 샘플링 시도)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_true, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_true
            )
        except ValueError as e:
            print(f"Stratify 적용 중 오류 (레이블 분포 문제 가능성): {e}. Stratify 없이 분할합니다.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_true, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("training_set_size", len(X_train))
        mlflow.log_metric("test_set_size", len(X_test))
        
        # 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # K-means 모델 학습
        print("K-means 모델 학습 중...")
        kmeans_model = KMeans(
            n_clusters=KMEANS_N_CLUSTERS, n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER, random_state=RANDOM_STATE
        )
        # 학습 데이터에 대한 클러스터 레이블 예측 (KMeans 내부에서 fit_predict 사용)
        train_cluster_labels = kmeans_model.fit_predict(X_train_scaled)
        
        # 학습 데이터 평가 메트릭
        train_inertia = kmeans_model.inertia_
        mlflow.log_metric("train_inertia", train_inertia)
        if len(np.unique(train_cluster_labels)) > 1: # 실루엣 점수는 2개 이상의 클러스터 필요
            train_silhouette = silhouette_score(X_train_scaled, train_cluster_labels)
            mlflow.log_metric("train_silhouette_score", train_silhouette)
        
        # 테스트 데이터에 대한 클러스터 레이블 예측
        test_cluster_labels = kmeans_model.predict(X_test_scaled)
        if len(np.unique(test_cluster_labels)) > 1:
            test_silhouette = silhouette_score(X_test_scaled, test_cluster_labels)
            mlflow.log_metric("test_silhouette_score", test_silhouette)

        # FIXED_CLASS_MAPPING_INDICES_TO_NAMES를 사용하여 y_test와 K-means 예측 결과를 문자열 레이블로 변환
        class_mapping = FIXED_CLASS_MAPPING_INDICES_TO_NAMES

        # y_test (실제 레이블 인덱스)를 문자열 레이블로 변환
        y_test_mapped_str = [class_mapping.get(idx, f"Unmapped_True_Label_{idx}") for idx in y_test.astype(int)]

        # test_cluster_labels (예측된 클러스터 인덱스)를 문자열 레이블로 변환
        y_pred_test_mapped_str = [class_mapping.get(idx, f"Unmapped_Pred_Label_{idx}") for idx in test_cluster_labels]

        # 분류 성능 평가에 사용할 고유 레이블 목록 (문자열, 정렬됨)
        unique_str_labels = sorted(list(class_mapping.values()))

        # 분류 성능 평가 (모두 문자열 레이블 사용)
        test_accuracy = accuracy_score(y_test_mapped_str, y_pred_test_mapped_str)
        report_dict = classification_report(y_test_mapped_str, y_pred_test_mapped_str, output_dict=True, zero_division=0, labels=unique_str_labels)
        report_str = classification_report(y_test_mapped_str, y_pred_test_mapped_str, zero_division=0, labels=unique_str_labels)
        
        print(f"Test Accuracy (after mapping): {test_accuracy:.4f}")
        print("Classification Report (Test Set):\n", report_str)
        
        mlflow.log_metric("test_accuracy", test_accuracy)
        for class_label, metrics_dict in report_dict.items():
            if isinstance(metrics_dict, dict): # 'accuracy', 'macro avg', 'weighted avg' 제외
                for metric_name, value in metrics_dict.items():
                    mlflow.log_metric(f"{class_label}_{metric_name}", value)
            elif class_label in ['accuracy', 'macro avg', 'weighted avg']:
                 mlflow.log_metric(f"{class_label.replace(' ', '_')}", metrics_dict)
        
        # 혼동 행렬 로깅
        cm = confusion_matrix(y_test_mapped_str, y_pred_test_mapped_str, labels=unique_str_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_str_labels, yticklabels=unique_str_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(BASE_OUTPUT_DIR, CONFUSION_MATRIX_FILENAME)
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

        # 모델 번들 생성 및 로컬 저장
        model_bundle = {
            'scaler': scaler,
            'kmeans_model': kmeans_model, # 학습된 KMeans 객체 자체를 저장
            'feature_names': FEATURE_NAMES,
            'class_mapping_from_indices': FIXED_CLASS_MAPPING_INDICES_TO_NAMES
        }
        joblib.dump(model_bundle, LOCAL_MODEL_BUNDLE_PATH)
        print(f"모델 번들이 '{LOCAL_MODEL_BUNDLE_PATH}'에 저장되었습니다.")
        
        # MLflow Pyfunc 모델 로깅 (시그니처 및 입력 예제 포함)
        # 입력 예제 생성 (첫 5개 학습 데이터 사용 또는 명시적 예제)
        input_example = X_train.head()
        # 모델 시그니처 추론
        # Pyfunc 모델의 predict 메서드가 DataFrame을 반환한다고 가정하고 출력 스키마 정의
        # 여기서는 문자열 리스트를 반환하므로, signature는 입력에 대해서만 정의하거나 출력 스키마를 신중히 설정
        # signature = infer_signature(X_train, pd.DataFrame(["SAMPLE_PREDICTION"]*len(X_train), columns=["prediction"])) 
        # KMeansPyfuncWrapper의 predict는 리스트를 반환하므로, 출력 시그니처를 리스트 형태로 맞추기 어려움.
        # 입력 시그니처만 명시하거나, predict가 DataFrame을 반환하도록 수정할 수 있음.
        # 여기서는 입력 시그니처만 명시적으로 사용.
        signature = infer_signature(input_example)

        pyfunc_model_wrapper = KMeansPyfuncWrapper()
        artifacts_for_pyfunc = {"model_bundle": LOCAL_MODEL_BUNDLE_PATH}
        
        print(f"MLflow에 Pyfunc 모델 로깅 시작: {MLFLOW_MODEL_ARTIFACT_PATH}")
        mlflow.pyfunc.log_model(
            artifact_path=MLFLOW_MODEL_ARTIFACT_PATH,
            python_model=pyfunc_model_wrapper,
            artifacts=artifacts_for_pyfunc,
            input_example=input_example,
            signature=signature,
            # conda_env=... # 필요시 conda 환경 지정
        )
        print(f"MLflow에 Pyfunc 모델 로깅 완료: {MLFLOW_MODEL_ARTIFACT_PATH}")
        mlflow.log_artifact(LOCAL_MODEL_BUNDLE_PATH, artifact_path="model_bundle_archive")

        print("학습 및 MLflow 로깅이 완료되었습니다.")

if __name__ == "__main__":
    train_centroid_model()