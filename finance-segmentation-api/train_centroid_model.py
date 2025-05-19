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

# --- MLflow 설정 및 기본 파라미터 ---
MLFLOW_EXPERIMENT_NAME = "Centroid Model Training"

# !!! 중요: 이 경로를 실제 로컬 데이터 파일 경로로 수정하세요 !!!
DATA_PATH = '/Users/henry/Desktop/final-project/ai/masterpiece.csv' 

# 학습된 모델 및 아티팩트 저장 경로
BASE_OUTPUT_DIR = "models/mlflow_trained"
OUTPUT_MODEL_FILENAME = "centroid_weighted_model.joblib"
OUTPUT_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, OUTPUT_MODEL_FILENAME)

# 학습 파라미터
TEST_SIZE = 0.3
RANDOM_STATE = 42
KMEANS_N_CLUSTERS = 3
KMEANS_N_INIT = 'auto' # 권장값으로 변경 (또는 10)
KMEANS_MAX_ITER = 100 # 원본 코드에서는 100이었음, KMeans 기본값은 300

# 고정된 피처 이름 및 가중치 (inspect_model_components.py와 일관성 유지)
FEATURE_NAMES = [
    'essential_pct', 'discretionary_pct', 'risk_profile_score',
    'complex_product_flag', 'digital_engagement', 'is_married',
    'spend_volatility', 'sav_inv_ratio'
]
WEIGHTS = np.array([2.0, -2.0, -1.5, 1.0, 1.0, 0.1, 1.0, 2.0])

# 거리 함수 정의
def euclid(Xp, centers, w):
    return np.sqrt(((Xp[:,None,:] - centers[None,:,:])**2 * np.abs(w)[None,None,:]).sum(axis=2))

def manhattan(Xp, centers, w):
    return (np.abs(Xp[:,None,:] - centers[None,:,:]) * np.abs(w)[None,None,:]).sum(axis=2)

def chebyshev(Xp, centers, w):
    return (np.abs(Xp[:,None,:] - centers[None,:,:]) * np.abs(w)[None,None,:]).max(axis=2)

def minkowski_p3(Xp, centers, w):
    return ((np.abs(Xp[:,None,:] - centers[None,:,:])**3 * np.abs(w)[None,None,:]).sum(axis=2))**(1/3)

DISTANCE_METRICS = {
    'euclidean': euclid,
    'manhattan': manhattan,
    'chebyshev': chebyshev,
    'minkowski_p3': minkowski_p3
}

def train_and_evaluate():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_param("run_id", run_id)

        # --- 1) 데이터 로드 & 피처 준비 ---
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            print(f"Error: Data file not found at {DATA_PATH}. Please update DATA_PATH variable.")
            mlflow.log_param("data_loading_status", "failed_file_not_found")
            return
        
        X = df[FEATURE_NAMES]
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("features_used", json.dumps(FEATURE_NAMES))
        mlflow.log_metric("num_samples_total", len(df))
        mlflow.log_metric("num_features", len(FEATURE_NAMES))

        # --- 2) 스케일링 ---
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        # 스케일러 저장 (MLflow 아티팩트 & 로컬)
        mlflow.sklearn.log_model(scaler, "scaler_model")
        print(f"Scaler saved to MLflow artifacts (scaler_model).")

        # --- 3) Train/Test 분할 ---
        X_train, X_test = train_test_split(Xs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state_split", RANDOM_STATE)
        mlflow.log_metric("num_train_samples", len(X_train))
        mlflow.log_metric("num_test_samples", len(X_test))

        # --- 4) 훈련용 KMeans로 centroids 생성 ---
        km = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE, 
                    n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER)
        km.fit(X_train)
        centroids = km.cluster_centers_
        # KMeans 모델 저장 (MLflow 아티팩트 & 로컬)
        mlflow.sklearn.log_model(km, "kmeans_model")
        print(f"KMeans model saved to MLflow artifacts (kmeans_model).")
        mlflow.log_param("kmeans_n_clusters", KMEANS_N_CLUSTERS)
        mlflow.log_param("kmeans_random_state", RANDOM_STATE)
        mlflow.log_param("kmeans_n_init", KMEANS_N_INIT)
        mlflow.log_param("kmeans_max_iter", KMEANS_MAX_ITER)
        np.save("centroids.npy", centroids)
        mlflow.log_artifact("centroids.npy", "model_components")
        os.remove("centroids.npy") # 임시 파일 삭제
        print(f"Centroids saved to MLflow artifacts (model_components/centroids.npy).")

        # --- 5) 테스트용 “정답” 레이블 (KMeans로 예측) ---
        y_test_labels = km.predict(X_test)

        # --- 6) 가중치 로깅 ---
        mlflow.log_param("weights", json.dumps(WEIGHTS.tolist()))
        np.save("weights.npy", WEIGHTS)
        mlflow.log_artifact("weights.npy", "model_components")
        os.remove("weights.npy") # 임시 파일 삭제
        print(f"Weights saved to MLflow artifacts (model_components/weights.npy).")

        # --- 7) 각 거리 함수별 성능 비교 ---
        results_accuracy = {}
        best_metric_name = None
        best_accuracy = -1.0

        print("\n--- Evaluating Distance Metrics ---")
        for name, fn in DISTANCE_METRICS.items():
            dists = fn(X_test, centroids, WEIGHTS)
            y_pred = np.argmin(dists, axis=1)
            acc = accuracy_score(y_test_labels, y_pred)
            results_accuracy[name] = acc
            mlflow.log_metric(f"accuracy_{name}", acc)

            print(f"\n=== {name.upper()} ===")
            print(f"Accuracy: {acc:.4f}")
            
            # Confusion Matrix 및 Classification Report 저장 및 로깅
            cm = confusion_matrix(y_test_labels, y_pred)
            cr = classification_report(y_test_labels, y_pred)
            print("Confusion Matrix:\n", cm)
            print("Classification Report:\n", cr)
            
            report_content = f"Metric: {name.upper()}\nAccuracy: {acc:.4f}\n\nConfusion Matrix:\n{cm}\n\nClassification Report:\n{cr}"
            report_filename = f"report_{name}.txt"
            with open(report_filename, "w") as f:
                f.write(report_content)
            mlflow.log_artifact(report_filename, "reports")
            os.remove(report_filename) # 임시 파일 삭제
            print(f"Report for {name} saved to MLflow artifacts (reports/{report_filename}).")

            if acc > best_accuracy:
                best_accuracy = acc
                best_metric_name = name
        
        mlflow.log_param("best_distance_metric", best_metric_name)
        mlflow.log_metric("best_accuracy", best_accuracy)
        print(f"\n▶ Best metric: {best_metric_name} (Accuracy {best_accuracy:.4f})")

        # --- 8) 최종 모델 번들 저장 (inspect_model_components.py 호환) ---
        model_bundle = {
            'centroids': centroids,
            'weights': WEIGHTS,
            'scaler': scaler,
            'best_distance_metric': best_metric_name # 추가 정보로 포함
        }
        # 로컬에 저장
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        joblib.dump(model_bundle, OUTPUT_MODEL_PATH)
        print(f"\nFinal model bundle saved locally to: {OUTPUT_MODEL_PATH}")
        # MLflow에 아티팩트로 로깅
        mlflow.log_artifact(OUTPUT_MODEL_PATH, "model_bundle")
        print(f"Final model bundle also logged to MLflow artifacts (model_bundle/{OUTPUT_MODEL_FILENAME}).")

        # --- 9) 학습 스크립트 자체를 아티팩트로 로깅 ---
        mlflow.log_artifact(__file__, "code")
        print(f"Training script '{__file__}' logged to MLflow artifacts (code/).")

        print(f"\nMLflow Run completed. Check MLflow UI for experiment '{MLFLOW_EXPERIMENT_NAME}' and run ID '{run_id}'.")

if __name__ == "__main__":
    train_and_evaluate()
