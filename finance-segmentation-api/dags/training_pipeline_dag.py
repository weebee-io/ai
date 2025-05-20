from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# DAG 기본 인수 정의
default_args = {
    'owner': 'airflow', # DAG의 소유자
    'depends_on_past': False, # 이전 DAG 실행이 성공해야 현재 DAG 실행이 가능한지 여부
    'email_on_failure': False, # 실패 시 이메일 알림 여부
    'email_on_retry': False, # 재시도 시 이메일 알림 여부
    'retries': 1, # 실패 시 재시도 횟수
    'retry_delay': pendulum.duration(minutes=5), # 재시도 간격
}

# finance-segmentation-api 프로젝트의 절대 경로
# Airflow가 실행되는 환경에 따라 이 경로가 정확해야 합니다.
PROJECT_ROOT_PATH = "/Users/henry/Desktop/final-project/ai/finance-segmentation-api"
# 실행할 학습 스크립트 경로
TRAIN_SCRIPT_PATH = f"{PROJECT_ROOT_PATH}/train_centroid_model.py"
# Python 가상환경 경로 (Memory에 따라 venv_mlflow 사용)
# Airflow 실행 환경에서 이 가상환경에 접근 가능해야 합니다.
PYTHON_EXECUTABLE_PATH = f"{PROJECT_ROOT_PATH}/venv_mlflow/bin/python3"


with DAG(
    dag_id='centroid_model_training_pipeline', # DAG의 고유 ID
    default_args=default_args,
    description='Centroid model training pipeline using MLflow and Airflow',
    # schedule_interval='@daily', # 매일 실행 (또는 None, 또는 cron 표현식 "0 0 * * *")
    schedule_interval=None,
    start_date=days_ago(1), # DAG 시작 날짜 (어제부터 시작)
    catchup=False, # DAG가 처음 활성화될 때 과거의 스케줄되지 않은 실행을 실행할지 여부
    tags=['ml', 'training', 'mlflow', 'finance-segmentation'], # DAG에 태그 추가
) as dag:
    # 학습 스크립트 실행을 위한 BashOperator 정의
    run_training_script = BashOperator(
        task_id='run_train_centroid_model_script',
        # (venv_mlflow) 가상환경의 python3로 train_centroid_model.py 실행
        # 스크립트가 참조하는 경로(데이터, 모델 저장 등)가 올바르게 해석되도록executor = SequentialExecutor
        # 스크립트가 위치한 디렉토리에서 실행되도록 cwd 설정이 중요합니다.
        bash_command=f"cd {PROJECT_ROOT_PATH} && {PYTHON_EXECUTABLE_PATH} {TRAIN_SCRIPT_PATH}",
        cwd=PROJECT_ROOT_PATH, # 작업 디렉토리 설정
        doc_md="""
        #### Run Training Script Task
        This task executes the `train_centroid_model.py` script using the specified Python virtual environment.
        It changes the current working directory to the project root to ensure correct path resolution for data and model files.
        MLflow logging within the script will create a new run under the 'Centroid Model Training' experiment.
        """
    )

    # 작업 순서 (현재는 task가 하나만 있음)
    # run_training_script 
    # (추후 다른 task 추가 시 >> 연산자로 연결: task1 >> task2)
