from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator

# DAG 기본 인수 정의
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
}

# finance-segmentation-api 프로젝트의 절대 경로
PROJECT_ROOT_PATH = "/Users/henry/Desktop/final-project/ai/finance-segmentation-api"
# 실행할 학습 스크립트 경로
TRAIN_SCRIPT_PATH = f"{PROJECT_ROOT_PATH}/train_centroid_model.py"
# Python 가상환경 경로
PYTHON_EXECUTABLE_PATH = f"{PROJECT_ROOT_PATH}/venv_mlflow/bin/python3"


with DAG(
    dag_id='centroid_model_training_pipeline_csv',
    default_args=default_args,
    description='Centroid model training pipeline using MLflow, Airflow, and CSV input. Runs every 5 minutes for testing.',
    schedule= '*/30 * * * *',  # 매 30분마다 실행
    start_date=pendulum.today('UTC'), # 오늘 UTC 자정부터 시작
    catchup=False,
    tags=['ml', 'training', 'mlflow', 'finance-segmentation', 'csv', 'test'],
) as dag:
    run_training_script = BashOperator(
        task_id='run_train_centroid_model_script_csv',
        bash_command=f"export PYTHONFAULTHANDLER=true; export no_proxy='*'; cd {PROJECT_ROOT_PATH} && {PYTHON_EXECUTABLE_PATH} {TRAIN_SCRIPT_PATH}",
        cwd=PROJECT_ROOT_PATH,
        doc_md="""
        #### Run Training Script Task (CSV Input)
        This task executes the `train_centroid_model.py` script (modified for CSV input)
        using the specified Python virtual environment (`venv_mlflow`).
        It changes the current working directory to the project root to ensure correct path resolution 
        for `masterpiece.csv` and other model files.
        MLflow logging within the script will create a new run under the 'Centroid Model Training (CSV Input)' experiment.
        """
    )

    # 작업 순서는 현재 단일 작업
    # run_training_script
