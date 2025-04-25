import pickle
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from component.LGBM_Trainer import LGBM_Trainer
from component.XGB_Trainer import XGB_Trainer
from component.model_evaluation import ModelEvaluation
import pandas as pd
import mlflow
import mlflow.sklearn

default_args = {
    'owner': 'minhle',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

def LGBM_trainer(**kwargs):
    X_train = pd.read_csv(kwargs['X_train_transformed_file_path'])
    y_train = pd.read_csv(kwargs['y_train_file_path']).squeeze()
    X_test = pd.read_csv(kwargs['X_test_transformed_file_path'])
    y_test = pd.read_csv(kwargs['y_test_file_path']).squeeze()
    result = LGBM_Trainer().fit(X_train, y_train, X_test, y_test)

    kwargs['ti'].xcom_push(key="lgbm_run_id", value=result["run_id"])


def XGB_trainer(**kwargs):
    X_train = pd.read_csv(kwargs['X_train_transformed_file_path'])
    y_train = pd.read_csv(kwargs['y_train_file_path']).squeeze()
    X_test = pd.read_csv(kwargs['X_test_transformed_file_path'])
    y_test = pd.read_csv(kwargs['y_test_file_path']).squeeze()
    result = XGB_Trainer().fit(X_train, y_train, X_test, y_test)

    kwargs['ti'].xcom_push(key="xgb_run_id", value=result["run_id"])

def model_evaluation(**kwargs):
    ti = kwargs['ti']

    # Lấy run_id của từng model
    lgbm_run_id = ti.xcom_pull(task_ids='lgbm_train', key='lgbm_run_id')
    xgb_run_id = ti.xcom_pull(task_ids='xgb_train', key='xgb_run_id')

    # Lấy AUC từ mỗi run
    client = mlflow.tracking.MlflowClient()
    lgbm_auc = float(client.get_metric_history(lgbm_run_id, "LGBM_auc")[-1].value)
    xgb_auc = float(client.get_metric_history(xgb_run_id, "XGB_auc")[-1].value)

    # So sánh và chọn model tốt hơn
    best_run_id = lgbm_run_id if lgbm_auc > xgb_auc else xgb_run_id
    if best_run_id == lgbm_run_id:
        best_model_path = f"runs:/{best_run_id}/LGBM_model"
    else:
        best_model_path = f"runs:/{best_run_id}/XGB_model"

    # Load model tốt nhất từ MLflow
    best_model = mlflow.sklearn.load_model(best_model_path)

    # Load tập test
    X_test = pd.read_csv(kwargs['X_test_transformed_file_path'])
    y_test = pd.read_csv(kwargs['y_test_file_path']).squeeze()

    # Đánh giá
    evaluator = ModelEvaluation(best_model)
    y_pred, y_proba = evaluator.predictions(X_test)
    acc, prec, rec, f1, roc_auc, pr_auc = evaluator.model_evaluation(y_test, y_pred, y_proba)

    print(f"Evaluation results:\nAccuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
          f"F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
with DAG(
    default_args=default_args,
    dag_id='model_pipeline_v01',
    description='Train LGBM, XGB and evaluate best model',
    start_date=datetime(2024, 10, 6),
    schedule_interval='@daily',
    catchup=False,
) as dag:
    task1 = PythonOperator(
        task_id='lgbm_train',
        python_callable=LGBM_trainer,
        op_kwargs={
            'X_train_transformed_file_path': '/home/minhle/mlops/data/X_train_transformed.csv',
            'y_train_file_path': '/home/minhle/mlops/data/y_train.csv',
            'X_test_transformed_file_path': '/home/minhle/mlops/data/X_test_transformed.csv',
            'y_test_file_path': '/home/minhle/mlops/data/y_test.csv',
        },
    )

    task2 = PythonOperator(
        task_id='xgb_train',
        python_callable=XGB_trainer,
        op_kwargs={
            'X_train_transformed_file_path': '/home/minhle/mlops/data/X_train_transformed.csv',
            'y_train_file_path': '/home/minhle/mlops/data/y_train.csv',
            'X_test_transformed_file_path': '/home/minhle/mlops/data/X_test_transformed.csv',
            'y_test_file_path': '/home/minhle/mlops/data/y_test.csv',
        },
    )

    task3 = PythonOperator(
        task_id='evaluate_model',
        python_callable=model_evaluation,
        op_kwargs={
            'X_test_transformed_file_path': '/home/minhle/mlops/data/X_test_transformed.csv',
            'y_test_file_path': '/home/minhle/mlops/data/y_test.csv'
        },
    )

    [task1, task2] >> task3
