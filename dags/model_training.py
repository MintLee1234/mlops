from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from component.LGBM_Trainer import LGBM_Trainer
from component.XGB_Trainer import XGB_Trainer
from component.model_evaluation import ModelEvaluation
from component.data_transform import DataTransformation
from component.data_ingestion import PostgresDataIngestor
from psycopg2 import OperationalError
import shutil
from dotenv import load_dotenv
import os


load_dotenv()

default_args = {
    'owner': 'minhle',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DATABASE"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PW")
}


def get_last_month_range():
    today = date.today()
    first_day_this_month = today.replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    start = last_day_last_month.replace(day=1)
    end = last_day_last_month
    return str(start), str(end)


def monthly_crawl_data(**kwargs):
    ti = kwargs['ti']
    df = pd.read_csv(kwargs['crawl_source_path'])
    df['joining_date'] = pd.to_datetime(df['joining_date'])

    start, end = get_last_month_range()
    print(f"📅 Crawl label dữ liệu từ {start} đến {end}")

    filtered = df[(df['joining_date'] >= start) & (df['joining_date'] <= end)][['user_id', 'joining_date', 'churn_risk_score']]
    label_path = '/tmp/monthly_labels.csv'
    filtered.to_csv(label_path, index=False)
    ti.xcom_push(key='label_path', value=label_path)


def update_gold_data(**kwargs):
    ti = kwargs['ti']
    start, end = get_last_month_range()

    try:
        pg_ingestor = PostgresDataIngestor(**DB_CONFIG)
        df = pg_ingestor.read_table('silver_data')
    except OperationalError:
        print("⚠️ Không thể kết nối PostgreSQL. Dùng file local silver_data.csv thay thế.")
        df = pd.read_csv('/home/minhle/mlops/data/silver_data.csv')

    filtered = df[(df['joining_date'] >= start) & (df['joining_date'] <= end)].copy()

    label_path = ti.xcom_pull(task_ids='monthly_crawl_data', key='label_path')
    if label_path:
        labels = pd.read_csv(label_path)

        filtered = filtered.drop(columns=['churn_risk_score'], errors='ignore')
        filtered = pd.merge(filtered, labels[['user_id', 'churn_risk_score']], on='user_id', how='left')

        try:
            pg_ingestor.ingest_data('gold_data', filtered, mode='append')
        except Exception:
            print("⚠️ Không thể ghi vào PostgreSQL. Ghi vào gold_data.csv")
            filtered.to_csv('/home/minhle/mlops/data/gold_data.csv', index=False)


def monthly_evaluation(**kwargs):
    ti = kwargs['ti']
    start, end = get_last_month_range()

    try:
        pg_ingestor = PostgresDataIngestor(**DB_CONFIG)
        df = pg_ingestor.read_table('predictions')
    except OperationalError:
        print("⚠️ Không thể kết nối PostgreSQL. Dùng predictions_local.csv thay thế.")
        df = pd.read_csv('/home/minhle/mlops/data/predictions.csv')

    filtered = df[(df['joining_date'] >= start) & (df['joining_date'] <= end)]
    print('filter', filtered.shape)
    label_path = ti.xcom_pull(task_ids='monthly_crawl_data', key='label_path')
    y_test = pd.read_csv(label_path)['churn_risk_score'] if label_path else []
    y_pred = filtered['prediction']
    print(y_test.shape, y_pred.shape)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"✅ Evaluation completed: Accuracy={acc:.4f}, ROC AUC={roc_auc:.4f}")
    ti.xcom_push(key='accuracy', value=acc)


def check_accuracy_branch(**kwargs):
    acc = kwargs['ti'].xcom_pull(task_ids='monthly_evaluation', key='accuracy')
    return 'data_prepare' if acc and acc < 0.95 else 'skip_tasks'


def data_prepare():
    try:
        pg_ingestor = PostgresDataIngestor(**DB_CONFIG)
        df = pg_ingestor.read_table('gold_data')
    except OperationalError:
        print("⚠️ Không thể kết nối PostgreSQL. Dùng gold_data_local.csv thay thế.")
        df = pd.read_csv('/home/minhle/mlops/data/gold_data.csv')

    df.drop(columns=['user_id', 'joining_date'], inplace=True)
    DataTransformation().initiate_data_transformation(df)


def _train_model(trainer_class, **kwargs):
    X_train = pd.read_csv(kwargs['X_train_transformed_file_path'])
    y_train = pd.read_csv(kwargs['y_train_file_path']).squeeze()
    X_test = pd.read_csv(kwargs['X_test_transformed_file_path'])
    y_test = pd.read_csv(kwargs['y_test_file_path']).squeeze()

    result = trainer_class().fit(X_train, y_train, X_test, y_test)
    kwargs['ti'].xcom_push(key=f"{trainer_class.__name__.lower()}_run_id", value=result["run_id"])


def LGBM_trainer(**kwargs): _train_model(LGBM_Trainer, **kwargs)
def XGB_trainer(**kwargs): _train_model(XGB_Trainer, **kwargs)


def model_evaluation(**kwargs):
    ti = kwargs['ti']
    client = mlflow.tracking.MlflowClient()

    lgbm_run_id = ti.xcom_pull(task_ids='lgbm_train', key='lgbm_trainer_run_id')
    xgb_run_id = ti.xcom_pull(task_ids='xgb_train', key='xgb_trainer_run_id')

    if not lgbm_run_id or not xgb_run_id:
        print("❌ Missing run_id(s)")
        return

    lgbm_auc = float(client.get_metric_history(lgbm_run_id, "LGBM_auc")[-1].value)
    xgb_auc = float(client.get_metric_history(xgb_run_id, "XGB_auc")[-1].value)
    best_run_id = lgbm_run_id if lgbm_auc > xgb_auc else xgb_run_id

    with open("last_best_run_id.txt", "a") as f:
        f.write(f"{datetime.now()} - {best_run_id}\n")
    original_file_path = '/home/minhle/mlops/last_best_run_id.txt'
    destination_file_path = '/home/minhle/fastapi_model_serving/last_best_run_id.txt'
    try:
        shutil.copy2(original_file_path, destination_file_path)
        print(f"Đã sao chép {original_file_path} tới {destination_file_path}")
    except Exception as e:
        print(f"Lỗi khi sao chép file: {e}")
    kwargs["ti"].xcom_push(key="best_run_id", value=best_run_id)

    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    X_test = pd.read_csv(kwargs['X_test_transformed_file_path'])
    y_test = pd.read_csv(kwargs['y_test_file_path']).squeeze()

    evaluator = ModelEvaluation(best_model)
    y_pred, y_proba = evaluator.predictions(X_test)
    metrics = evaluator.model_evaluation(y_test, y_pred, y_proba)
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics(metrics)
    print(f"Model evaluation complete. Metrics: {metrics}")

    kwargs["ti"].xcom_push(key="evaluation_metrics", value=metrics)

    return metrics


# -------------------- DAG --------------------
with DAG(
    dag_id='model_pipeline_v01',
    description='Train LGBM, XGB and evaluate best model',
    default_args=default_args,
    start_date=datetime(2024, 10, 6),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    task0 = PythonOperator(
        task_id='monthly_crawl_data', 
        python_callable=monthly_crawl_data,
        op_kwargs={
            'crawl_source_path': '/home/minhle/mlops/data/web_churn_raw.csv'
        }
    )

    task1 = PythonOperator(
        task_id='update_gold_data', 
        python_callable=update_gold_data,
    )

    task2 = PythonOperator(
        task_id='monthly_evaluation', 
        python_callable=monthly_evaluation,
    )

    branch = BranchPythonOperator(task_id='check_accuracy_branch', python_callable=check_accuracy_branch)
    skip = DummyOperator(task_id='skip_tasks')

    task3 = PythonOperator(
        task_id='data_prepare', 
        python_callable=data_prepare,
    )

    task4 = PythonOperator(
        task_id='lgbm_train', 
        python_callable=LGBM_trainer,
        op_kwargs={
            f'{k}': f'/home/minhle/mlops/data/{v}' for k, v in {
                'X_train_transformed_file_path': 'X_train_transformed.csv',
                'y_train_file_path': 'y_train.csv',
                'X_test_transformed_file_path': 'X_test_transformed.csv',
                'y_test_file_path': 'y_test.csv',
            }.items()
        }
    )

    task5 = PythonOperator(
        task_id='xgb_train', 
        python_callable=XGB_trainer, 
        op_kwargs=task4.op_kwargs
    )

    task6 = PythonOperator(
        task_id='evaluate_model', 
        python_callable=model_evaluation,
        op_kwargs={
            'X_test_transformed_file_path': '/home/minhle/mlops/data/X_test_transformed.csv',
            'y_test_file_path': '/home/minhle/mlops/data/y_test.csv'
        }
    )

    # Dependencies
    task0 >> task1 >> task2 >> branch
    branch >> task3 >> [task4, task5] >> task6 
    branch >> skip