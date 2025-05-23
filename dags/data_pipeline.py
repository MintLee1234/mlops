from airflow import DAG
from airflow.operators.python import PythonOperator
from component.data_transform import DataTransformation
from component.data_ingestion import PostgresDataIngestor
import pandas as pd
import numpy as np
import datetime as dt
import mlflow
import joblib
from dotenv import load_dotenv
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monitoring.metric_server import (prediction_class_0, 
                                      prediction_class_1, 
                                      daily_crawled_count)


load_dotenv()


# DB Config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DATABASE"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PW")
}

default_args = {
    'owner': 'minhle',
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1)
}


def crawl_data(**kwargs):
    today_str = dt.date.today().strftime('%Y-%m-%d')
    df = pd.read_csv(kwargs['crawl_source_path'])
    df_today = df[df['joining_date'] == today_str].drop(columns=['churn_risk_score'], errors='ignore')

    daily_crawled_count.set(len(df_today))

    if df_today.empty:
        print("⚠️ Không có dữ liệu để crawl.")
        return

    ingestor = PostgresDataIngestor(**DB_CONFIG)
    ingestor.ingest_data(table_name='bronze_data', data_source=df_today, mode='append')


def transform_data():
    today_str = dt.date.today().strftime('%Y-%m-%d')

    ingestor = PostgresDataIngestor(**DB_CONFIG)
    df = ingestor.read_table('bronze_data')
    df_today = df[df['joining_date'] == today_str]

    if df_today.empty:
        print("⚠️ Không có dữ liệu để transform.")
        return

    transformer = DataTransformation()
    silver_df = transformer.transform_data(df_today)
    silver_df['churn_risk_score'] = pd.NA

    ingestor.ingest_data(table_name='silver_data', data_source=silver_df, mode='append')
    print("✅ Transform completed")


def daily_prediction(**kwargs):
    today_str = dt.date.today().strftime('%Y-%m-%d')

    ingestor = PostgresDataIngestor(**DB_CONFIG)
    df = ingestor.read_table('silver_data')
    df_today = df[df['joining_date'] == today_str]

    if df_today.empty:
        print("⚠️ Không có dữ liệu để dự đoán.")
        return

    user_id = df_today['user_id'].reset_index(drop=True)
    joining_date = df_today['joining_date'].reset_index(drop=True)
    df_today = df_today.drop(columns=['user_id', 'churn_risk_score'], errors='ignore')

    try:
        with open(kwargs['preprocessor_log']) as f:
            preprocessor_path = f.readlines()[-1].strip().split(' - ')[-1]
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        print(f"❌ Lỗi load preprocessor: {e}")
        return

    try:
        with open(kwargs['model_id_log']) as f:
            model_id = f.readlines()[-1].strip().split(' - ')[-1]
        model = mlflow.pyfunc.load_model(f"runs:/{model_id}/model")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    df_transformed = preprocessor.transform(df_today)
    predictions = model.predict(df_transformed)

    unique, counts = np.unique(predictions, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)
    prediction_class_0.set(count_dict.get(0, 0))
    prediction_class_1.set(count_dict.get(1, 0))

    results = pd.DataFrame({
        'user_id': user_id,
        'joining_date': joining_date,
        'prediction': predictions
    })

    ingestor.ingest_data(table_name='predictions', data_source=results, mode='append')
    print("✅ Predictions inserted")


with DAG(
    default_args=default_args,
    dag_id='data_pipeline_v01',
    description='Daily prediction pipeline',
    start_date=dt.datetime(2024, 10, 6),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    task0 = PythonOperator(
        task_id='crawl_data',
        python_callable=crawl_data,
        op_kwargs={
            'crawl_source_path': '/home/minhle/mlops/data/web_churn_raw.csv'
        },
    )

    task1 = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )

    task2 = PythonOperator(
        task_id='daily_prediction',
        python_callable=daily_prediction,
        op_kwargs={
            'preprocessor_log': '/home/minhle/mlops/preprocessors/preprocessor_versions.txt',
            'model_id_log': '/home/minhle/mlops/last_best_run_id.txt'
        },
    )

    task0 >> task1 >> task2
