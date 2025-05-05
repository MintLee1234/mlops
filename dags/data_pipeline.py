from airflow import DAG
from airflow.operators.python import PythonOperator
from component.data_transform import DataTransformation
import pandas as pd
import datetime as dt
import mlflow
import os
import joblib

default_args = {
    'owner': 'minhle',
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1)
}

def save_csv_append_or_create(df, path):
    df.to_csv(path, mode='a' if os.path.exists(path) else 'w', index=False, header=not os.path.exists(path))

def crawl_data(**kwargs):
    today_str = dt.date.today().strftime('%Y-%m-%d')
    df = pd.read_csv(kwargs['crawl_source_path'])
    df = df[df['joining_date'] == today_str].drop(columns=['churn_risk_score'], errors='ignore')
    save_csv_append_or_create(df, kwargs['bronze_data'])
    print("✅ Crawled data successfully")

def transform_data(**kwargs):
    today_str = dt.date.today().strftime('%Y-%m-%d')
    df = pd.read_csv(kwargs['bronze_data'])
    df_today = df[df['joining_date'] == today_str]
    
    if df_today.empty:
        print("⚠️ Không có dữ liệu để transform.")
        return
    
    transformer = DataTransformation()
    silver_df = transformer.transform_data(df_today)
    silver_df['churn_risk_score'] = pd.NA
    save_csv_append_or_create(silver_df, kwargs['silver_table'])

    print("✅ Transform data successfully")
    print(silver_df.head())

def daily_prediction(**kwargs):
    today_str = dt.date.today().strftime('%Y-%m-%d')
    df = pd.read_csv(kwargs['silver_table'])
    df_today = df[df['joining_date'] == today_str]
    
    if df_today.empty:
        print("⚠️ Không có dữ liệu người dùng mới cho hôm nay.")
        return

    user_id = df_today['user_id'].reset_index(drop=True)
    joining_date = df_today['joining_date'].reset_index(drop=True)
    df_today = df_today.drop(columns=['user_id', 'churn_risk_score'], errors='ignore')

    try:
        with open('preprocessors/preprocessor_versions.txt') as f:
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
        print(f"❌ Lỗi load model từ MLflow: {e}")
        return

    df_transformed = preprocessor.transform(df_today)
    predictions = model.predict(df_transformed)

    results = pd.DataFrame({
        'user_id': user_id,
        'joining_date': joining_date,
        'prediction': predictions
    })

    save_csv_append_or_create(results, '/home/minhle/mlops/data/predictions.csv')
    print("✅ Daily prediction completed successfully")

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
            'crawl_source_path': '/home/minhle/mlops/data/web_churn_raw.csv',
            'bronze_data': '/home/minhle/mlops/data/bronze_data.csv'
        },
    )

    task1 = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        op_kwargs={
            'silver_table': '/home/minhle/mlops/data/silver_data.csv',
            'bronze_data': '/home/minhle/mlops/data/bronze_data.csv'
        },
    )

    task2 = PythonOperator(
        task_id='daily_prediction',
        python_callable=daily_prediction,
        op_kwargs={
            'silver_table': '/home/minhle/mlops/data/silver_data.csv',
            'model_id_log': '/home/minhle/mlops/last_best_run_id.txt'
        },
    )

    task0 >> task1 >> task2
