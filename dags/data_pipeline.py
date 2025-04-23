from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from component.data_transform import DataTransformation
import pandas as pd
import numpy as np
import datetime

default_args = {
    'owner': 'minhle',
    'retries': 5,
    'retry_delay': datetime.timedelta(minutes=5)
}

def crawl_data(**kwargs):
    try:
        df = pd.read_csv(kwargs['data_crawl_path'])
    except pd.errors.EmptyDataError:
        df = pd.DataFrame() 
    craw_source = pd.read_csv(kwargs['crawl_source_path'])
    weekday_number = datetime.date.today().weekday()
    splits = np.array_split(craw_source, 7)
    today_split = splits[weekday_number]
    df = pd.concat([df, today_split], ignore_index=True)
    df = df.drop_duplicates(keep='last')
    df.to_csv(kwargs['data_crawl_path'], index=False)
    print("✅ Crawled data suscessfully")


def get_data(**kwargs):
    file_path = kwargs['file_path']
    df = pd.read_csv(file_path)
    print("✅ Loaded data:")
    print(df.head())
    # Truyền dữ liệu qua XCom (dạng JSON serializable)
    return df.to_json()

# Task 2: Nhận dữ liệu từ task 1 qua XCom và xử lý
def transform_data(**kwargs):
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='get_data')  # lấy kết quả từ task1
    df = pd.read_json(df_json)
    print("✅ Got data from get_data:")
    print(df.head())

    data_transformation = DataTransformation(df)
    data_transformation.initiate_data_transformation()

with DAG(
    default_args=default_args,
    dag_id='data_pipeline_v01',
    description='Our first dag using python operator',
    start_date = datetime.datetime(2024, 10, 6),
    schedule_interval='@daily',
    catchup=False,
) as dag:
    task0 = PythonOperator(
        task_id='crawl_data',
        python_callable=crawl_data,
        op_kwargs={'data_crawl_path': '/home/minhle/mlops/data/data_crawl.csv',
                   'crawl_source_path': '/home/minhle/mlops/data/web_churn_raw.csv'},
        provide_context=True,
    )
    task1 = PythonOperator(
        task_id='get_data',
        python_callable=get_data,
        op_kwargs={'file_path': '/home/minhle/mlops/data/web_churn_raw.csv'},
        provide_context=True,
    )
    task2 = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True,
    )

    task0 >> task1 >> task2