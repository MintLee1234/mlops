from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from component.data_transform import DataTransformation
import pandas as pd

default_args = {
    'owner': 'minhle',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}


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
    X_train, X_test, y_train, y_test = data_transformation.train_val_test_splitting()
    X_train_transformed, X_test_transformed, y_train, y_test = data_transformation.initiate_data_transformation(
        X_train, X_test, y_train, y_test
    )
with DAG(
    default_args=default_args,
    dag_id='data_pipeline_v01',
    description='Our first dag using python operator',
    start_date=datetime(2024, 10, 6),
    schedule_interval='@daily',
    catchup=False,
) as dag:
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

    task1 >> task2