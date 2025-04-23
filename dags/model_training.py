import pickle
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from component.LGBM_Trainer import LGBM_Trainer
from component.XGB_Trainer import XGB_Trainer
from component.model_evaluation import ModelEvaluation
import pandas as pd

default_args = {
    'owner': 'minhle',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

def LGBM_trainer(**kwargs):
    X_train = pd.read_csv(kwargs['X_train_transformed_file_path'])
    y_train = pd.read_csv(kwargs['y_train_file_path']).squeeze()
    model1, auc1 = LGBM_Trainer().fit(X_train, y_train)

    # Save model to file
    with open('/tmp/model1.pkl', 'wb') as f:
        pickle.dump(model1, f)

    return {'model_path': '/tmp/model1.pkl', 'auc': auc1}

def XGB_trainer(**kwargs):
    X_train = pd.read_csv(kwargs['X_train_transformed_file_path'])
    y_train = pd.read_csv(kwargs['y_train_file_path']).squeeze()
    model2, auc2 = XGB_Trainer().fit(X_train, y_train)

    with open('/tmp/model2.pkl', 'wb') as f:
        pickle.dump(model2, f)

    return {'model_path': '/tmp/model2.pkl', 'auc': auc2}

def model_evaluation(**kwargs):
    ti = kwargs['ti']
    result1 = ti.xcom_pull(task_ids='lgbm_train')
    result2 = ti.xcom_pull(task_ids='xgb_train')

    # So sánh AUC và chọn model tốt hơn
    best_model_path = result1['model_path'] if result1['auc'] > result2['auc'] else result2['model_path']
    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)

    X_test = pd.read_csv(kwargs['X_test_transformed_file_path'])
    y_test = pd.read_csv(kwargs['y_test_file_path']).squeeze()

    evaluator = ModelEvaluation(model)
    y_pred, y_proba = evaluator.predictions(X_test)
    acc, prec, rec, f1, roc_auc, pr_auc = evaluator.model_evaluation(y_test, y_pred, y_proba)

    print(f"Evaluation results:\nAccuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

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
            'y_train_file_path': '/home/minhle/mlops/data/y_train.csv'
        },
    )

    task2 = PythonOperator(
        task_id='xgb_train',
        python_callable=XGB_trainer,
        op_kwargs={
            'X_train_transformed_file_path': '/home/minhle/mlops/data/X_train_transformed.csv',
            'y_train_file_path': '/home/minhle/mlops/data/y_train.csv'
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
