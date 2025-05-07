import sys
import os
import logging

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from
from airflow.models import Variable
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__)))  # Add the current directory to sys.path
from callables.preprocess import preprocess_data
from callables.check_img import check_new_images
from callables.log_model import log_model
from callables.s3_to_csv import generate_dataset_csv

logger = logging.getLogger(__name__)

def train_model():
    print("Training model... (mock)")
    Variable.set("last_retrain_time", datetime.utcnow().isoformat())

default_args = {
    'owner': 'Blabla',
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='crack_cnn_training_pipeline',
    default_args=default_args,
    description='Retrain model when enough new data is available',
    schedule_interval='@monthly',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['concrete', 'dag'],
) as dag:

    start = DummyOperator(task_id='start')

    # return s3_to_csv if there are more than 10 imgs, stop_no_data otherwise
    check_data = BranchPythonOperator(
        task_id='check_new_data_from_cloudinary',
        python_callable=check_new_images,
        provide_context=True,  # optional depending on your version

    )
    
    retrieve_new_images_url = PythonOperator(
        task_id='s3_to_csv',
        python_callable = generate_dataset_csv,
    )
    
    # Define dummy task if not enough data
    skip_training = DummyOperator(task_id='stop_no_data')

    preprocess = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    log = PythonOperator(
        task_id='log_model',
        python_callable=log_model
    )

    end = DummyOperator(task_id='end', trigger_rule='none_failed_min_one_success')

    # DAG flow with branching
    start >> check_data
    check_data >> retrieve_new_images_url >> preprocess >> train >> log >> end
    check_data >> skip_training >> log >> end