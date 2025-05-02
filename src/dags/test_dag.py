from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import callables
from airflow.models import Variable
from datetime import datetime

def preprocess_data():
    print("Preprocessing data... (mock)")

def train_model():
    print("Training model... (mock)")
    Variable.set("last_retrain_time", datetime.utcnow().isoformat())

default_args = {
    'owner': 'concrete_structure',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='crack_cnn_training_pipeline',
    default_args=default_args,
    description='Retrain model when enough new data is available',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    start = DummyOperator(task_id='start')

    check_data = PythonOperator(
        task_id='check_new_data_from_cloudinary',
        python_callable=callables.check_new_images
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    log = PythonOperator(
        task_id='log_model',
        python_callable=callables.log_model
    )

    end = DummyOperator(task_id='end')

    start >> check_data >> preprocess >> train >> log >> end