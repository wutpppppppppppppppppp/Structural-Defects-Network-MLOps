import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))  # Add the current directory to sys.path

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta

from callables.preprocess import get_transform, load_combined_dataset, split_dataset, balance_dataset, save_dataset_as_folder
from callables.check_img import check_new_images
from callables.log_model import log_model

def preprocess_data():
    print("Preprocessing and saving data...")

    # Load transforms
    train_transform, val_test_transform = get_transform()

    # Load dataset with validation/test transform
    full_dataset, class_names = load_combined_dataset(val_test_transform)

    # Split dataset
    train_set, val_set, test_set = split_dataset(full_dataset)

    # Load dataset again with training transform for balancing
    full_train_dataset, _ = load_combined_dataset(train_transform)
    balanced_train_set = balance_dataset(full_train_dataset)

    # Save to artifact folder
    save_dataset_as_folder(balanced_train_set, "artifact_folder", "train", class_names)
    save_dataset_as_folder(val_set, "artifact_folder", "val", class_names)
    save_dataset_as_folder(test_set, "artifact_folder", "test", class_names)

def train_model():
    print("Training model... (mock)")
    Variable.set("last_retrain_time", datetime.utcnow().isoformat())

default_args = {
    'owner': 'Blabla',
    'retries': 1,
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

    check_data = PythonOperator(
        task_id='check_new_data_from_cloudinary',
        python_callable=check_new_images
    )

    preprocess = PythonOperator(
        task_id='merge_datasets',
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

    end = DummyOperator(task_id='end')

    start >> check_data >> preprocess >> train >> log >> end
