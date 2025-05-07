from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def install_opencvpython():
    subprocess.run(['pip', 'install', 'opencv-python-headless'], check=True)

default_args = {
    'owner': 'Blabla',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='concrete_structure_install_opencv-python',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['concrete','utility'],
) as dag:

    install_task = PythonOperator(
        task_id='install_opencv-python_task',
        python_callable=install_opencvpython,
    )
