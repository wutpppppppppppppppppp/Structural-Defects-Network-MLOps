from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os

def install_requirements():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    requirements_path = os.path.join(current_dir, 'requirements.txt')
    subprocess.run(['pip', 'install', '-r', requirements_path], check=True)

default_args = {
    'owner': 'Blabla',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='concrete_structure_install_requirements',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['concrete', 'utility'],
) as dag:

    install_task = PythonOperator(
        task_id='install_requirements_task',
        python_callable=install_requirements,
    )
