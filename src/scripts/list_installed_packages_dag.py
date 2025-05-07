from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def list_installed_packages():
    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    with open('/home/santitham/airflow/dags/Structural-Defects-Network-MLOps/env_package_list.txt', 'w') as f:
        f.write(result.stdout)

default_args = {
    'owner': 'Blabla',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='concrete_structure_list_airflow_env_packages',
    default_args=default_args,
    schedule_interval=None,  # Run manually
    catchup=False,
    tags=['concrete','utility'],
) as dag:

    list_packages_task = PythonOperator(
        task_id='list_installed_packages',
        python_callable=list_installed_packages,
    )
