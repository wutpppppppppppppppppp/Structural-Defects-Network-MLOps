from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pprint import pprint

def save_sys_path():
    with open("/home/santitham/airflow/dags/Structural-Defects-Network-MLOps/airflow_sys_path.txt", "w") as f:
        for path in sys.path:
            f.write(path + "\n")
    print("✔ sys.path saved to /home/santitham/airflow/dags/Structural-Defects-Network-MLOps/airflow_sys_path.txt")
    pprint(sys.path)

default_args = {
    'owner': 'Blabla',
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='concrete_structure_print_sys_path',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    description='Print and save sys.path for Airflow environment',
    tags=['concrete','utility'],
) as dag:

    print_paths = PythonOperator(
        task_id='print_and_save_sys_path',
        python_callable=save_sys_path
    )
