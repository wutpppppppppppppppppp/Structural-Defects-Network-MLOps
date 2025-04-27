FROM apache/airflow:slim-2.10.5-python3.11

ADD requirements.txt .

COPY --chown=airflow:root dags/test_dag.py /opt/airflow/dags

RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt