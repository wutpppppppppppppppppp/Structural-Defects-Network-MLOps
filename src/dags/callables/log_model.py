import mlflow
import mlflow.pytorch
import logging
import torch

logger = logging.getLogger(__name__)

def log_model(**kwargs):
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='train_model', key='model_path')
    accuracy = ti.xcom_pull(task_ids='train_model', key='accuracy')
    optimizer = ti.xcom_pull(task_ids='train_model', key='optimizer')
    learning_rate = ti.xcom_pull(task_ids='train_model', key='learning_rate')
    epochs = ti.xcom_pull(task_ids='train_model', key='epochs')

    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    logger.info("📦 Logging model to MLflow...")

    with mlflow.start_run(run_name="Concrete_Crack_Classifier"):
        model = torch.load(model_path, map_location='cpu')

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="ConcreteCrackClassifierModel"
        )

        # Log parameters and metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

    logger.info("✅ Model and metadata logged to MLflow.")
