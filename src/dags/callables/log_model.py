import mlflow
import mlflow.pytorch  # or mlflow.tensorflow / mlflow.sklearn depending on your model
import logging
logger = logging.getLogger(__name__)

def log_model():
    # mlflow.set_tracking_uri("http://<your-mlflow-server>:5000") for remote server
    # mlflow.set_tracking_uri("file:/path/to/mlruns") for local run
    logger.info("Logging model to MLflow...")
    with mlflow.start_run(run_name="Concrete_Crack_Classifier"):
        model = torch.load("/tmp/latest_model.pth", map_location='cpu')

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="ConcreteCrackClassifierModel"
        )

        mlflow.log_metric("accuracy", 0.92)  # Replace with real value
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 20)

    logger.info("Model logged and registered to MLflow.")
