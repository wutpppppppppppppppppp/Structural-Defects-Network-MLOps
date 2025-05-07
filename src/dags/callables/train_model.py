def train_model(**kwargs):
    model_path = "/tmp/latest_model.pth"
    accuracy = 0.92  # Replace with real evaluation
    optimizer = "Adam"
    learning_rate = 0.001
    epochs = 20

    # Simulate saving model
    torch.save(model, model_path)

    # Push multiple XCom values
    ti = kwargs['ti']
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='accuracy', value=accuracy)
    ti.xcom_push(key='optimizer', value=optimizer)
    ti.xcom_push(key='learning_rate', value=learning_rate)
    ti.xcom_push(key='epochs', value=epochs)

    Variable.set("last_retrain_time", datetime.utcnow().isoformat())
    logger.info("✅ Model trained and metadata pushed to XCom.")
