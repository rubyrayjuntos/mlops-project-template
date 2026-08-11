import os
import json
import uuid
import logging
from datetime import datetime, timezone

import mlflow
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_model_path(model_dir):
    if os.path.isfile(os.path.join(model_dir, "MLmodel")):
        return model_dir

    candidates = [
        os.path.join(model_dir, entry)
        for entry in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, entry, "MLmodel"))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one MLflow model under {model_dir}, found {len(candidates)}"
        )
    return candidates[0]


def init():
    global model, blob_service, storage_account, container

    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"AZUREML_MODEL_DIR: {model_dir}")
    model_path = _resolve_model_path(model_dir if model_dir else "./model")

    try:
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

    storage_account = os.environ.get("MONITORING_STORAGE_ACCOUNT")
    container = os.environ.get("MONITORING_CONTAINER", "monitoring")
    if storage_account:
        credential = DefaultAzureCredential(
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        )
        blob_service = BlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=credential,
        )
    else:
        blob_service = None
        logger.warning("MONITORING_STORAGE_ACCOUNT not set - inference logging disabled.")


def run(raw_data):
    logger.info("Received scoring request")
    body = json.loads(raw_data)
    input_data = body["input_data"]
    df = pd.DataFrame(data=input_data["data"], columns=input_data["columns"])

    predictions = model.predict(df)

    if blob_service is not None:
        log_df = df.copy()
        log_df["prediction"] = predictions
        log_df["logged_at"] = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc)
        blob_path = f"monitoring/inference-log/online/{now:%Y}/{now:%m}/{now:%d}/{uuid.uuid4()}.parquet"
        try:
            blob_client = blob_service.get_blob_client(container=container, blob=blob_path)
            blob_client.upload_blob(log_df.to_parquet(index=False), overwrite=True)
        except Exception as e:
            # Logging failure must never break a live scoring request.
            logger.error(f"Inference logging failed (non-fatal): {str(e)}")

    return json.dumps(predictions.tolist())
