# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Snapshots per-feature reference statistics from the test set to blob
storage, for later drift comparison against production inference data.

Takes register_model's model_info_output_path as an input specifically to
create an AML pipeline data dependency on register_model succeeding - AML
schedules jobs by data dependency, not YAML file order, so without this the
snapshot could run (or succeed after a register_model failure) for a model
that was never actually promoted. If model_info.json doesn't exist at that
path, register.py's own deploy_flag gate declined to register - skip.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]
CAT_NOM_COLS = ["store_forward", "vendor"]


def parse_args():
    parser = argparse.ArgumentParser("snapshot_baseline")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset (parquet dir)")
    parser.add_argument("--model_info_path", type=str, required=True,
                         help="register_model's model_info_output_path - existence of model_info.json here IS the promotion signal")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--storage_account", type=str, required=True, help="Storage account name, e.g. stazmlops0001dev")
    parser.add_argument("--container", type=str, default="monitoring", help="Blob container to write to")
    return parser.parse_args()


def compute_reference_stats(df: pd.DataFrame) -> dict:
    stats = {"numeric": {}, "categorical": {}}
    for col in NUMERIC_COLS:
        if col in df.columns:
            stats["numeric"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "values_sample": df[col].sample(min(500, len(df)), random_state=42).tolist(),
            }
    for col in CAT_NOM_COLS:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True)
            stats["categorical"][col] = counts.to_dict()
    return stats


def main(args):
    model_info_file = Path(args.model_info_path) / "model_info.json"
    if not model_info_file.exists():
        print("model_info.json not found - register_model declined to promote this run. Skipping baseline snapshot.")
        return

    with open(model_info_file) as f:
        model_info = json.load(f)  # {"id": "taxi-model:<version>"}
    model_id = model_info["id"]
    model_version = model_id.split(":")[-1]

    test_data = pd.read_parquet(Path(args.test_data))
    stats = compute_reference_stats(test_data)
    payload = {
        "model_name": args.model_name,
        "model_version": model_version,
        "training_run_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
        # Dataset version/hash lineage is a documented future addition - would need the resolved
        # input dataset URI threaded through as an extra pipeline input, not done in this pass.
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(test_data),
        "stats": stats,
    }

    credential = DefaultAzureCredential(
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    )
    account_url = f"https://{args.storage_account}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=args.container, blob="monitoring/baseline/reference.json")
    blob_client.upload_blob(json.dumps(payload, indent=2), overwrite=True)
    print(f"Baseline snapshot written to {args.container}/monitoring/baseline/reference.json "
          f"({payload['row_count']} rows, captured {payload['captured_at']}).")


if __name__ == "__main__":
    main(parse_args())
