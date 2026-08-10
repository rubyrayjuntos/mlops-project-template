# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Compares recent production inference data against the stored training
baseline. Prints MONITORING_STATUS=<NOT_READY|INSUFFICIENT_DATA|HEALTHY|
DRIFT_DETECTED> as the last stdout line - only DRIFT_DETECTED should trigger
a retrain; the other two are informational states that must stay visibly
distinct from "checked and found healthy."

Numeric features: two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp),
gated by both statistical significance (after Benjamini-Hochberg correction
across all tested features) and practical significance (D-statistic).

Categorical features: chi-square goodness-of-fit against the baseline's
frequency distribution (scipy.stats.chisquare), computed over the union of
baseline and production categories with additive smoothing so an unseen
category doesn't produce a zero-expected-frequency cell. Gated by the same
BH-corrected significance plus a Cramer's V practical-effect threshold.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

NUMERIC_COLS = [
    "distance", "dropoff_latitude", "dropoff_longitude", "passengers",
    "pickup_latitude", "pickup_longitude", "pickup_weekday", "pickup_month",
    "pickup_monthday", "pickup_hour", "pickup_minute", "pickup_second",
    "dropoff_weekday", "dropoff_month", "dropoff_monthday",
    "dropoff_hour", "dropoff_minute", "dropoff_second",
]
CAT_NOM_COLS = ["store_forward", "vendor"]
SMOOTHING_EPSILON = 1e-4  # additive smoothing so unseen categories get nonzero expected probability


def parse_args():
    p = argparse.ArgumentParser("check_drift")
    p.add_argument("--storage_account", type=str, required=True)
    p.add_argument("--container", type=str, default="monitoring")
    p.add_argument("--lookback_days", type=int, default=7)
    p.add_argument("--min_rows", type=int, default=30, help="Minimum recent inference rows before running tests at all")
    p.add_argument("--fdr_alpha", type=float, default=0.05, help="Benjamini-Hochberg FDR level")
    p.add_argument("--ks_effect_threshold", type=float, default=0.1,
                    help="Minimum KS D-statistic to count as practically significant (starting heuristic, tune with real data)")
    p.add_argument("--cramers_v_threshold", type=float, default=0.1,
                    help="Minimum Cramer's V to count as practically significant (0.1=small, 0.3=medium per Cohen's convention)")
    p.add_argument("--min_drifted_features", type=int, default=2,
                    help="How many features must clear both gates before DRIFT_DETECTED fires")
    p.add_argument("--report_output", type=str, default=None, help="Optional path to write a JSON report")
    return p.parse_args()


def load_baseline(blob_service, container):
    client = blob_service.get_blob_client(container=container, blob="monitoring/baseline/reference.json")
    if not client.exists():
        return None
    return json.loads(client.download_blob().readall())


def load_recent_inference_data(blob_service, container, lookback_days):
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    prefix = "monitoring/inference-log/"
    container_client = blob_service.get_container_client(container)
    frames = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if blob.last_modified is not None and blob.last_modified < cutoff:
            continue
        data = container_client.download_blob(blob.name).readall()
        frames.append(pd.read_parquet(BytesIO(data)))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def benjamini_hochberg(p_values: dict, alpha: float) -> set:
    """Standard BH-FDR step-up procedure. Returns the set of feature names
    that remain significant after correcting for testing len(p_values) features."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    if m == 0:
        return set()
    largest_k = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            largest_k = i
    return {name for name, _ in items[:largest_k]}


def check_numeric_drift(baseline_numeric: dict, recent: pd.DataFrame) -> dict:
    """Returns {col: {statistic, p_value}} for every numeric col present in both."""
    results = {}
    for col, ref in baseline_numeric.items():
        if col not in recent.columns:
            continue
        sample = recent[col].dropna()
        if len(sample) == 0:
            continue
        stat, p_value = ks_2samp(ref["values_sample"], sample)
        results[col] = {"statistic": float(stat), "p_value": float(p_value)}
    return results


def check_categorical_drift(baseline_categorical: dict, recent: pd.DataFrame) -> dict:
    """Returns {col: {statistic, p_value, cramers_v, new_categories}} for every
    categorical col present in both, using the union of categories with additive
    smoothing so unseen production categories don't break chisquare."""
    results = {}
    for col, ref_dist_raw in baseline_categorical.items():
        if col not in recent.columns:
            continue
        # Normalize types on both sides - JSON round-trips baseline keys to str;
        # a pandas column read from parquet may be int/str/other.
        ref_dist = {str(k): v for k, v in ref_dist_raw.items()}
        observed_counts = recent[col].astype(str).value_counts()

        baseline_categories = set(ref_dist.keys())
        production_categories = set(observed_counts.index)
        all_categories = sorted(baseline_categories | production_categories)
        new_categories = sorted(production_categories - baseline_categories)

        n = len(recent)
        k = len(all_categories)
        # Additive smoothing over the union so a brand-new category gets a small
        # nonzero expected probability instead of an undefined/zero-expected cell.
        smoothed = {c: ref_dist.get(c, 0.0) + SMOOTHING_EPSILON for c in all_categories}
        total = sum(smoothed.values())
        expected = np.array([smoothed[c] / total * n for c in all_categories])
        observed = np.array([observed_counts.get(c, 0) for c in all_categories])

        stat, p_value = chisquare(observed, f_exp=expected)
        # Cramer's V: effect size for chi-square goodness-of-fit against k categories.
        # Clipped to the conventional [0, 1] bound - the goodness-of-fit variant can
        # exceed 1 when a brand-new (baseline-unseen) category captures a nontrivial
        # share of production traffic, which would otherwise break the Cohen's-convention
        # interpretation (0.1=small, 0.3=medium) surfaced in --cramers_v_threshold's help
        # text. Clipping never changes the drift-detection gate: for any threshold <= 1.0,
        # a value that already cleared the threshold still clears it after clipping to 1.0,
        # and a value already < 1.0 is unaffected.
        cramers_v = min(float(np.sqrt(stat / (n * max(k - 1, 1)))), 1.0) if n > 0 else 0.0

        results[col] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "cramers_v": cramers_v,
            "new_categories": new_categories,  # surfaced regardless of test outcome - worth knowing about
        }
    return results


def main(args):
    credential = DefaultAzureCredential(
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    )
    blob_service = BlobServiceClient(
        account_url=f"https://{args.storage_account}.blob.core.windows.net",
        credential=credential,
    )

    baseline = load_baseline(blob_service, args.container)
    if baseline is None:
        print("No baseline found yet - a model has not been promoted through the pipeline since monitoring was added.")
        print("MONITORING_STATUS=NOT_READY")
        return

    recent = load_recent_inference_data(blob_service, args.container, args.lookback_days)
    row_count = 0 if recent is None else len(recent)
    if row_count < args.min_rows:
        print(f"Only {row_count} inference rows in the last {args.lookback_days} days "
              f"(minimum {args.min_rows}) - too few for a statistically meaningful comparison.")
        print("MONITORING_STATUS=INSUFFICIENT_DATA")
        return

    numeric_results = check_numeric_drift(baseline["stats"]["numeric"], recent)
    categorical_results = check_categorical_drift(baseline["stats"]["categorical"], recent)

    all_p_values = {f"numeric:{k}": v["p_value"] for k, v in numeric_results.items()}
    all_p_values.update({f"categorical:{k}": v["p_value"] for k, v in categorical_results.items()})
    significant_after_correction = benjamini_hochberg(all_p_values, args.fdr_alpha)

    drifted_features = []
    for col, r in numeric_results.items():
        if f"numeric:{col}" in significant_after_correction and r["statistic"] >= args.ks_effect_threshold:
            drifted_features.append(col)
    for col, r in categorical_results.items():
        if f"categorical:{col}" in significant_after_correction and r["cramers_v"] >= args.cramers_v_threshold:
            drifted_features.append(col)

    report = {
        "baseline_model_version": baseline.get("model_version"),
        "baseline_captured_at": baseline["captured_at"],
        "inference_rows_compared": row_count,
        "numeric": numeric_results,
        "categorical": categorical_results,
        "significant_after_fdr_correction": sorted(significant_after_correction),
        "drifted_features": drifted_features,
    }

    if args.report_output:
        with open(args.report_output, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    status = "DRIFT_DETECTED" if len(drifted_features) >= args.min_drifted_features else "HEALTHY"
    print(f"MONITORING_STATUS={status}")


if __name__ == "__main__":
    sys.exit(main(parse_args()) or 0)
