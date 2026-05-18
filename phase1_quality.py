"""
phase1_quality.py — Data quality audit for 2018 NYC Yellow Taxi sample

Description:
    Audits data_sample.parquet for five quality issues: RatecodeID=99 anomalies,
    negative trip durations, negative fares, zero-passenger trips, and future dates.
    Writes findings to quality_results.json.

Input:
    - data_sample.parquet

Output:
    - quality_results.json  : structured quality findings per category

NOTES:
    - Does NOT re-sample from CSV. Loads parquet directly.
    - "future dates" means pickup year != 2018.

Version: 1.0.0
Created: 2026-05-17
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
PARQUET_CACHE = "data_sample.parquet"
QUALITY_RESULTS = "quality_results.json"
LOGS_DIR = Path("logs")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"error_log_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def fmt_pct(n: int, total: int) -> float:
    return round(n / total * 100, 4) if total else 0.0


def compare_means(df: pd.DataFrame, mask: pd.Series, cols: list[str]) -> dict:
    """Return mean values for masked vs unmasked rows."""
    result = {}
    for col in cols:
        if col not in df.columns:
            continue
        anomaly_mean = float(df.loc[mask, col].mean()) if mask.any() else None
        normal_mean = float(df.loc[~mask, col].mean()) if (~mask).any() else None
        result[col] = {
            "anomaly_mean": round(anomaly_mean, 4)
            if anomaly_mean is not None
            else None,
            "normal_mean": round(normal_mean, 4) if normal_mean is not None else None,
        }
    return result


# ---------------------------------------------------------------------------
# AUDIT FUNCTIONS
# ---------------------------------------------------------------------------
def audit_ratecode_99(df: pd.DataFrame, n_total: int) -> dict:
    """1a: Rows where RatecodeID == 99."""
    if "RatecodeID" not in df.columns:
        return {"count": 0, "pct": 0.0, "notes": "Column not found", "comparison": {}}

    mask = df["RatecodeID"] == 99
    count = int(mask.sum())
    pct = fmt_pct(count, n_total)

    comparison = compare_means(
        df, mask, ["fare_amount", "trip_distance", "trip_duration_min", "total_amount"]
    )

    # Check vendor distribution for RatecodeID=99
    vendor_dist = {}
    if "VendorID" in df.columns and count > 0:
        vendor_dist = {
            str(k): int(v) for k, v in df.loc[mask, "VendorID"].value_counts().items()
        }

    logger.info(f"RatecodeID=99: {count:,} rows ({pct}%)")
    return {
        "count": count,
        "pct": pct,
        "notes": (
            "RatecodeID=99 is an undocumented/negotiated rate code. "
            f"{count:,} rows ({pct}%). Higher fares than normal expected."
        ),
        "comparison": comparison,
        "vendor_distribution": vendor_dist,
    }


def audit_negative_duration(df: pd.DataFrame, n_total: int) -> dict:
    """1b: Rows where trip_duration_min < 0 (dropoff before pickup)."""
    if "trip_duration_min" not in df.columns:
        return {"count": 0, "pct": 0.0, "notes": "Column not found", "worst_10": []}

    mask = df["trip_duration_min"] < 0
    count = int(mask.sum())
    pct = fmt_pct(count, n_total)

    worst_10 = []
    if count > 0:
        cols_to_show = [
            c
            for c in [
                "VendorID",
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "trip_duration_min",
                "fare_amount",
                "pickup_hour",
            ]
            if c in df.columns
        ]
        worst = df.loc[mask].nsmallest(10, "trip_duration_min")[cols_to_show]
        worst_10 = json.loads(worst.to_json(orient="records", date_format="iso"))

        # Cluster by vendor
        vendor_dist = {}
        if "VendorID" in df.columns:
            vendor_dist = {
                str(k): int(v)
                for k, v in df.loc[mask, "VendorID"].value_counts().items()
            }

        # Cluster by hour
        hour_dist = {}
        if "pickup_hour" in df.columns:
            hour_dist = {
                str(k): int(v)
                for k, v in df.loc[mask, "pickup_hour"].value_counts().head(5).items()
            }

        # Cluster by pickup_month
        month_dist = {}
        if "pickup_month" in df.columns:
            month_dist = {
                str(k): int(v)
                for k, v in df.loc[mask, "pickup_month"].value_counts().head(5).items()
            }
    else:
        vendor_dist = {}
        hour_dist = {}
        month_dist = {}

    logger.info(f"Negative duration: {count:,} rows ({pct}%)")
    return {
        "count": count,
        "pct": pct,
        "notes": (
            f"{count:,} rows ({pct}%) have trip_duration_min < 0 "
            "(dropoff timestamp precedes pickup — likely data entry errors or DST transitions)."
        ),
        "worst_10": worst_10,
        "vendor_distribution": vendor_dist,
        "top_hours": hour_dist,
        "top_months": month_dist,
    }


def audit_negative_fare(df: pd.DataFrame, n_total: int) -> dict:
    """Count and characterize negative fare_amount rows."""
    if "fare_amount" not in df.columns:
        return {"count": 0, "pct": 0.0, "notes": "Column not found"}

    mask = df["fare_amount"] < 0
    count = int(mask.sum())
    pct = fmt_pct(count, n_total)
    min_fare = float(df.loc[mask, "fare_amount"].min()) if count > 0 else None

    logger.info(f"Negative fare: {count:,} rows ({pct}%)")
    return {
        "count": count,
        "pct": pct,
        "notes": (
            f"{count:,} rows ({pct}%) have fare_amount < 0. "
            f"Most negative value: {min_fare}. Likely reversals or adjustments."
        ),
        "min_fare": min_fare,
    }


def audit_zero_passenger(df: pd.DataFrame, n_total: int) -> dict:
    """Count rows where passenger_count == 0."""
    if "passenger_count" not in df.columns:
        return {"count": 0, "pct": 0.0, "notes": "Column not found"}

    mask = df["passenger_count"] == 0
    count = int(mask.sum())
    pct = fmt_pct(count, n_total)

    logger.info(f"Zero passengers: {count:,} rows ({pct}%)")
    return {
        "count": count,
        "pct": pct,
        "notes": (
            f"{count:,} rows ({pct}%) have passenger_count=0. "
            "May indicate dispatch/test trips or missing data."
        ),
    }


def audit_future_dates(df: pd.DataFrame, n_total: int) -> dict:
    """Count rows with pickup year != 2018."""
    if "tpep_pickup_datetime" not in df.columns:
        return {
            "count": 0,
            "pct": 0.0,
            "notes": "Column not found",
            "year_distribution": {},
        }

    dt_col = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    year_counts = dt_col.dt.year.value_counts()
    non_2018 = year_counts[year_counts.index != 2018]

    count = int(non_2018.sum())
    pct = fmt_pct(count, n_total)
    year_dist = {str(k): int(v) for k, v in year_counts.items()}

    logger.info(f"Future/invalid dates: {count:,} rows ({pct}%)")
    return {
        "count": count,
        "pct": pct,
        "notes": (
            f"{count:,} rows ({pct}%) have pickup year != 2018. "
            f"Year distribution: {year_dist}"
        ),
        "year_distribution": year_dist,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA QUALITY AUDIT")
    logger.info("=" * 60)

    df = pd.read_parquet(PARQUET_CACHE)
    # Ensure datetime columns are parsed
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    n_total = len(df)
    logger.info(f"Loaded {n_total:,} rows × {df.shape[1]} columns")

    quality = {
        "generated_at": datetime.now().isoformat(),
        "sample_size": n_total,
        "ratecode_99": audit_ratecode_99(df, n_total),
        "negative_duration": audit_negative_duration(df, n_total),
        "negative_fare": audit_negative_fare(df, n_total),
        "zero_passenger": audit_zero_passenger(df, n_total),
        "future_dates": audit_future_dates(df, n_total),
    }

    # Summary table to stdout
    logger.info("\n--- QUALITY SUMMARY ---")
    for key in [
        "ratecode_99",
        "negative_duration",
        "negative_fare",
        "zero_passenger",
        "future_dates",
    ]:
        entry = quality[key]
        logger.info(f"  {key:25s}: {entry['count']:>7,} rows  ({entry['pct']:.4f}%)")

    with open(QUALITY_RESULTS, "w") as f:
        json.dump(quality, f, indent=2, default=str)
    logger.info(f"\nResults saved to {QUALITY_RESULTS}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
