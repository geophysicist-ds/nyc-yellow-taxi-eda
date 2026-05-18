"""
eda_main.py — Main EDA pipeline for 2018 NYC Yellow Taxi Trip Data

Description:
    Samples 500K rows from a 14GB CSV using lambda-based random sampling,
    caches to Parquet for reuse, performs full statistical analysis
    (null handling, outlier detection, correlation, distributions),
    and writes schema JSON + analysis results consumed by eda_viz.py.

Input:
    - 2018_Yellow_Taxi_Trip_Data_20260516.csv  (14.47 GB, ~112M rows)

Output:
    - data_sample.parquet   : 500K-row sample cache
    - data_schema.json      : Column schema, dtypes, null counts, sample values
    - analysis_results.json : Statistical summaries, drop log, outlier info
    - logs/error_log_*.txt  : Error logs

NOTES:
    - Run from project root with: uv run python eda_main.py
    - Sampling uses random.seed(42) for reproducibility.
    - If data_sample.parquet already exists, it is loaded directly (no re-sampling).
    - Columns with >40% nulls are dropped; rows with <40% column nulls are dropped.
    - datetime column year 2018 validation is applied (data quality flag).

Version: 1.0.0
Created: 2026-05-17
"""

import json
import logging
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
CSV_FILE = "2018_Yellow_Taxi_Trip_Data_20260516.csv"
PARQUET_CACHE = "data_sample.parquet"
SCHEMA_FILE = "data_schema.json"
RESULTS_FILE = "analysis_results.json"
PLOTS_DIR = Path("plots")
LOGS_DIR = Path("logs")
SAMPLE_RATIO = 0.005  # ~500K rows from 112M
RANDOM_SEED = 42
NULL_DROP_COL_THRESHOLD = 0.40  # drop column if >40% null
NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Track step-level error counts for the 3-tier retry rule
ERROR_COUNTS: dict[str, int] = {}
SKIPPED_STEPS: list[str] = []


def log_error(step: str, exc: Exception, critical: bool = False) -> bool:
    """Log an error; return True if step should be skipped (>2 repeats or critical)."""
    ERROR_COUNTS[step] = ERROR_COUNTS.get(step, 0) + 1
    msg = f"[{step}] {type(exc).__name__}: {exc}"
    logger.error(msg)
    logger.debug(traceback.format_exc())

    if critical:
        logger.critical(f"CRITICAL ERROR in {step} — stopping execution.")
        SKIPPED_STEPS.append(step)
        return True

    if ERROR_COUNTS[step] > 2:
        logger.warning(f"Step '{step}' failed 3+ times — marking SKIPPED.")
        SKIPPED_STEPS.append(step)
        return True

    return False


# ---------------------------------------------------------------------------
# STEP 1: SAMPLE OR LOAD CACHE
# ---------------------------------------------------------------------------
def load_or_sample() -> pd.DataFrame:
    """Load cached parquet sample or sample from CSV using lambda approach."""
    if Path(PARQUET_CACHE).exists():
        logger.info(f"Cache found: loading {PARQUET_CACHE}")
        df = pd.read_parquet(PARQUET_CACHE)
        logger.info(f"Loaded sample from cache: {df.shape}")
        return df

    if not Path(CSV_FILE).exists():
        raise FileNotFoundError(f"Source CSV not found: {CSV_FILE}")

    logger.info(f"No cache found. Sampling from {CSV_FILE} with ratio={SAMPLE_RATIO}")
    random.seed(RANDOM_SEED)

    # Lambda-based random sampling: keep header row (x==0) + random rows
    df = pd.read_csv(
        CSV_FILE,
        skiprows=lambda x: x != 0 and random.random() > SAMPLE_RATIO,
        low_memory=False,
    )
    logger.info(f"Sampled {len(df):,} rows from CSV.")

    # Save to parquet cache
    df.to_parquet(PARQUET_CACHE, index=False)
    logger.info(f"Sample cached to {PARQUET_CACHE}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: SAVE SCHEMA
# ---------------------------------------------------------------------------
def save_schema(df: pd.DataFrame) -> dict:
    """Cache column schema to JSON after first load."""
    schema = {
        "shape": list(df.shape),
        "columns": {},
    }
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        schema["columns"][col] = {
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2),
            "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
        }
    with open(SCHEMA_FILE, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Schema saved to {SCHEMA_FILE}")
    return schema


# ---------------------------------------------------------------------------
# STEP 3: NULL HANDLING
# ---------------------------------------------------------------------------
def handle_nulls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop columns >40% null, then drop rows with any remaining nulls."""
    drop_log = {
        "dropped_columns": [],
        "dropped_rows": 0,
        "original_shape": list(df.shape),
    }
    n_rows = len(df)

    # Drop columns exceeding threshold
    for col in df.columns:
        null_pct = df[col].isna().sum() / n_rows
        if null_pct > NULL_DROP_COL_THRESHOLD:
            logger.info(f"Dropping column '{col}' — {null_pct:.1%} null")
            drop_log["dropped_columns"].append(
                {"column": col, "null_pct": round(null_pct, 4)}
            )
            df = df.drop(columns=[col])

    # Drop rows with remaining nulls
    before = len(df)
    df = df.dropna()
    dropped_rows = before - len(df)
    drop_log["dropped_rows"] = dropped_rows
    drop_log["final_shape"] = list(df.shape)
    logger.info(
        f"Dropped {len(drop_log['dropped_columns'])} columns, {dropped_rows:,} rows with nulls."
    )
    return df, drop_log


# ---------------------------------------------------------------------------
# STEP 4: TYPE CONVERSION + FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetimes, create duration/hour/day features, binarize store_and_fwd_flag."""
    # Parse datetimes — note: raw data has non-standard format "2084 Nov 04 12:32:24 PM"
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Flag rows with year != 2018 as data quality issues
    if "tpep_pickup_datetime" in df.columns:
        invalid_year = df["tpep_pickup_datetime"].dt.year != 2018
        n_invalid = invalid_year.sum()
        if n_invalid > 0:
            logger.warning(
                f"Found {n_invalid:,} rows with pickup year != 2018 (data quality)."
            )
        df["year_flag"] = (~invalid_year).astype(int)

    # Trip duration in minutes
    if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
        df["trip_duration_min"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

    # Hour of day and day of week from pickup time
    if "tpep_pickup_datetime" in df.columns:
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=Mon
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    # Binarize store_and_fwd_flag: Y→1, N→0
    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (
            df["store_and_fwd_flag"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
        )
        logger.info("Binarized 'store_and_fwd_flag': Y=1, N=0")

    # Numeric coercions for safety
    numeric_cols = [
        "VendorID",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# STEP 5: STATISTICAL ANALYSIS
# ---------------------------------------------------------------------------
def statistical_analysis(df: pd.DataFrame) -> dict:
    """Compute descriptive stats, correlations, and outlier flags."""
    results = {}

    # Numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()

    # Descriptive statistics
    desc = numeric_df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    results["descriptive_stats"] = json.loads(desc.to_json())

    # Correlation matrix
    corr = numeric_df.corr(method="pearson")
    results["correlation_matrix"] = json.loads(corr.to_json())

    # Outlier detection via IQR for key numeric columns
    key_cols = [
        c
        for c in [
            "trip_distance",
            "fare_amount",
            "total_amount",
            "tip_amount",
            "passenger_count",
            "trip_duration_min",
        ]
        if c in numeric_cols
    ]
    outlier_info = {}
    for col in key_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_info[col] = {
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_fence": round(float(lower), 4),
            "upper_fence": round(float(upper), 4),
            "n_outliers": n_outliers,
            "outlier_pct": round(n_outliers / len(df) * 100, 2),
        }
        logger.info(
            f"Outliers in '{col}': {n_outliers:,} ({outlier_info[col]['outlier_pct']}%)"
        )
    results["outlier_analysis"] = outlier_info

    # Value counts for categorical-like columns
    cat_cols = ["VendorID", "RatecodeID", "payment_type", "passenger_count"]
    value_counts = {}
    for col in cat_cols:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).head(20)
            value_counts[col] = {str(k): int(v) for k, v in vc.items()}
    results["value_counts"] = value_counts

    # Temporal distributions
    temporal = {}
    for col in ["pickup_hour", "pickup_dow", "pickup_month"]:
        if col in df.columns:
            vc = df[col].value_counts(sort=False).sort_index()
            temporal[col] = {str(k): int(v) for k, v in vc.items()}
    results["temporal_distributions"] = temporal

    # Data quality: invalid datetime years
    if "year_flag" in df.columns:
        results["data_quality"] = {
            "invalid_year_rows": int((df["year_flag"] == 0).sum()),
            "valid_year_rows": int((df["year_flag"] == 1).sum()),
        }

    logger.info("Statistical analysis complete.")
    return results


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("EDA PIPELINE START")
    logger.info("=" * 60)

    # Step 1: Load or sample
    try:
        df = load_or_sample()
    except FileNotFoundError as e:
        log_error("load_or_sample", e, critical=True)
        sys.exit(1)
    except MemoryError as e:
        log_error("load_or_sample", e, critical=True)
        sys.exit(1)
    except Exception as e:
        log_error("load_or_sample", e, critical=True)
        sys.exit(1)

    # Step 2: Save schema
    try:
        schema = save_schema(df)
        logger.info(f"Schema: {schema['shape'][0]:,} rows × {schema['shape'][1]} cols")
    except Exception as e:
        skip = log_error("save_schema", e)
        if skip:
            schema = {}

    # Step 3: Null handling
    try:
        df, drop_log = handle_nulls(df)
    except Exception as e:
        skip = log_error("handle_nulls", e)
        if skip:
            drop_log = {}

    # Step 4: Feature engineering
    try:
        df = engineer_features(df)
    except Exception as e:
        skip = log_error("engineer_features", e)
        if skip:
            logger.warning("Feature engineering skipped.")

    # Step 5: Statistical analysis
    analysis_results = {}
    try:
        analysis_results = statistical_analysis(df)
    except Exception as e:
        skip = log_error("statistical_analysis", e)
        if skip:
            logger.warning("Statistical analysis skipped.")

    # Attach drop log and skipped steps
    analysis_results["drop_log"] = drop_log
    analysis_results["skipped_steps"] = SKIPPED_STEPS
    analysis_results["sample_shape"] = list(df.shape)
    analysis_results["columns"] = list(df.columns)

    # Save analysis results
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info(f"Analysis results saved to {RESULTS_FILE}")
    except Exception as e:
        log_error("save_results", e)

    # Save cleaned sample back to parquet (with engineered features)
    try:
        df.to_parquet(PARQUET_CACHE, index=False)
        logger.info("Enriched sample saved back to parquet cache.")
    except Exception as e:
        log_error("save_enriched_parquet", e)

    logger.info("=" * 60)
    logger.info("EDA PIPELINE COMPLETE — Run eda_viz.py next.")
    logger.info(f"Final dataframe: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
