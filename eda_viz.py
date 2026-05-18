"""
eda_viz.py — Visualization subagent for 2018 NYC Yellow Taxi Trip EDA

Description:
    Loads the enriched data_sample.parquet and analysis_results.json produced
    by eda_main.py, then generates all EDA plots and the final eda_report.html.
    Called after eda_main.py completes.

Input:
    - data_sample.parquet      : Enriched 500K-row sample
    - analysis_results.json    : Statistical summaries from eda_main.py
    - data_schema.json         : Column schema

Output:
    - plots/*.png              : All static plot images
    - plots/*.html             : Interactive Plotly HTML plots
    - eda_report.html          : Final assembled EDA report

NOTES:
    - Run from project root with: uv run python eda_viz.py
    - All Plotly plots use width=1200, height=600, template="plotly_white"
    - Matplotlib fallbacks use figsize=(12,6), dpi=300, colormap="viridis"
    - Geographic lat/lon columns are not present in this dataset (only LocationIDs).
      Folium heatmaps are therefore skipped with a note in the report.
    - This script is designed to be spawned as a subagent by eda_main.py
      but can also be run independently.

Version: 1.0.0
Created: 2026-05-17
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
PARQUET_CACHE = "data_sample.parquet"
RESULTS_FILE = "analysis_results.json"
SCHEMA_FILE = "data_schema.json"
REPORT_FILE = "eda_report.html"
PLOTS_DIR = Path("plots")
LOGS_DIR = Path("logs")

PLOTLY_WIDTH = 1200
PLOTLY_HEIGHT = 600
PLOTLY_TEMPLATE = "plotly_white"

MPL_FIGSIZE = (12, 6)
MPL_DPI = 300
MPL_CMAP = "viridis"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / f"error_log_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

ERROR_COUNTS: dict[str, int] = {}
SKIPPED_PLOTS: list[str] = []
GENERATED_PLOTS: list[dict] = []  # [{name, png, html, description}]


def log_error(step: str, exc: Exception) -> bool:
    """Log plot error; skip step if >2 repeats."""
    ERROR_COUNTS[step] = ERROR_COUNTS.get(step, 0) + 1
    logger.error(f"[{step}] {type(exc).__name__}: {exc}")
    logger.debug(traceback.format_exc())
    if ERROR_COUNTS[step] > 2:
        logger.warning(f"Plot '{step}' failed 3+ times — SKIPPED.")
        SKIPPED_PLOTS.append(step)
        return True
    return False


def save_plotly(fig, name: str, description: str = ""):
    """Save a plotly figure as both PNG and interactive HTML."""
    png_path = PLOTS_DIR / f"{name}.png"
    html_path = PLOTS_DIR / f"{name}.html"
    try:
        fig.write_image(str(png_path))
        fig.write_html(str(html_path))
        GENERATED_PLOTS.append(
            {
                "name": name,
                "png": str(png_path),
                "html": str(html_path),
                "description": description,
            }
        )
        logger.info(f"Saved plot: {name}")
    except Exception as e:
        logger.error(f"Failed to save plot {name}: {e}")


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_parquet(PARQUET_CACHE)
    logger.info(f"Loaded sample: {df.shape}")

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    schema = {}
    if Path(SCHEMA_FILE).exists():
        with open(SCHEMA_FILE) as f:
            schema = json.load(f)

    return df, results, schema


# ---------------------------------------------------------------------------
# PLOT 1: Numeric distributions — histograms
# ---------------------------------------------------------------------------
def plot_distributions(df: pd.DataFrame):
    """Histograms for key numeric columns."""
    step = "distributions"
    try:
        cols = [
            c
            for c in [
                "trip_distance",
                "fare_amount",
                "total_amount",
                "tip_amount",
                "passenger_count",
                "trip_duration_min",
            ]
            if c in df.columns
        ]
        n = len(cols)
        rows = (n + 1) // 2
        fig = make_subplots(rows=rows, cols=2, subplot_titles=cols)

        for i, col in enumerate(cols):
            row = i // 2 + 1
            col_idx = i % 2 + 1
            # Clip extreme outliers for display clarity
            data = df[col].dropna()
            p01, p99 = data.quantile(0.01), data.quantile(0.99)
            data_clipped = data.clip(p01, p99)
            fig.add_trace(
                go.Histogram(
                    x=data_clipped,
                    name=col,
                    nbinsx=60,
                    marker_color="#2196F3",
                    showlegend=False,
                ),
                row=row,
                col=col_idx,
            )

        fig.update_layout(
            title_text="Distribution of Key Numeric Variables (clipped at 1st–99th percentile)",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT * rows,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "01_distributions", "Histograms of key numeric features")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 2: Box plots — outlier visualization
# ---------------------------------------------------------------------------
def plot_boxplots(df: pd.DataFrame):
    """Box plots for numeric columns to show outliers."""
    step = "boxplots"
    try:
        cols = [
            c
            for c in [
                "trip_distance",
                "fare_amount",
                "total_amount",
                "tip_amount",
                "tolls_amount",
            ]
            if c in df.columns
        ]
        fig = go.Figure()
        for col in cols:
            data = df[col].dropna()
            fig.add_trace(go.Box(y=data, name=col, boxpoints="outliers", marker_size=2))
        fig.update_layout(
            title="Box Plots — Outlier Detection for Fare/Distance/Amount Columns",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
            yaxis_title="Value",
        )
        save_plotly(fig, "02_boxplots", "Box plots showing outlier distribution")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 3: Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation(df: pd.DataFrame):
    """Pearson correlation heatmap for numeric columns."""
    step = "correlation"
    try:
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=["year_flag"], errors="ignore"
        )
        # Drop columns with zero variance
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        corr = numeric_df.corr(method="pearson")

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 9},
            )
        )
        fig.update_layout(
            title="Pearson Correlation Heatmap",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT + 200,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(
            fig,
            "03_correlation_heatmap",
            "Pearson correlation matrix of numeric features",
        )
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 4: Trip count by hour of day
# ---------------------------------------------------------------------------
def plot_hourly_trips(df: pd.DataFrame):
    """Bar chart of trip counts by hour of pickup."""
    step = "hourly_trips"
    try:
        if "pickup_hour" not in df.columns:
            return
        hourly = df["pickup_hour"].value_counts(sort=False).sort_index().reset_index()
        hourly.columns = ["hour", "trip_count"]
        fig = px.bar(
            hourly,
            x="hour",
            y="trip_count",
            title="Trip Count by Hour of Day (Pickup)",
            labels={"hour": "Hour of Day", "trip_count": "Number of Trips"},
            color="trip_count",
            color_continuous_scale="Blues",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "04_hourly_trips", "Trip volume per hour of day")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 5: Trip count by day of week
# ---------------------------------------------------------------------------
def plot_dow_trips(df: pd.DataFrame):
    """Bar chart of trip counts by day of week."""
    step = "dow_trips"
    try:
        if "pickup_dow" not in df.columns:
            return
        dow_labels = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
        dow = df["pickup_dow"].value_counts(sort=False).sort_index().reset_index()
        dow.columns = ["dow", "trip_count"]
        dow["day"] = dow["dow"].map(dow_labels)
        fig = px.bar(
            dow,
            x="day",
            y="trip_count",
            title="Trip Count by Day of Week",
            labels={"day": "Day of Week", "trip_count": "Number of Trips"},
            color="trip_count",
            color_continuous_scale="Greens",
            category_orders={"day": list(dow_labels.values())},
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "05_dow_trips", "Trip volume by day of week")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 6: Fare amount vs. trip distance scatter
# ---------------------------------------------------------------------------
def plot_fare_vs_distance(df: pd.DataFrame):
    """Scatter plot of fare amount vs trip distance."""
    step = "fare_vs_distance"
    try:
        if "fare_amount" not in df.columns or "trip_distance" not in df.columns:
            return
        # Sample for scatter performance (max 20K points)
        sample = df[["trip_distance", "fare_amount", "payment_type"]].dropna()
        sample = sample[
            (sample["trip_distance"] > 0)
            & (sample["trip_distance"] < 50)
            & (sample["fare_amount"] > 0)
            & (sample["fare_amount"] < 200)
        ]
        if len(sample) > 20000:
            sample = sample.sample(20000, random_state=42)

        fig = px.scatter(
            sample,
            x="trip_distance",
            y="fare_amount",
            color=sample["payment_type"].astype(str),
            title="Fare Amount vs. Trip Distance",
            labels={
                "trip_distance": "Trip Distance (miles)",
                "fare_amount": "Fare ($)",
            },
            opacity=0.4,
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(
            fig,
            "06_fare_vs_distance",
            "Scatter: fare amount vs trip distance by payment type",
        )
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 7: Payment type distribution
# ---------------------------------------------------------------------------
def plot_payment_type(df: pd.DataFrame):
    """Pie chart of payment type distribution."""
    step = "payment_type"
    try:
        if "payment_type" not in df.columns:
            return
        payment_labels = {
            1: "Credit Card",
            2: "Cash",
            3: "No Charge",
            4: "Dispute",
            5: "Unknown",
            6: "Voided Trip",
        }
        vc = df["payment_type"].value_counts().reset_index()
        vc.columns = ["type_code", "count"]
        vc["label"] = vc["type_code"].map(payment_labels).fillna("Other")
        fig = px.pie(
            vc,
            names="label",
            values="count",
            title="Payment Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "07_payment_type", "Payment type breakdown")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 8: Tip amount distribution by payment type
# ---------------------------------------------------------------------------
def plot_tip_by_payment(df: pd.DataFrame):
    """Box plot of tip amounts by payment type."""
    step = "tip_by_payment"
    try:
        if "tip_amount" not in df.columns or "payment_type" not in df.columns:
            return
        payment_labels = {
            1: "Credit Card",
            2: "Cash",
            3: "No Charge",
            4: "Dispute",
            5: "Unknown",
            6: "Voided",
        }
        plot_df = df[["tip_amount", "payment_type"]].dropna()
        plot_df = plot_df[plot_df["tip_amount"] >= 0]
        plot_df["payment_label"] = (
            plot_df["payment_type"].map(payment_labels).fillna("Other")
        )
        fig = px.box(
            plot_df,
            x="payment_label",
            y="tip_amount",
            title="Tip Amount by Payment Type",
            labels={"payment_label": "Payment Type", "tip_amount": "Tip ($)"},
            color="payment_label",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_yaxes(range=[0, plot_df["tip_amount"].quantile(0.99)])
        save_plotly(fig, "08_tip_by_payment", "Tip amount distribution by payment type")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 9: Trip duration distribution
# ---------------------------------------------------------------------------
def plot_trip_duration(df: pd.DataFrame):
    """Histogram of trip duration in minutes."""
    step = "trip_duration"
    try:
        if "trip_duration_min" not in df.columns:
            return
        data = df["trip_duration_min"].dropna()
        # Filter reasonable durations: 1–120 min
        data = data[(data > 0) & (data <= 120)]
        fig = px.histogram(
            x=data,
            nbins=80,
            title="Trip Duration Distribution (minutes)",
            labels={"x": "Duration (minutes)", "y": "Count"},
            color_discrete_sequence=["#FF5722"],
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(
            fig, "09_trip_duration", "Distribution of trip durations in minutes"
        )
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 10: Vendor comparison
# ---------------------------------------------------------------------------
def plot_vendor_comparison(df: pd.DataFrame):
    """Compare key metrics by VendorID."""
    step = "vendor_comparison"
    try:
        if "VendorID" not in df.columns:
            return
        vendor_labels = {1: "Creative Mobile (1)", 2: "VeriFone (2)"}
        metrics = ["fare_amount", "trip_distance", "tip_amount", "total_amount"]
        metrics = [m for m in metrics if m in df.columns]
        plot_df = df[["VendorID"] + metrics].dropna()
        plot_df["vendor"] = plot_df["VendorID"].map(vendor_labels).fillna("Other")

        fig = make_subplots(
            rows=1,
            cols=len(metrics),
            subplot_titles=metrics,
        )
        colors = {
            "Creative Mobile (1)": "#2196F3",
            "VeriFone (2)": "#FF9800",
            "Other": "#9E9E9E",
        }
        for i, metric in enumerate(metrics, 1):
            for vendor in plot_df["vendor"].unique():
                subset = plot_df[plot_df["vendor"] == vendor][metric]
                p99 = subset.quantile(0.99)
                fig.add_trace(
                    go.Box(
                        y=subset.clip(upper=p99),
                        name=vendor,
                        marker_color=colors.get(vendor, "#607D8B"),
                        showlegend=(i == 1),
                        boxpoints=False,
                    ),
                    row=1,
                    col=i,
                )
        fig.update_layout(
            title="Key Metrics by Vendor",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(
            fig, "10_vendor_comparison", "Vendor-level comparison of key metrics"
        )
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 11: Monthly trip volume (data quality note)
# ---------------------------------------------------------------------------
def plot_monthly_trips(df: pd.DataFrame):
    """Bar chart of trips per month — highlights data quality (non-2018 dates)."""
    step = "monthly_trips"
    try:
        if "pickup_month" not in df.columns:
            return
        monthly = df["pickup_month"].value_counts(sort=False).sort_index().reset_index()
        monthly.columns = ["month", "count"]
        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        monthly["month_name"] = monthly["month"].map(month_names)
        fig = px.bar(
            monthly,
            x="month_name",
            y="count",
            title="Trip Count by Month (2018 Pickup Dates)",
            labels={"month_name": "Month", "count": "Trip Count"},
            color="count",
            color_continuous_scale="Teal",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "11_monthly_trips", "Monthly trip volume")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 12: Top pickup/dropoff location IDs
# ---------------------------------------------------------------------------
def plot_top_locations(df: pd.DataFrame):
    """Bar charts of top 20 pickup and dropoff location IDs."""
    step = "top_locations"
    try:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Top 20 Pickup Locations", "Top 20 Dropoff Locations"],
        )

        for i, col in enumerate(["PULocationID", "DOLocationID"], 1):
            if col not in df.columns:
                continue
            top = df[col].value_counts().head(20).reset_index()
            top.columns = ["location_id", "count"]
            fig.add_trace(
                go.Bar(
                    x=top["location_id"].astype(str),
                    y=top["count"],
                    name=col,
                    marker_color="#7986CB" if i == 1 else "#EF9A9A",
                    showlegend=False,
                ),
                row=1,
                col=i,
            )
        fig.update_layout(
            title="Top 20 Pickup and Dropoff Location IDs",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(
            fig, "12_top_locations", "Most frequent pickup and dropoff location IDs"
        )
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 13: Fare components stacked bar
# ---------------------------------------------------------------------------
def plot_fare_components(df: pd.DataFrame):
    """Average fare component breakdown."""
    step = "fare_components"
    try:
        components = [
            c
            for c in [
                "fare_amount",
                "extra",
                "mta_tax",
                "tip_amount",
                "tolls_amount",
                "improvement_surcharge",
            ]
            if c in df.columns
        ]
        means = df[components].mean().reset_index()
        means.columns = ["component", "avg_amount"]
        fig = px.bar(
            means,
            x="component",
            y="avg_amount",
            title="Average Fare Component Breakdown",
            labels={"component": "Fare Component", "avg_amount": "Average ($)"},
            color="component",
            text_auto=".2f",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "13_fare_components", "Average breakdown of fare components")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 14: Passenger count distribution
# ---------------------------------------------------------------------------
def plot_passenger_count(df: pd.DataFrame):
    """Bar chart of passenger count distribution."""
    step = "passenger_count"
    try:
        if "passenger_count" not in df.columns:
            return
        vc = df["passenger_count"].value_counts().sort_index().reset_index()
        vc.columns = ["passengers", "count"]
        vc = vc[vc["passengers"].between(0, 9)]
        fig = px.bar(
            vc,
            x="passengers",
            y="count",
            title="Passenger Count Distribution",
            labels={"passengers": "Number of Passengers", "count": "Trip Count"},
            color="count",
            color_continuous_scale="Oranges",
            text_auto=True,
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "14_passenger_count", "Distribution of passenger counts")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# PLOT 15: Heatmap — pickup hour vs day of week
# ---------------------------------------------------------------------------
def plot_hour_dow_heatmap(df: pd.DataFrame):
    """Heatmap of trip counts by hour × day of week."""
    step = "hour_dow_heatmap"
    try:
        if "pickup_hour" not in df.columns or "pickup_dow" not in df.columns:
            return
        pivot = (
            df.groupby(["pickup_dow", "pickup_hour"]).size().reset_index(name="count")
        )
        pivot_table = pivot.pivot(
            index="pickup_dow", columns="pickup_hour", values="count"
        ).fillna(0)
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_table.values,
                x=[str(h) for h in pivot_table.columns],
                y=dow_labels[: len(pivot_table)],
                colorscale="YlOrRd",
                colorbar_title="Trips",
            )
        )
        fig.update_layout(
            title="Trip Volume Heatmap: Hour of Day × Day of Week",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,
            template=PLOTLY_TEMPLATE,
        )
        save_plotly(fig, "15_hour_dow_heatmap", "Trip density by hour and day of week")
    except Exception as e:
        log_error(step, e)


# ---------------------------------------------------------------------------
# ASSEMBLE HTML REPORT
# ---------------------------------------------------------------------------
def build_report(results: dict, schema: dict):
    """Assemble final eda_report.html with all plots embedded."""
    drop_log = results.get("drop_log", {})
    sample_shape = results.get("sample_shape", ["N/A", "N/A"])
    skipped = results.get("skipped_steps", []) + SKIPPED_PLOTS
    desc_stats = results.get("descriptive_stats", {})
    outlier_info = results.get("outlier_analysis", {})
    data_quality = results.get("data_quality", {})

    # Read each plot HTML for embedding
    plot_sections = ""
    for plot in GENERATED_PLOTS:
        html_path = Path(plot["html"])
        if html_path.exists():
            # Read iframe-style embed — use object tag with relative path
            rel_path = html_path.name
            plot_sections += f"""
<div class="plot-card">
  <h3>{plot["name"].replace("_", " ").title()}</h3>
  <p class="plot-desc">{plot["description"]}</p>
  <iframe src="plots/{rel_path}" width="100%" height="650px" frameborder="0"
          loading="lazy" style="border-radius:8px;"></iframe>
</div>
"""

    # Build descriptive stats table
    stats_rows = ""
    if desc_stats and "mean" in desc_stats:
        cols = list(desc_stats["mean"].keys())
        for col in cols:
            mean_val = desc_stats.get("mean", {}).get(col, "—")
            med_val = desc_stats.get("50%", {}).get(col, "—")
            min_val = desc_stats.get("min", {}).get(col, "—")
            max_val = desc_stats.get("max", {}).get(col, "—")
            std_val = desc_stats.get("std", {}).get(col, "—")

            def fmt(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return str(v)

            stats_rows += f"""<tr>
              <td>{col}</td><td>{fmt(mean_val)}</td><td>{fmt(med_val)}</td>
              <td>{fmt(min_val)}</td><td>{fmt(max_val)}</td><td>{fmt(std_val)}</td>
            </tr>"""

    # Outlier table
    outlier_rows = ""
    for col, info in outlier_info.items():
        outlier_rows += f"""<tr>
          <td>{col}</td><td>{info["q1"]}</td><td>{info["q3"]}</td>
          <td>{info["lower_fence"]}</td><td>{info["upper_fence"]}</td>
          <td>{info["n_outliers"]:,}</td><td>{info["outlier_pct"]}%</td>
        </tr>"""

    # Dropped columns table
    dropped_cols_rows = ""
    for item in drop_log.get("dropped_columns", []):
        dropped_cols_rows += (
            f"<tr><td>{item['column']}</td><td>{item['null_pct'] * 100:.1f}%</td></tr>"
        )
    if not dropped_cols_rows:
        dropped_cols_rows = "<tr><td colspan='2'>None — all columns retained</td></tr>"

    # Data quality note
    dq_note = ""
    if data_quality:
        inv = data_quality.get("invalid_year_rows", 0)
        val = data_quality.get("valid_year_rows", 0)
        dq_note = f"""
        <div class="alert">
          <strong>Data Quality Warning:</strong>
          {inv:,} rows have pickup year ≠ 2018 (e.g. year 2084 observed in raw data).
          {val:,} rows have valid 2018 dates.
        </div>"""

    skipped_note = ""
    if skipped:
        skipped_note = f"""
        <div class="alert warn">
          <strong>Skipped Steps:</strong> {", ".join(skipped)}
        </div>"""

    geo_note = """
    <div class="alert info">
      <strong>Geographic Visualizations:</strong>
      This dataset contains <code>PULocationID</code> / <code>DOLocationID</code> (integer zone IDs),
      not raw lat/lon coordinates. Folium heatmaps require coordinate columns, so geographic
      map visualizations are not generated for this dataset.
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2018 Yellow Taxi EDA Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f6fa; color: #333; }}
    header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              color: white; padding: 40px 60px; }}
    header h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
    header p {{ opacity: 0.8; font-size: 1rem; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 40px; }}
    .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.07); }}
    .section h2 {{ font-size: 1.4rem; color: #0f3460; border-bottom: 2px solid #e8eaf6;
                   padding-bottom: 10px; margin-bottom: 20px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 16px; margin-bottom: 20px; }}
    .kpi-card {{ background: #f0f4ff; border-radius: 10px; padding: 20px; text-align: center;
                 border-left: 4px solid #3f51b5; }}
    .kpi-card .value {{ font-size: 1.8rem; font-weight: 700; color: #3f51b5; }}
    .kpi-card .label {{ font-size: 0.85rem; color: #555; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th {{ background: #e8eaf6; color: #3f51b5; padding: 10px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
    tr:hover td {{ background: #fafafa; }}
    .plot-card {{ margin-bottom: 30px; }}
    .plot-card h3 {{ font-size: 1.1rem; color: #333; margin-bottom: 6px; }}
    .plot-desc {{ font-size: 0.85rem; color: #777; margin-bottom: 12px; }}
    .alert {{ padding: 14px 18px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem; }}
    .alert {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
    .alert.warn {{ background: #fce4ec; border-left-color: #e91e63; }}
    .alert.info {{ background: #e3f2fd; border-left-color: #2196f3; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 20px;
              font-size: 0.75rem; font-weight: 600; }}
    .badge-blue {{ background: #e3f2fd; color: #1565c0; }}
    footer {{ text-align: center; padding: 30px; color: #aaa; font-size: 0.85rem; }}
  </style>
</head>
<body>
<header>
  <h1>2018 NYC Yellow Taxi — Exploratory Data Analysis</h1>
  <p>Automated EDA Report &nbsp;|&nbsp; Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
     &nbsp;|&nbsp; Sample: {sample_shape[0]:,} rows</p>
</header>

<div class="container">

  <!-- KPI Summary -->
  <div class="section">
    <h2>Dataset Summary</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="value">{sample_shape[0]:,}</div>
        <div class="label">Sampled Rows</div>
      </div>
      <div class="kpi-card">
        <div class="value">{sample_shape[1] if len(sample_shape) > 1 else "N/A"}</div>
        <div class="label">Columns (after cleaning)</div>
      </div>
      <div class="kpi-card">
        <div class="value">112M</div>
        <div class="label">Total Source Rows</div>
      </div>
      <div class="kpi-card">
        <div class="value">{drop_log.get("dropped_rows", 0):,}</div>
        <div class="label">Rows Dropped (nulls)</div>
      </div>
      <div class="kpi-card">
        <div class="value">{len(drop_log.get("dropped_columns", []))}</div>
        <div class="label">Columns Dropped (&gt;40% null)</div>
      </div>
    </div>
    {dq_note}
    {geo_note}
    {skipped_note}
  </div>

  <!-- Data Quality -->
  <div class="section">
    <h2>Null Handling & Data Quality</h2>
    <h3 style="font-size:1rem;margin-bottom:10px;">Dropped Columns (&gt;40% null)</h3>
    <table>
      <tr><th>Column</th><th>Null %</th></tr>
      {dropped_cols_rows}
    </table>
    <p style="margin-top:12px;font-size:0.9rem;color:#555;">
      Rows dropped (remaining nulls): <strong>{drop_log.get("dropped_rows", 0):,}</strong>
    </p>
  </div>

  <!-- Descriptive Statistics -->
  <div class="section">
    <h2>Descriptive Statistics</h2>
    <table>
      <tr><th>Column</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Std Dev</th></tr>
      {stats_rows}
    </table>
  </div>

  <!-- Outlier Analysis -->
  <div class="section">
    <h2>Outlier Analysis (IQR Method)</h2>
    <table>
      <tr><th>Column</th><th>Q1</th><th>Q3</th>
          <th>Lower Fence</th><th>Upper Fence</th><th>Outliers</th><th>Outlier %</th></tr>
      {outlier_rows}
    </table>
  </div>

  <!-- Visualizations -->
  <div class="section">
    <h2>Visualizations</h2>
    {plot_sections}
  </div>

</div>

<footer>
  2018 NYC Yellow Taxi EDA &mdash; Auto-generated by eda_viz.py
</footer>
</body>
</html>"""

    with open(REPORT_FILE, "w") as f:
        f.write(html)
    logger.info(f"Report saved to {REPORT_FILE}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("VIZ PIPELINE START")
    logger.info("=" * 60)

    try:
        df, results, schema = load_data()
    except FileNotFoundError as e:
        logger.critical(f"Required input not found: {e}. Run eda_main.py first.")
        sys.exit(1)

    # Ensure datetime columns are parsed
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Run all plots
    plot_distributions(df)
    plot_boxplots(df)
    plot_correlation(df)
    plot_hourly_trips(df)
    plot_dow_trips(df)
    plot_fare_vs_distance(df)
    plot_payment_type(df)
    plot_tip_by_payment(df)
    plot_trip_duration(df)
    plot_vendor_comparison(df)
    plot_monthly_trips(df)
    plot_top_locations(df)
    plot_fare_components(df)
    plot_passenger_count(df)
    plot_hour_dow_heatmap(df)

    # Assemble report
    build_report(results, schema)

    logger.info("=" * 60)
    logger.info(f"VIZ COMPLETE — {len(GENERATED_PLOTS)} plots generated.")
    logger.info(f"Skipped plots: {SKIPPED_PLOTS if SKIPPED_PLOTS else 'None'}")
    logger.info(f"Report: {REPORT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
