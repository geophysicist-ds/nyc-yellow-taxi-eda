"""
phase2_timeseries.py — Time series visualizations for 2018 NYC Yellow Taxi EDA

Description:
    Produces four time-series/temporal plots from data_sample.parquet:
      17_trips_by_hour     : trip volume + avg fare by hour, colored by time period
      18_trips_by_dow      : trip volume + avg distance by day of week
      19_trips_by_month    : monthly trip volume + avg total fare
      20_fare_by_hour_heatmap : avg fare heatmap by hour × day of week

Input:
    - data_sample.parquet

Output:
    - plots/17_trips_by_hour.{png,html}
    - plots/18_trips_by_dow.{png,html}
    - plots/19_trips_by_month.{png,html}
    - plots/20_fare_by_hour_heatmap.{png,html}

NOTES:
    - Does NOT re-sample from CSV. Loads parquet directly.
    - Hour periods: Night 0-5, Morning Rush 6-9, Midday 10-15,
                    Evening Rush 16-19, Evening 20-23.

Version: 1.0.0
Created: 2026-05-17
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
PARQUET_CACHE = "data_sample.parquet"
PLOTS_DIR = Path("plots")
LOGS_DIR = Path("logs")

PLOTLY_WIDTH = 1200
PLOTLY_HEIGHT = 600
PLOTLY_TEMPLATE = "plotly_white"

# Hour-of-day period definitions: (label, color, hour_range inclusive)
HOUR_PERIODS = [
    ("Night", "#1a237e", range(0, 6)),
    ("Morning Rush", "#e65100", range(6, 10)),
    ("Midday", "#1565c0", range(10, 16)),
    ("Evening Rush", "#b71c1c", range(16, 20)),
    ("Evening", "#6a1b9a", range(20, 24)),
]

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = {
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

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"error_log_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def save_fig(fig: go.Figure, name: str):
    png = PLOTS_DIR / f"{name}.png"
    html = PLOTS_DIR / f"{name}.html"
    fig.write_image(str(png))
    fig.write_html(str(html))
    logger.info(f"Saved: {name}")


def hour_to_period_color(hour: int) -> str:
    """Return the bar color for a given hour."""
    for _, color, hours in HOUR_PERIODS:
        if hour in hours:
            return color
    return "#607d8b"


# ---------------------------------------------------------------------------
# PLOT 17: Trips by hour + avg fare (dual-axis, colored bars)
# ---------------------------------------------------------------------------
def plot_trips_by_hour(df: pd.DataFrame):
    if "pickup_hour" not in df.columns:
        logger.warning("pickup_hour not found — skipping plot 17")
        return

    hourly = (
        df.groupby("pickup_hour")
        .agg(trip_count=("pickup_hour", "count"), avg_fare=("fare_amount", "mean"))
        .reset_index()
        .sort_values("pickup_hour")
    )

    bar_colors = [hour_to_period_color(h) for h in hourly["pickup_hour"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar: trip count per hour
    fig.add_trace(
        go.Bar(
            x=hourly["pickup_hour"],
            y=hourly["trip_count"],
            name="Trip Count",
            marker_color=bar_colors,
            opacity=0.85,
        ),
        secondary_y=False,
    )

    # Line: avg fare per hour
    fig.add_trace(
        go.Scatter(
            x=hourly["pickup_hour"],
            y=hourly["avg_fare"].round(2),
            name="Avg Fare ($)",
            mode="lines+markers",
            line=dict(color="#f9a825", width=2.5),
            marker=dict(size=6, symbol="circle"),
        ),
        secondary_y=True,
    )

    # Period legend annotations
    period_legend = []
    for label, color, _ in HOUR_PERIODS:
        period_legend.append(
            go.Bar(x=[None], y=[None], name=label, marker_color=color, showlegend=True)
        )
    for trace in period_legend:
        fig.add_trace(trace, secondary_y=False)

    fig.update_layout(
        title="Trip Volume & Average Fare by Hour of Day",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="Number of Trips"),
        yaxis2=dict(title="Average Fare ($)", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        bargap=0.1,
    )
    save_fig(fig, "17_trips_by_hour")


# ---------------------------------------------------------------------------
# PLOT 18: Trips by day of week + avg distance (dual-axis)
# ---------------------------------------------------------------------------
def plot_trips_by_dow(df: pd.DataFrame):
    if "pickup_dow" not in df.columns:
        logger.warning("pickup_dow not found — skipping plot 18")
        return

    dow = (
        df.groupby("pickup_dow")
        .agg(trip_count=("pickup_dow", "count"), avg_distance=("trip_distance", "mean"))
        .reset_index()
        .sort_values("pickup_dow")
    )
    dow["day_label"] = dow["pickup_dow"].map(dict(enumerate(DOW_LABELS)))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=dow["day_label"],
            y=dow["trip_count"],
            name="Trip Count",
            marker_color="#1565c0",
            opacity=0.8,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dow["day_label"],
            y=dow["avg_distance"].round(3),
            name="Avg Distance (miles)",
            mode="lines+markers",
            line=dict(color="#e53935", width=2.5),
            marker=dict(size=8, symbol="diamond"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Trip Volume & Avg Distance by Day of Week",
        xaxis=dict(
            title="Day of Week", categoryorder="array", categoryarray=DOW_LABELS
        ),
        yaxis=dict(title="Number of Trips"),
        yaxis2=dict(title="Avg Trip Distance (miles)", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        bargap=0.15,
    )
    save_fig(fig, "18_trips_by_dow")


# ---------------------------------------------------------------------------
# PLOT 19: Trips by month + avg total fare (dual-axis)
# ---------------------------------------------------------------------------
def plot_trips_by_month(df: pd.DataFrame):
    if "pickup_month" not in df.columns:
        logger.warning("pickup_month not found — skipping plot 19")
        return

    monthly = (
        df.groupby("pickup_month")
        .agg(trip_count=("pickup_month", "count"), avg_total=("total_amount", "mean"))
        .reset_index()
        .sort_values("pickup_month")
    )
    monthly["month_label"] = monthly["pickup_month"].map(MONTH_LABELS)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=monthly["month_label"],
            y=monthly["trip_count"],
            name="Trip Count",
            marker_color="#00695c",
            opacity=0.8,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["avg_total"].round(2),
            name="Avg Total Fare ($)",
            mode="lines+markers",
            line=dict(color="#ff6f00", width=2.5),
            marker=dict(size=8, symbol="square"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Monthly Trip Volume & Avg Total Fare (2018)",
        xaxis=dict(
            title="Month",
            categoryorder="array",
            categoryarray=list(MONTH_LABELS.values()),
        ),
        yaxis=dict(title="Number of Trips"),
        yaxis2=dict(title="Avg Total Fare ($)", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        bargap=0.15,
    )
    save_fig(fig, "19_trips_by_month")


# ---------------------------------------------------------------------------
# PLOT 20: Average fare heatmap — hour × day of week
# ---------------------------------------------------------------------------
def plot_fare_heatmap(df: pd.DataFrame):
    if "pickup_hour" not in df.columns or "pickup_dow" not in df.columns:
        logger.warning("pickup_hour/pickup_dow not found — skipping plot 20")
        return

    pivot = (
        df.groupby(["pickup_dow", "pickup_hour"])["fare_amount"]
        .mean()
        .reset_index()
        .pivot(index="pickup_dow", columns="pickup_hour", values="fare_amount")
        .fillna(0)
    )

    # Ensure all 24 hours are represented
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    # Ensure all 7 days present
    for d in range(7):
        if d not in pivot.index:
            pivot.loc[d] = np.nan
    pivot = pivot.sort_index()

    y_labels = [DOW_LABELS[i] for i in pivot.index if i < len(DOW_LABELS)]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values.round(2),
            x=[str(h) for h in pivot.columns],
            y=y_labels,
            colorscale="YlOrRd",
            colorbar=dict(title="Avg Fare ($)"),
            text=np.round(pivot.values, 2),
            texttemplate="$%{text}",
            textfont={"size": 9},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Average Fare Heatmap — Hour × Day of Week",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="Day of Week"),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
    )
    save_fig(fig, "20_fare_by_hour_heatmap")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("PHASE 2: TIME SERIES ANALYSIS")
    logger.info("=" * 60)

    df = pd.read_parquet(PARQUET_CACHE)
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info(f"Loaded {len(df):,} rows")

    plot_trips_by_hour(df)
    plot_trips_by_dow(df)
    plot_trips_by_month(df)
    plot_fare_heatmap(df)

    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE — 4 plots saved to plots/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
