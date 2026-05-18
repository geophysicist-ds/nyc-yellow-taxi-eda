"""
phase3_fare_tip.py — Fare & Tip deep dive visualizations

Description:
    Produces three plots exploring fare efficiency and tipping behavior:
      21_tip_pct_distribution  : Tip % histogram for credit card trips with
                                 mean/median/20% reference lines
      22_fare_per_mile_by_borough : Box plot of fare-per-mile by pickup borough
      23_surge_by_hour         : Avg extra charge by hour with color thresholds

Input:
    - data_sample.parquet
    - taxi_zone_lookup.csv

Output:
    - plots/21_tip_pct_distribution.{png,html}
    - plots/22_fare_per_mile_by_borough.{png,html}
    - plots/23_surge_by_hour.{png,html}

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
ZONE_LOOKUP = "taxi_zone_lookup.csv"
PLOTS_DIR = Path("plots")
LOGS_DIR = Path("logs")

PLOTLY_WIDTH = 1200
PLOTLY_HEIGHT = 600
PLOTLY_TEMPLATE = "plotly_white"

BOROUGH_COLORS = {
    "Manhattan": "#1565c0",
    "Queens": "#e65100",
    "Brooklyn": "#2e7d32",
    "Bronx": "#c62828",
    "EWR": "#6a1b9a",
    "Staten Island": "#00695c",
    "Unknown": "#757575",
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"error_log_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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


# ---------------------------------------------------------------------------
# PLOT 21: Tip % distribution — credit card trips only
# ---------------------------------------------------------------------------
def plot_tip_pct_distribution(df: pd.DataFrame):
    cc = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0)].copy()
    cc["tip_pct"] = cc["tip_amount"] / cc["fare_amount"] * 100
    # Clip to 0–50% range for histogram
    cc = cc[(cc["tip_pct"] >= 0) & (cc["tip_pct"] <= 50)]

    mean_tip = float(cc["tip_pct"].mean())
    median_tip = float(cc["tip_pct"].median())
    standard_20 = 20.0

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=cc["tip_pct"],
            xbins=dict(start=0, end=50, size=1),
            name="Trip Count",
            marker_color="#42a5f5",
            opacity=0.75,
        )
    )

    # Reference vertical lines — valid dash values: solid, dot, dash, longdash, dashdot
    line_defs = [
        (mean_tip, "red", "dash", f"Mean: {mean_tip:.1f}%"),
        (median_tip, "#2e7d32", "dashdot", f"Median: {median_tip:.1f}%"),
        (standard_20, "#1565c0", "dot", "20% Standard"),
    ]
    for x_val, color, dash, label in line_defs:
        fig.add_vline(
            x=x_val,
            line=dict(color=color, width=2, dash=dash),
            annotation_text=label,
            annotation_position="top right" if x_val >= 20 else "top left",
            annotation_font=dict(color=color, size=11),
        )

    fig.update_layout(
        title="Tip % Distribution — Credit Card Trips Only",
        xaxis=dict(title="Tip % of Fare", ticksuffix="%", range=[0, 50]),
        yaxis=dict(title="Number of Trips"),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        bargap=0.02,
    )
    logger.info(
        f"Tip% — CC trips: {len(cc):,}, mean={mean_tip:.2f}%, median={median_tip:.2f}%"
    )
    save_fig(fig, "21_tip_pct_distribution")


# ---------------------------------------------------------------------------
# PLOT 22: Fare per mile by pickup borough — box plot + mean diamond
# ---------------------------------------------------------------------------
def plot_fare_per_mile_by_borough(df: pd.DataFrame):
    zones = pd.read_csv(ZONE_LOOKUP)
    zones.columns = [c.strip() for c in zones.columns]
    zones = zones.rename(columns={"LocationID": "PULocationID"})

    merged = df.merge(zones[["PULocationID", "Borough"]], on="PULocationID", how="left")
    merged["Borough"] = merged["Borough"].fillna("Unknown")

    # Compute fare per mile; exclude zero-distance trips
    work = merged[(merged["trip_distance"] > 0) & (merged["fare_amount"] > 0)].copy()
    work["fare_per_mile"] = work["fare_amount"] / work["trip_distance"]
    work = work[(work["fare_per_mile"] >= 1) & (work["fare_per_mile"] <= 25)]

    boroughs = (
        work.groupby("Borough")["fare_per_mile"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = go.Figure()
    for borough in boroughs:
        subset = work[work["Borough"] == borough]["fare_per_mile"]
        color = BOROUGH_COLORS.get(borough, "#607d8b")
        mean_val = float(subset.mean())

        fig.add_trace(
            go.Box(
                y=subset,
                name=borough,
                marker_color=color,
                boxpoints=False,
                line_width=1.5,
            )
        )
        # Mean diamond overlay
        fig.add_trace(
            go.Scatter(
                x=[borough],
                y=[mean_val],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color=color,
                    line=dict(width=1, color="black"),
                ),
                showlegend=False,
                hovertemplate=f"Mean: ${mean_val:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Fare per Mile by Pickup Borough",
        xaxis=dict(title="Borough"),
        yaxis=dict(title="Fare per Mile ($)", range=[0, 25]),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        showlegend=True,
    )
    save_fig(fig, "22_fare_per_mile_by_borough")


# ---------------------------------------------------------------------------
# PLOT 23: Average extra/surge charge by hour of day
# ---------------------------------------------------------------------------
def plot_surge_by_hour(df: pd.DataFrame):
    if "pickup_hour" not in df.columns or "extra" not in df.columns:
        logger.warning("Required columns missing — skipping plot 23")
        return

    hourly = (
        df.groupby("pickup_hour")
        .agg(avg_extra=("extra", "mean"), trip_count=("pickup_hour", "count"))
        .reset_index()
        .sort_values("pickup_hour")
    )

    # Color bars by avg_extra threshold
    def bar_color(v: float) -> str:
        if v < 0.3:
            return "#2e7d32"
        if v <= 0.5:
            return "#f9a825"
        return "#c62828"

    bar_colors = [bar_color(v) for v in hourly["avg_extra"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=hourly["pickup_hour"],
            y=hourly["avg_extra"].round(4),
            name="Avg Extra ($)",
            marker_color=bar_colors,
            opacity=0.85,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=hourly["pickup_hour"],
            y=hourly["trip_count"],
            name="Trip Count",
            mode="lines+markers",
            line=dict(color="#1565c0", width=2),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )

    # Threshold reference lines
    for threshold, color, label in [(0.3, "#f9a825", "0.30"), (0.5, "#c62828", "0.50")]:
        fig.add_hline(
            y=threshold,
            line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=f"${label}",
            annotation_position="right",
            secondary_y=False,
        )

    fig.update_layout(
        title="Average Surge/Extra Charge by Hour of Day",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis=dict(title="Avg Extra Charge ($)"),
        yaxis2=dict(title="Trip Count", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        bargap=0.1,
    )
    save_fig(fig, "23_surge_by_hour")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: FARE & TIP DEEP DIVE")
    logger.info("=" * 60)

    df = pd.read_parquet(PARQUET_CACHE)
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"Loaded {len(df):,} rows")

    plot_tip_pct_distribution(df)
    plot_fare_per_mile_by_borough(df)
    plot_surge_by_hour(df)

    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETE — 3 plots saved to plots/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
