"""
phase4_geographic.py — Geographic analysis visualizations

Description:
    Produces three geographic plots using PULocationID/DOLocationID joined
    to the NYC TLC taxi zone lookup:
      24_top20_pickup_zones  : Horizontal bar chart, top 20 pickup zones
      25_top20_dropoff_zones : Horizontal bar chart, top 20 dropoff zones
      26_borough_sankey      : Sankey diagram of borough-to-borough trip flow

Input:
    - data_sample.parquet
    - taxi_zone_lookup.csv

Output:
    - plots/24_top20_pickup_zones.{png,html}
    - plots/25_top20_dropoff_zones.{png,html}
    - plots/26_borough_sankey.{png,html}

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

# Sankey link colors (semi-transparent) keyed by source borough
SANKEY_LINK_COLORS = {
    "Manhattan": "rgba(21,101,192,0.45)",
    "Queens": "rgba(230,81,0,0.45)",
    "Brooklyn": "rgba(46,125,50,0.45)",
    "Bronx": "rgba(198,40,40,0.45)",
    "EWR": "rgba(106,27,154,0.45)",
    "Staten Island": "rgba(0,105,92,0.45)",
    "Unknown": "rgba(117,117,117,0.35)",
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"error_log_phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(ZONE_LOOKUP)
    zones.columns = [c.strip() for c in zones.columns]
    return zones


# ---------------------------------------------------------------------------
# PLOT 24 & 25 helper: top-20 zone horizontal bar chart
# ---------------------------------------------------------------------------
def plot_top20_zones(
    df: pd.DataFrame, zones: pd.DataFrame, loc_col: str, plot_num: int, title: str
):
    merged = df[[loc_col]].merge(
        zones[["LocationID", "Zone", "Borough"]].rename(
            columns={"LocationID": loc_col}
        ),
        on=loc_col,
        how="left",
    )
    merged["Zone"] = merged["Zone"].fillna("Unknown")
    merged["Borough"] = merged["Borough"].fillna("Unknown")

    top20 = (
        merged.groupby(["Zone", "Borough"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )
    # Reverse for horizontal bar (most at top)
    top20 = top20.sort_values("count", ascending=True)

    bar_colors = [BOROUGH_COLORS.get(b, "#607d8b") for b in top20["Borough"]]

    fig = go.Figure(
        go.Bar(
            x=top20["count"],
            y=top20["Zone"],
            orientation="h",
            marker_color=bar_colors,
            text=top20["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
            cliponaxis=False,
        )
    )

    # Borough legend traces (invisible bars)
    seen = set()
    for borough, color in BOROUGH_COLORS.items():
        if borough in top20["Borough"].values and borough not in seen:
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    orientation="h",
                    marker_color=color,
                    name=borough,
                )
            )
            seen.add(borough)

    max_count = int(top20["count"].max())
    fig.update_layout(
        title=title,
        xaxis=dict(title="Number of Trips", range=[0, max_count * 1.18]),
        yaxis=dict(title="Zone", automargin=True),
        width=PLOTLY_WIDTH,
        height=max(PLOTLY_HEIGHT, 80 + 28 * 20),
        template=PLOTLY_TEMPLATE,
        legend=dict(title="Borough", orientation="v"),
        barmode="overlay",
        bargap=0.15,
    )
    name = f"{plot_num}_top20_{'pickup' if 'PU' in loc_col else 'dropoff'}_zones"
    save_fig(fig, name)


# ---------------------------------------------------------------------------
# PLOT 26: Borough-to-Borough Sankey
# ---------------------------------------------------------------------------
def plot_borough_sankey(df: pd.DataFrame, zones: pd.DataFrame):
    # Join pickup borough
    pu = df[["PULocationID", "DOLocationID"]].merge(
        zones[["LocationID", "Borough"]].rename(
            columns={"LocationID": "PULocationID", "Borough": "PU_Borough"}
        ),
        on="PULocationID",
        how="left",
    )
    # Join dropoff borough
    full = pu.merge(
        zones[["LocationID", "Borough"]].rename(
            columns={"LocationID": "DOLocationID", "Borough": "DO_Borough"}
        ),
        on="DOLocationID",
        how="left",
    )
    full["PU_Borough"] = full["PU_Borough"].fillna("Unknown")
    full["DO_Borough"] = full["DO_Borough"].fillna("Unknown")

    # Aggregate flow counts
    flows = (
        full.groupby(["PU_Borough", "DO_Borough"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    logger.info(f"Sankey top-15 flows:\n{flows.to_string(index=False)}")

    # Build node list: left side = pickup boroughs, right side = dropoff boroughs
    # Use suffix to keep source/target nodes distinct even if same borough name
    pu_nodes = [f"{b} (pickup)" for b in flows["PU_Borough"].unique()]
    do_nodes = [f"{b} (dropoff)" for b in flows["DO_Borough"].unique()]
    all_nodes = pu_nodes + do_nodes

    node_idx = {n: i for i, n in enumerate(all_nodes)}

    sources, targets, values, link_colors = [], [], [], []
    for _, row in flows.iterrows():
        src_label = f"{row['PU_Borough']} (pickup)"
        tgt_label = f"{row['DO_Borough']} (dropoff)"
        sources.append(node_idx[src_label])
        targets.append(node_idx[tgt_label])
        values.append(int(row["count"]))
        link_colors.append(
            SANKEY_LINK_COLORS.get(row["PU_Borough"], "rgba(150,150,150,0.4)")
        )

    # Node colors: pickup nodes get borough color, dropoff nodes get lighter tint
    node_colors = []
    for node in all_nodes:
        borough = node.split(" (")[0]
        base = BOROUGH_COLORS.get(borough, "#757575")
        node_colors.append(base)

    # Node x/y positions: pickups on left (x=0.01), dropoffs on right (x=0.99)
    n_pu = len(pu_nodes)
    n_do = len(do_nodes)
    node_x = [0.01] * n_pu + [0.99] * n_do
    node_y_pu = [round((i + 1) / (n_pu + 1), 3) for i in range(n_pu)]
    node_y_do = [round((i + 1) / (n_do + 1), 3) for i in range(n_do)]
    node_y = node_y_pu + node_y_do

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=22,
                line=dict(color="white", width=0.5),
                label=all_nodes,
                color=node_colors,
                x=node_x,
                y=node_y,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )
    fig.update_layout(
        title="Borough-to-Borough Trip Flow (Sankey) — Top 15 Flows",
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
        font=dict(size=11),
    )
    save_fig(fig, "26_borough_sankey")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("PHASE 4: GEOGRAPHIC ANALYSIS")
    logger.info("=" * 60)

    df = pd.read_parquet(PARQUET_CACHE)
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"Loaded {len(df):,} rows")

    zones = load_zones()
    logger.info(f"Zone lookup: {len(zones)} zones")

    plot_top20_zones(df, zones, "PULocationID", 24, "Top 20 Pickup Zones")
    plot_top20_zones(df, zones, "DOLocationID", 25, "Top 20 Dropoff Zones")
    plot_borough_sankey(df, zones)

    logger.info("=" * 60)
    logger.info("PHASE 4 COMPLETE — 3 plots saved to plots/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
