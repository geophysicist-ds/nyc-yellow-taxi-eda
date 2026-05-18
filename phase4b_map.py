"""
phase4b_map.py — NYC borough choropleth map of pickup trip density

Description:
    Fetches NYC borough GeoJSON (tries 3 URLs, falls back to improved hardcoded
    polygons), joins with pickup data via taxi_zone_lookup.csv, and produces a
    Plotly choropleth_map showing trip density per borough.

Input:
    - data_sample.parquet
    - taxi_zone_lookup.csv
    - nyc_boroughs.geojson  (downloaded or created from fallback)

Output:
    - nyc_boroughs.geojson  (cached for re-use)
    - plots/27_borough_trip_density.{png,html}

NOTES:
    - Uses px.choropleth_map (non-deprecated Plotly ≥5.24 API).
    - Tries GeoJSON URLs in order; stops at first valid response.
    - On total network failure uses approximate borough polygons.
    - Run from project root: uv run python phase4b_map.py

Version: 1.1.0
Created: 2026-05-17
Updated: 2026-05-17
"""

import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
PARQUET_CACHE = "data_sample.parquet"
ZONE_LOOKUP = "taxi_zone_lookup.csv"
GEOJSON_FILE = "nyc_boroughs.geojson"
PLOTS_DIR = Path("plots")
LOGS_DIR = Path("logs")

PLOTLY_WIDTH = 1200
PLOTLY_HEIGHT = 650
PLOTLY_TEMPLATE = "plotly_white"

# Try these URLs in order — stop at first valid polygon GeoJSON
GEOJSON_URLS = [
    # NYC Open Data — official borough boundaries
    "https://data.cityofnewyork.us/api/geospatial/7t3b-ywvw?method=export&type=GeoJSON",
    # GitHub mirror — dwillis/nyc-maps
    "https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson",
    # click_that_hood mirror
    "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/new-york-city-boroughs.geojson",
]

# Improved fallback — approximate real borough boundaries (not rectangles)
FALLBACK_GEOJSON: dict = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"boro_name": "Manhattan"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-74.0479, 40.6829],
                        [-74.0200, 40.6996],
                        [-73.9790, 40.6960],
                        [-73.9714, 40.7282],
                        [-73.9730, 40.7614],
                        [-73.9580, 40.7850],
                        [-73.9340, 40.8510],
                        [-73.9100, 40.8780],
                        [-73.9340, 40.8820],
                        [-73.9500, 40.8700],
                        [-73.9650, 40.8300],
                        [-73.9800, 40.7960],
                        [-73.9990, 40.7200],
                        [-74.0100, 40.7050],
                        [-74.0200, 40.6996],
                        [-74.0479, 40.6829],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"boro_name": "Brooklyn"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-74.0420, 40.5700],
                        [-73.8330, 40.5820],
                        [-73.8600, 40.6880],
                        [-73.9100, 40.7050],
                        [-73.9790, 40.6960],
                        [-74.0200, 40.6996],
                        [-74.0420, 40.5700],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"boro_name": "Queens"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-73.9620, 40.5730],
                        [-73.7000, 40.5730],
                        [-73.7000, 40.8000],
                        [-73.8000, 40.8000],
                        [-73.8600, 40.7500],
                        [-73.9300, 40.7300],
                        [-73.9620, 40.5730],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"boro_name": "Bronx"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-73.9330, 40.7960],
                        [-73.7480, 40.7960],
                        [-73.7480, 40.9150],
                        [-73.8330, 40.9150],
                        [-73.9100, 40.8780],
                        [-73.9330, 40.7960],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"boro_name": "Staten Island"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-74.2590, 40.4960],
                        [-74.0500, 40.4960],
                        [-74.0500, 40.6510],
                        [-74.1500, 40.6510],
                        [-74.2590, 40.5800],
                        [-74.2590, 40.4960],
                    ]
                ],
            },
        },
    ],
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOG_FILE = (
    LOGS_DIR / f"error_log_phase4b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
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
# GEOJSON: DOWNLOAD OR CACHE
# ---------------------------------------------------------------------------
def is_valid_polygon_geojson(geo: dict) -> bool:
    """Return True if geo has at least one Polygon/MultiPolygon feature."""
    features = geo.get("features", [])
    if not features:
        return False
    return any(
        f.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon")
        for f in features
    )


def fetch_geojson() -> tuple[dict, str]:
    """Try each URL in GEOJSON_URLS; return (geojson, source_label)."""
    # Delete cached file to force re-fetch (in case previous cache was bad fallback)
    if Path(GEOJSON_FILE).exists():
        cached = json.loads(Path(GEOJSON_FILE).read_text())
        if is_valid_polygon_geojson(cached):
            logger.info(f"Using cached GeoJSON: {GEOJSON_FILE}")
            return cached, "cache"
        else:
            logger.info("Cached GeoJSON invalid — re-fetching")
            Path(GEOJSON_FILE).unlink()

    for url in GEOJSON_URLS:
        try:
            logger.info(f"Trying: {url}")
            with urllib.request.urlopen(url, timeout=10) as resp:
                geo = json.loads(resp.read().decode("utf-8"))
            if is_valid_polygon_geojson(geo):
                Path(GEOJSON_FILE).write_text(json.dumps(geo))
                logger.info(f"SUCCESS — GeoJSON from: {url}")
                return geo, url
            else:
                logger.warning(f"Response from {url} has no polygon features")
        except Exception as e:
            logger.warning(f"Failed {url}: {e}")

    logger.warning("All GeoJSON URLs failed — using improved fallback polygons")
    Path(GEOJSON_FILE).write_text(json.dumps(FALLBACK_GEOJSON))
    return FALLBACK_GEOJSON, "fallback"


# ---------------------------------------------------------------------------
# DETECT BOROUGH NAME KEY
# ---------------------------------------------------------------------------
def detect_borough_key(geo: dict) -> str:
    """Find the feature property key that holds borough names."""
    candidates = ["boro_name", "BoroName", "borough", "name", "NAME", "BORO_NM"]
    sample_props = geo["features"][0].get("properties", {})
    for key in candidates:
        if key in sample_props:
            logger.info(
                f"Borough key detected: '{key}' → sample value: '{sample_props[key]}'"
            )
            return key
    # Last resort: first string-valued property
    for key, val in sample_props.items():
        if isinstance(val, str):
            logger.warning(f"Borough key guessed from first string property: '{key}'")
            return key
    raise ValueError(f"Cannot detect borough key. Properties: {sample_props}")


def normalise_to_boro_name(geo: dict, src_key: str) -> dict:
    """Copy src_key value into 'boro_name' for consistent featureidkey."""
    if src_key == "boro_name":
        return geo
    for feat in geo["features"]:
        feat["properties"]["boro_name"] = feat["properties"].get(src_key, "")
    return geo


# ---------------------------------------------------------------------------
# BUILD BOROUGH TRIP COUNTS
# ---------------------------------------------------------------------------
def build_borough_counts(df: pd.DataFrame) -> pd.DataFrame:
    zones = pd.read_csv(ZONE_LOOKUP)
    zones.columns = [c.strip() for c in zones.columns]
    zones = zones.rename(columns={"LocationID": "PULocationID"})

    merged = df[["PULocationID"]].merge(
        zones[["PULocationID", "Borough"]], on="PULocationID", how="left"
    )
    merged["Borough"] = merged["Borough"].fillna("Unknown")

    counts = merged["Borough"].value_counts().reset_index()
    counts.columns = ["boro_name", "trip_count"]
    total = counts["trip_count"].sum()
    counts["pct"] = (counts["trip_count"] / total * 100).round(2)
    counts["hover_text"] = counts.apply(
        lambda r: f"{r['trip_count']:,} trips ({r['pct']:.1f}%)", axis=1
    )
    logger.info(
        "Borough counts:\n"
        + counts[["boro_name", "trip_count", "pct"]].to_string(index=False)
    )
    return counts


# ---------------------------------------------------------------------------
# PLOT 27: BOROUGH CHOROPLETH MAP
# ---------------------------------------------------------------------------
def plot_borough_choropleth(df: pd.DataFrame):
    geo, source = fetch_geojson()
    src_key = detect_borough_key(geo)
    geo = normalise_to_boro_name(geo, src_key)

    counts = build_borough_counts(df)

    # Match rate: how many trip data boroughs are in GeoJSON
    geo_boroughs = {f["properties"]["boro_name"] for f in geo["features"]}
    data_boroughs = set(counts["boro_name"]) - {"Unknown"}
    matched = data_boroughs & geo_boroughs
    logger.info(
        f"Borough match rate: {len(matched)}/{len(data_boroughs)} — "
        f"matched={sorted(matched)}, unmatched={sorted(data_boroughs - geo_boroughs)}"
    )

    # Filter to rows that have a matching GeoJSON feature (excludes Unknown)
    plot_counts = counts[counts["boro_name"].isin(geo_boroughs)].copy()

    fig = px.choropleth_map(
        plot_counts,
        geojson=geo,
        locations="boro_name",
        featureidkey="properties.boro_name",
        color="trip_count",
        color_continuous_scale="Blues",
        map_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.75,
        hover_name="boro_name",
        hover_data={"trip_count": True, "pct": True, "boro_name": False},
        labels={"trip_count": "Pickups", "pct": "% of Total"},
        title="NYC Yellow Taxi — Pickup Trip Density by Borough (2018)",
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Pickups"),
        template=PLOTLY_TEMPLATE,
    )

    src_label = source if source in ("cache", "fallback") else source.split("/")[-1]
    logger.info(f"Map generated — GeoJSON source: {src_label}")
    save_fig(fig, "27_borough_trip_density")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("PHASE 4B: BOROUGH CHOROPLETH MAP")
    logger.info("=" * 60)

    df = pd.read_parquet(PARQUET_CACHE)
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"Loaded {len(df):,} rows")

    plot_borough_choropleth(df)

    logger.info("=" * 60)
    logger.info("PHASE 4B COMPLETE — plot 27 saved to plots/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
