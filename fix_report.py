"""
fix_report.py — Patch eda_report.html with three improvements:
  1. Fix descriptive statistics table (JSON was column-oriented, template was stat-oriented)
  2. Add "Deep Dive Analysis" section (negative fares, passenger dist, zero-tip by payment)
  3. Add borough-level pickup bar chart using taxi_zone_lookup.csv join

Input:  data_sample.parquet, analysis_results.json, taxi_zone_lookup.csv
Output: eda_report.html (overwritten), plots/borough_pickups.{png,html}

Version: 1.0.0
Created: 2026-05-17
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
PARQUET_CACHE = "data_sample.parquet"
RESULTS_FILE = "analysis_results.json"
REPORT_FILE = "eda_report.html"
PLOTS_DIR = Path("plots")
LOGS_DIR = Path("logs")
ZONE_LOOKUP = "taxi_zone_lookup.csv"

PLOTLY_WIDTH = 1200
PLOTLY_HEIGHT = 600
PLOTLY_TEMPLATE = "plotly_white"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"error_log_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPER: Build descriptive stats table rows (fix #1)
# Actual JSON structure: {col_name: {stat: value, ...}, ...}
# ---------------------------------------------------------------------------
def build_stats_rows(desc_stats: dict) -> str:
    """Build HTML table rows from column-oriented descriptive stats JSON."""
    rows = ""
    for col, stats in desc_stats.items():

        def fmt(key):
            val = stats.get(key)
            if val is None:
                return "—"
            try:
                return f"{float(val):.4f}"
            except Exception:
                return str(val)

        rows += f"""<tr>
          <td><code>{col}</code></td>
          <td>{fmt("mean")}</td>
          <td>{fmt("50%")}</td>
          <td>{fmt("min")}</td>
          <td>{fmt("max")}</td>
          <td>{fmt("std")}</td>
          <td>{fmt("1%")}</td>
          <td>{fmt("99%")}</td>
        </tr>"""
    return rows


# ---------------------------------------------------------------------------
# DEEP DIVE ANALYSIS (fix #2)
# ---------------------------------------------------------------------------
def build_deep_dive(df: pd.DataFrame) -> str:
    """Generate HTML for the Deep Dive Analysis section."""

    # --- 2a: Negative fares ---
    neg_fares = df[df["fare_amount"] < 0]
    n_neg = len(neg_fares)
    pct_neg = n_neg / len(df) * 100
    neg_sample_html = ""
    if n_neg > 0:
        sample_cols = [
            c
            for c in [
                "VendorID",
                "tpep_pickup_datetime",
                "trip_distance",
                "fare_amount",
                "total_amount",
                "payment_type",
            ]
            if c in df.columns
        ]
        sample = neg_fares[sample_cols].head(5)
        neg_sample_html = sample.to_html(
            index=False,
            border=0,
            classes="data-table",
            justify="left",
        )

    # --- 2b: Passenger count distribution ---
    pc = df["passenger_count"].value_counts(dropna=False).sort_index()
    pc_pct = (pc / len(df) * 100).round(2)
    pc_rows = ""
    for pax, cnt in pc.items():
        pc_rows += f"<tr><td>{pax}</td><td>{cnt:,}</td><td>{pc_pct[pax]:.2f}%</td></tr>"

    # --- 2c: Zero-tip breakdown by payment_type ---
    payment_labels = {
        1: "Credit Card",
        2: "Cash",
        3: "No Charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided Trip",
    }
    tip_df = df[["tip_amount", "payment_type"]].dropna()
    zero_tip_rows = ""
    for ptype, grp in tip_df.groupby("payment_type"):
        n_total = len(grp)
        n_zero = (grp["tip_amount"] == 0).sum()
        pct_zero = n_zero / n_total * 100
        label = payment_labels.get(int(ptype), f"Type {ptype}")
        zero_tip_rows += (
            f"<tr><td>{label}</td><td>{n_total:,}</td>"
            f"<td>{n_zero:,}</td><td>{pct_zero:.1f}%</td></tr>"
        )

    overall_zero = (tip_df["tip_amount"] == 0).sum()
    overall_pct = overall_zero / len(tip_df) * 100

    section = f"""
<div class="section">
  <h2>Deep Dive Analysis</h2>

  <!-- 2a: Negative Fares -->
  <h3 style="font-size:1.1rem;color:#c62828;margin-bottom:8px;">
    Negative Fare Amounts
  </h3>
  <p style="margin-bottom:10px;font-size:0.9rem;">
    <strong>{n_neg:,} trips</strong> ({pct_neg:.3f}% of sample) have
    <code>fare_amount &lt; 0</code> — likely reversals or adjustments.
  </p>
  {
        "<div style='overflow-x:auto;margin-bottom:20px;'>" + neg_sample_html + "</div>"
        if n_neg > 0
        else "<p style='color:#777;font-size:0.9rem;margin-bottom:20px;'>No negative fares found.</p>"
    }

  <!-- 2b: Passenger count distribution -->
  <h3 style="font-size:1.1rem;color:#1565c0;margin:20px 0 8px;">
    Passenger Count Distribution
  </h3>
  <table style="max-width:400px;">
    <tr><th>Passengers</th><th>Trips</th><th>%</th></tr>
    {pc_rows}
  </table>

  <!-- 2c: Zero-tip breakdown -->
  <h3 style="font-size:1.1rem;color:#2e7d32;margin:24px 0 8px;">
    Zero-Tip Trips by Payment Type
  </h3>
  <p style="margin-bottom:10px;font-size:0.9rem;">
    Overall: <strong>{overall_zero:,} trips ({overall_pct:.1f}%)</strong> have
    <code>tip_amount = 0</code>.
    Note: cash trips (type 2) almost always show $0 tip as tips are not captured electronically.
  </p>
  <table style="max-width:600px;">
    <tr><th>Payment Type</th><th>Total Trips</th><th>Zero-Tip Trips</th><th>Zero-Tip %</th></tr>
    {zero_tip_rows}
  </table>
</div>"""
    return section


# ---------------------------------------------------------------------------
# BOROUGH PICKUP CHART (fix #3)
# ---------------------------------------------------------------------------
def build_borough_chart(df: pd.DataFrame) -> tuple[str, str]:
    """Join PULocationID with taxi zone lookup, plot borough-level pickup bar chart."""
    zones = pd.read_csv(ZONE_LOOKUP)
    # Normalise column names — the downloaded file may vary
    zones.columns = [c.strip() for c in zones.columns]
    logger.info(f"Zone lookup columns: {zones.columns.tolist()}")

    # Expected: LocationID, Borough, Zone, service_zone
    zones = zones.rename(columns={"LocationID": "PULocationID"})

    merged = df[["PULocationID"]].merge(
        zones[["PULocationID", "Borough"]], on="PULocationID", how="left"
    )
    borough_counts = merged["Borough"].fillna("Unknown").value_counts().reset_index()
    borough_counts.columns = ["borough", "trip_count"]
    borough_counts["pct"] = (borough_counts["trip_count"] / len(merged) * 100).round(2)

    fig = px.bar(
        borough_counts.sort_values("trip_count", ascending=False),
        x="borough",
        y="trip_count",
        text=borough_counts.sort_values("trip_count", ascending=False)["pct"].map(
            lambda x: f"{x:.1f}%"
        ),
        title="Pickup Trips by NYC Borough",
        labels={"borough": "Borough", "trip_count": "Number of Pickups"},
        color="borough",
        color_discrete_sequence=px.colors.qualitative.Set1,
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)

    png_path = PLOTS_DIR / "16_borough_pickups.png"
    html_path = PLOTS_DIR / "16_borough_pickups.html"
    fig.write_image(str(png_path))
    fig.write_html(str(html_path))
    logger.info("Borough pickup chart saved.")

    # Build borough section HTML
    borough_rows = ""
    for _, row in borough_counts.sort_values("trip_count", ascending=False).iterrows():
        borough_rows += (
            f"<tr><td>{row['borough']}</td>"
            f"<td>{row['trip_count']:,}</td>"
            f"<td>{row['pct']:.2f}%</td></tr>"
        )

    section = f"""
<div class="section">
  <h2>Geographic Analysis — Pickup Distribution by Borough</h2>
  <p style="font-size:0.9rem;color:#555;margin-bottom:16px;">
    <code>PULocationID</code> joined with NYC TLC taxi zone lookup to derive borough.
    This dataset does not contain raw lat/lon coordinates, so borough is the finest
    geographic granularity available.
  </p>
  <div class="plot-card">
    <h3>16 Borough Pickups</h3>
    <p class="plot-desc">Pickup trip count per NYC borough</p>
    <iframe src="plots/16_borough_pickups.html" width="100%" height="650px"
            frameborder="0" loading="lazy" style="border-radius:8px;"></iframe>
  </div>
  <table style="max-width:400px;margin-top:16px;">
    <tr><th>Borough</th><th>Pickups</th><th>%</th></tr>
    {borough_rows}
  </table>
</div>"""
    return section, str(html_path)


# ---------------------------------------------------------------------------
# ASSEMBLE UPDATED REPORT
# ---------------------------------------------------------------------------
def rebuild_report():
    logger.info("Loading data and results...")
    df = pd.read_parquet(PARQUET_CACHE)
    # Ensure datetimes are parsed
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # --- Fix 1: Descriptive stats ---
    desc_stats = results.get("descriptive_stats", {})
    stats_rows = build_stats_rows(desc_stats)

    # --- Fix 2: Deep dive ---
    deep_dive_section = build_deep_dive(df)

    # --- Fix 3: Borough chart ---
    borough_section, _ = build_borough_chart(df)

    # --- Rebuild remaining report parts (same as eda_viz.py build_report) ---
    drop_log = results.get("drop_log", {})
    sample_shape = results.get("sample_shape", [0, 0])
    skipped = results.get("skipped_steps", [])
    outlier_info = results.get("outlier_analysis", {})
    data_quality = results.get("data_quality", {})

    # Collect all plots from /plots dir
    plot_sections = ""
    for html_path in sorted(PLOTS_DIR.glob("*.html")):
        name = html_path.stem
        plot_sections += f"""
<div class="plot-card">
  <h3>{name.replace("_", " ").title()}</h3>
  <iframe src="plots/{html_path.name}" width="100%" height="650px" frameborder="0"
          loading="lazy" style="border-radius:8px;"></iframe>
</div>"""

    # Outlier table
    outlier_rows = ""
    for col, info in outlier_info.items():
        outlier_rows += f"""<tr>
          <td>{col}</td><td>{info["q1"]}</td><td>{info["q3"]}</td>
          <td>{info["lower_fence"]}</td><td>{info["upper_fence"]}</td>
          <td>{info["n_outliers"]:,}</td><td>{info["outlier_pct"]}%</td>
        </tr>"""

    # Dropped columns
    dropped_cols_rows = ""
    for item in drop_log.get("dropped_columns", []):
        dropped_cols_rows += (
            f"<tr><td>{item['column']}</td><td>{item['null_pct'] * 100:.1f}%</td></tr>"
        )
    if not dropped_cols_rows:
        dropped_cols_rows = "<tr><td colspan='2'>None — all columns retained</td></tr>"

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
      not raw lat/lon coordinates. Borough-level analysis is performed via zone lookup join below.
      Raw folium heatmaps are not applicable without coordinates.
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
    .section h3 {{ margin-top: 18px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 16px; margin-bottom: 20px; }}
    .kpi-card {{ background: #f0f4ff; border-radius: 10px; padding: 20px; text-align: center;
                 border-left: 4px solid #3f51b5; }}
    .kpi-card .value {{ font-size: 1.8rem; font-weight: 700; color: #3f51b5; }}
    .kpi-card .label {{ font-size: 0.85rem; color: #555; margin-top: 4px; }}
    table, .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
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
    code {{ background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
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
    <h2>Null Handling &amp; Data Quality</h2>
    <h3 style="font-size:1rem;margin-bottom:10px;">Dropped Columns (&gt;40% null)</h3>
    <table>
      <tr><th>Column</th><th>Null %</th></tr>
      {dropped_cols_rows}
    </table>
    <p style="margin-top:12px;font-size:0.9rem;color:#555;">
      Rows dropped (remaining nulls): <strong>{drop_log.get("dropped_rows", 0):,}</strong>
    </p>
  </div>

  <!-- Descriptive Statistics (fixed) -->
  <div class="section">
    <h2>Descriptive Statistics</h2>
    <div style="overflow-x:auto;">
      <table>
        <tr><th>Column</th><th>Mean</th><th>Median (50%)</th><th>Min</th>
            <th>Max</th><th>Std Dev</th><th>1st Pct</th><th>99th Pct</th></tr>
        {stats_rows}
      </table>
    </div>
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

  {deep_dive_section}

  {borough_section}

  <!-- All Visualizations -->
  <div class="section">
    <h2>Visualizations</h2>
    {plot_sections}
  </div>

</div>

<footer>
  2018 NYC Yellow Taxi EDA &mdash; Auto-generated by eda_viz.py / fix_report.py
</footer>
</body>
</html>"""

    with open(REPORT_FILE, "w") as f:
        f.write(html)
    logger.info(f"Report rebuilt: {REPORT_FILE}")


if __name__ == "__main__":
    rebuild_report()
    logger.info("Done.")
