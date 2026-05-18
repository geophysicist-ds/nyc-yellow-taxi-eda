# NYC Yellow Taxi EDA (2018)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-NYC%20TLC%20Public-orange)

Automated exploratory data analysis of the 2018 NYC Yellow Taxi dataset — a 14 GB source file
containing 112 million trip records. A stratified 560K-row random sample is drawn using a
lambda-based approach, cached to Parquet, and put through a five-phase analysis pipeline that
produces 27 interactive Plotly visualizations and a polished, self-contained `eda_report.html`.

---

## Key Findings

- **Manhattan dominates** — 91% of all sampled pickups originate in Manhattan; all top-20
  pickup zones are Manhattan TLC zones.
- **20% tip nudge effect** — the payment terminal's default "20%" button creates a sharp spike
  in the tip distribution; the median credit card tip is exactly **22%** of the fare.
- **Late-night surge** — average `extra` charge exceeds **$0.50** between midnight and 6 AM,
  consistent with the NYC TLC overnight surcharge; fares run roughly 18% higher than midday.
- **EWR flat-rate variance** — Newark Airport trips show the widest fare-per-mile spread of any
  borough, driven by distance-independent flat-rate pricing plus traffic-dependent extras.
- **Friday/Saturday evening peak** — the hour × day-of-week heatmap shows the highest trip
  density at 6–9 PM on Fridays and Saturdays, with a secondary peak during weekday morning rush.

---

## Project Structure

```
nyc-yellow-taxi-eda/
├── eda_main.py            # Entry point: sampling, null handling, statistical analysis
├── eda_viz.py             # Core EDA visualizations (plots 01–16)
├── phase1_quality.py      # Data quality audit (RatecodeID=99, neg durations, etc.)
├── phase2_timeseries.py   # Time series analysis (plots 17–20)
├── phase3_fare_tip.py     # Fare & tip deep dive (plots 21–23)
├── phase3_4_inject.py     # Report injection helper (Phase 3 & 4 sections)
├── phase4_geographic.py   # Geographic zone analysis (plots 24–26)
├── phase4b_map.py         # Borough choropleth map (plot 27)
├── phase5_polish.py       # Report polish & final assembly
├── fix_report.py          # Report layout fixes (stats table, deep dive, borough bar)
├── plots/                 # 27 visualizations (PNG + interactive HTML)
├── eda_report.html        # Final self-contained interactive report
├── nyc_boroughs.geojson   # NYC borough polygons (from NYC TLC via GitHub mirror)
├── requirements.txt       # Python dependencies
└── data_schema.json       # Column schema cache (dtypes, null counts, sample values)
```

---

## How to Run

```bash
# 1. Install uv (if not installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# 2. Download the 2018 Yellow Taxi trip data from NYC TLC and place in project root:
#    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#    Expected filename: 2018_Yellow_Taxi_Trip_Data_20260516.csv

# 3. Create virtual environment and install dependencies
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# 4. Run the full pipeline in order:
uv run python eda_main.py           # Sample 500K rows, analyze, cache to parquet
uv run python eda_viz.py            # Generate plots 01–16

uv run python phase1_quality.py     # Data quality audit → quality_results.json
uv run python phase2_timeseries.py  # Time series plots 17–20
uv run python phase3_fare_tip.py    # Fare & tip plots 21–23
uv run python phase4_geographic.py  # Geographic plots 24–26
uv run python phase4b_map.py        # Borough choropleth map (plot 27)

uv run python fix_report.py         # Assemble base report with all plots
uv run python phase3_4_inject.py    # Inject Phase 3 & 4 sections
uv run python phase5_polish.py      # Final polish: exec summary, TOC, centering

# 5. Open the report
open eda_report.html
```

> **Note:** On subsequent runs, `eda_main.py` detects `data_sample.parquet` and skips
> re-sampling. Delete the parquet file to force a fresh sample.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Runtime (managed via `uv`) |
| Pandas 2.x | Data loading, sampling, cleaning, feature engineering |
| NumPy | Numerical operations |
| Plotly + Kaleido | Interactive charts and static PNG export |
| Matplotlib / Seaborn | Fallback static plots |
| PyArrow | Parquet read/write |
| ruff | Linting and formatting |
| uv | Fast virtual environment and package management |

---

## Data Source

NYC TLC Trip Record Data (Yellow Taxi, 2018):
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>

| Attribute | Value |
|-----------|-------|
| Source file | `2018_Yellow_Taxi_Trip_Data_20260516.csv` |
| Size | 14.47 GB |
| Total rows | ~112 million |
| Sample used | 560,500 rows (~0.5%, `random.seed(42)`) |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Report generated: 2026-05-17 | 27 interactive plots | 560,500 rows analyzed*
