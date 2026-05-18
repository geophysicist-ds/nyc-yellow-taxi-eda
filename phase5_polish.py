"""
phase5_polish.py — Final polish pass for eda_report.html

Description:
    Transforms eda_report.html into a polished, self-contained,
    presentation-ready report by injecting:
      5a. Executive summary hero banner with 5 stat cards + Top-5 findings
      5b. Sticky table of contents with anchor links
      5c. Consistent h2 section styling (colored left border, tinted bg)
      5d. Footer with generation metadata
    Then verifies the final output.

Input:
    - eda_report.html
    - analysis_results.json

Output:
    - eda_report.html  (overwritten in-place)

Version: 1.0.0
Created: 2026-05-17
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

REPORT_FILE = "eda_report.html"
RESULTS_FILE = "analysis_results.json"
TODAY = date.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# LOAD METRICS FROM analysis_results.json
# ---------------------------------------------------------------------------


def load_metrics() -> dict:
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    desc = results.get("descriptive_stats", {})
    sample_shape = results.get("sample_shape", [0, 0])
    temporal = results.get("temporal_distributions", {})

    def stat(col: str, key: str) -> float:
        return desc.get(col, {}).get(key, 0.0) or 0.0

    avg_fare = stat("fare_amount", "mean")
    avg_distance = stat("trip_distance", "mean")
    # CC-only avg tip % measured in phase3_fare_tip.py (mean of tip/fare for payment_type==1)
    avg_tip_pct = 20.88

    # Busiest hour: max count in temporal.pickup_hour
    hour_dist = temporal.get("pickup_hour", {})
    busiest_hour = 0
    if hour_dist:
        busiest_hour = int(max(hour_dist, key=lambda h: hour_dist[h]))

    hour_label = f"{busiest_hour:02d}:00"

    return {
        "total_trips": f"{sample_shape[0]:,}",
        "avg_fare": f"${avg_fare:.2f}",
        "avg_distance": f"{avg_distance:.2f} mi",
        "avg_tip_pct": f"{avg_tip_pct:.1f}%",
        "busiest_hour": hour_label,
    }


# ---------------------------------------------------------------------------
# 5A: EXECUTIVE SUMMARY HERO
# ---------------------------------------------------------------------------


def build_executive_summary(metrics: dict) -> str:
    cards = [
        ("Total Trips Sampled", metrics["total_trips"], "🚕"),
        ("Avg Fare", metrics["avg_fare"], "💵"),
        ("Avg Trip Distance", metrics["avg_distance"], "📍"),
        ("Avg Tip % (CC)", metrics["avg_tip_pct"], "💳"),
        ("Busiest Hour", metrics["busiest_hour"], "⏰"),
    ]

    card_html = ""
    for label, value, icon in cards:
        card_html += f"""
      <div class="exec-card">
        <div class="exec-icon">{icon}</div>
        <div class="exec-value">{value}</div>
        <div class="exec-label">{label}</div>
      </div>"""

    findings = [
        "Manhattan→Manhattan accounts for <strong>84%</strong> of all sampled trips — "
        "the borough is by far the dominant trip origin and destination.",
        "The taxi app's 20% tip button creates a sharp histogram spike — "
        "median credit card tip is <strong>22%</strong> of fare.",
        "Late-night surcharge (midnight–6 AM) adds <strong>&gt;$0.50 extra</strong> "
        "on average, clearly visible in the hourly surge chart.",
        "Zero-passenger trips (<strong>0.91%</strong> of sample) likely represent "
        "automated dispatch or test records rather than actual passenger journeys.",
        "EWR airport fares show the <strong>widest fare-per-mile variance</strong> "
        "of any borough, driven by the TLC flat-rate pricing to Newark.",
    ]
    findings_html = "".join(f"<li>{f}</li>" for f in findings)

    return f"""<!-- EXECUTIVE SUMMARY START -->
<section id="executive-summary" class="exec-section">
  <div class="exec-inner">
    <div class="exec-header">
      <div>
        <h1 class="exec-title">NYC Yellow Taxi — 2018 Exploratory Data Analysis</h1>
        <p class="exec-subtitle">
          Based on 560,500-row stratified sample from 112M+ trip records
          &nbsp;·&nbsp;
          <span class="quality-badge">✓ 99.03% Clean</span>
        </p>
      </div>
    </div>

    <div class="exec-cards">
      {card_html}
    </div>

    <div class="exec-findings">
      <h2 class="findings-heading">Top 5 Key Findings</h2>
      <ol class="findings-list">
        {findings_html}
      </ol>
    </div>
  </div>
</section>
<!-- EXECUTIVE SUMMARY END -->"""


# ---------------------------------------------------------------------------
# 5B: TABLE OF CONTENTS
# ---------------------------------------------------------------------------

TOC_ENTRIES = [
    ("executive-summary", "Executive Summary"),
    ("dataset-summary", "Dataset Overview"),
    ("data-quality", "Data Quality Audit"),
    ("descriptive-stats", "Descriptive Statistics"),
    ("outlier-analysis", "Outlier Analysis"),
    ("deep-dive", "Deep Dive Analysis"),
    ("fare-tip-analysis", "Fare &amp; Tip Analysis"),
    ("geographic-analysis", "Geographic Analysis"),
    ("visualizations", "All Visualizations (incl. Time Series)"),
]


def build_toc() -> str:
    items = "".join(
        f'<li><a href="#{anchor}" class="toc-link">{label}</a></li>'
        for anchor, label in TOC_ENTRIES
    )
    return f"""<!-- TOC START -->
<nav id="toc" class="toc-nav" aria-label="Table of contents">
  <div class="toc-header">Contents</div>
  <ul class="toc-list">
    {items}
  </ul>
</nav>
<!-- TOC END -->"""


# ---------------------------------------------------------------------------
# 5C: SECTION ID MAP — maps current h2 text to anchor id + injects styling
# ---------------------------------------------------------------------------

# Maps h2 text fragment → id. More specific fragments MUST appear before general ones.
# Matching is substring-based after HTML-entity normalisation.
SECTION_ID_MAP = [
    ("Dataset Summary", "dataset-summary"),
    ("Null Handling", "data-quality"),
    ("Descriptive Statistics", "descriptive-stats"),
    ("Outlier Analysis", "outlier-analysis"),
    ("Deep Dive Analysis", "deep-dive"),
    # "Pickup Distribution" is more specific than bare "Geographic Analysis"
    ("Pickup Distribution", "geo-borough"),
    ("Fare", "fare-tip-analysis"),  # "Fare & Tip Analysis"
    ("Geographic Analysis", "geographic-analysis"),  # phase4 section (no "Pickup")
    ("Visualizations", "visualizations"),
]


def add_section_ids(html: str) -> str:
    """Inject unique id= attributes on .section divs that don't already have one."""
    used_ids: set[str] = set()

    def inject_id(match: re.Match) -> str:
        opening_div = match.group(1)
        h2_text = match.group(2)

        # Already has an id — collect it and leave untouched
        existing = re.search(r'id="([^"]+)"', opening_div)
        if existing:
            used_ids.add(existing.group(1))
            return match.group(0)

        # Normalise h2 text for matching
        clean = re.sub(r"<[^>]+>", "", h2_text)
        clean = clean.replace("&amp;", "&").replace("&gt;", ">").strip()

        anchor = None
        for fragment, aid in SECTION_ID_MAP:
            frag_clean = fragment.replace("&amp;", "&")
            if frag_clean in clean and aid not in used_ids:
                anchor = aid
                break

        if anchor:
            used_ids.add(anchor)
            new_div = opening_div.replace(
                'class="section"', f'class="section" id="{anchor}"', 1
            )
        else:
            new_div = opening_div

        return new_div + h2_text

    pattern = r'(<div class="section"[^>]*>)(\s*<h2[^>]*>.*?</h2>)'
    html = re.sub(pattern, inject_id, html, flags=re.DOTALL)
    return html


# ---------------------------------------------------------------------------
# 5C CONTINUED: Inject global CSS for section styling + TOC
# ---------------------------------------------------------------------------

INJECTED_CSS = """
<style id="phase5-styles">
/* ── Executive Summary ─────────────────────────────────────── */
.exec-section {
  background: linear-gradient(160deg, #0d1b2a 0%, #1a3a5c 55%, #0f3460 100%);
  color: #fff;
  padding: 48px 60px 40px;
}
.exec-inner { max-width: 1400px; margin: 0 auto; }
.exec-title {
  font-size: 2rem; font-weight: 700; margin: 0 0 8px;
  letter-spacing: -0.5px;
}
.exec-subtitle { font-size: 1rem; opacity: 0.78; margin: 0 0 32px; }
.quality-badge {
  background: #00c853; color: #fff; border-radius: 20px;
  padding: 2px 12px; font-size: 0.82rem; font-weight: 600;
}
.exec-cards {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  margin-bottom: 36px;
}
@media (max-width: 900px) { .exec-cards { grid-template-columns: repeat(2, 1fr); } }
.exec-card {
  background: rgba(255,255,255,0.09);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 12px;
  padding: 20px 16px;
  text-align: center;
  backdrop-filter: blur(4px);
}
.exec-icon { font-size: 1.6rem; margin-bottom: 6px; }
.exec-value { font-size: 1.55rem; font-weight: 700; color: #82cfff; }
.exec-label { font-size: 0.78rem; opacity: 0.75; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.exec-findings {
  background: rgba(255,255,255,0.06);
  border-left: 4px solid #82cfff;
  border-radius: 0 8px 8px 0;
  padding: 20px 24px;
}
.findings-heading { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin: 0 0 12px; font-weight: 600; }
.findings-list { margin: 0; padding-left: 20px; }
.findings-list li { margin-bottom: 8px; font-size: 0.92rem; line-height: 1.6; opacity: 0.9; }
.findings-list strong { color: #ffd54f; }

/* ── Table of Contents ─────────────────────────────────────── */
.toc-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  background: #fff;
  border-bottom: 2px solid #e8eaf6;
  padding: 0 40px;
  display: flex;
  align-items: center;
  gap: 0;
  overflow-x: auto;
  white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.toc-header {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #1a73e8;
  margin-right: 20px;
  flex-shrink: 0;
  padding: 14px 0;
}
.toc-list { list-style: none; margin: 0; padding: 0; display: flex; gap: 0; }
.toc-link {
  display: block;
  padding: 14px 14px;
  font-size: 0.82rem;
  color: #555;
  text-decoration: none;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}
.toc-link:hover {
  color: #1a73e8;
  border-bottom-color: #1a73e8;
}

/* ── Section centering & layout ────────────────────────────── */
.section {
  font-family: 'Inter', system-ui, 'Segoe UI', sans-serif;
  max-width: 1400px;
  margin: 32px auto !important;
  padding: 32px 40px !important;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.section h2 {
  border-left: 4px solid #1a73e8 !important;
  background: #f8f9ff !important;
  padding: 10px 16px !important;
  border-radius: 0 6px 6px 0 !important;
  font-size: 1.25rem !important;
  color: #0d2137 !important;
  margin-bottom: 20px !important;
}
.section iframe {
  display: block;
  margin: 0 auto;
  max-width: 100%;
}
.section img {
  display: block;
  margin: 0 auto;
  max-width: 100%;
  border-radius: 8px;
}
.plot-card { margin-bottom: 40px; }
.plot-card h3 {
  text-align: left;
  font-size: 1.05rem;
  color: #333;
  margin-bottom: 4px;
}
.plot-desc {
  text-align: left;
  font-size: 0.85rem;
  color: #777;
  margin-bottom: 10px;
}

/* ── Footer ────────────────────────────────────────────────── */
.report-footer {
  background: #0d1b2a;
  color: rgba(255,255,255,0.55);
  text-align: center;
  padding: 28px 40px;
  font-size: 0.82rem;
  letter-spacing: 0.3px;
  border-top: 2px solid #1a3a5c;
}
.report-footer a { color: rgba(255,255,255,0.7); }
</style>"""


# ---------------------------------------------------------------------------
# 5D: FOOTER
# ---------------------------------------------------------------------------


def build_footer() -> str:
    return f"""
<footer class="report-footer">
  NYC TLC Yellow Taxi EDA &nbsp;|&nbsp;
  Data: 2018 TLC Trip Records (NYC Open Data) &nbsp;|&nbsp;
  Generated: {TODAY} &nbsp;|&nbsp;
  Plots: 27 &nbsp;|&nbsp;
  Sample: 560,500 rows &nbsp;|&nbsp;
  <span style="opacity:0.4">eda_report.html</span>
</footer>"""


# ---------------------------------------------------------------------------
# MAIN TRANSFORM
# ---------------------------------------------------------------------------


def strip_stale_phase5(html: str) -> str:
    """Remove artifacts injected by a previous run of this script so it is idempotent."""
    # Remove injected CSS block
    html = re.sub(r"<style id=\"phase5-styles\">.*?</style>", "", html, flags=re.DOTALL)
    # Remove exec summary using sentinel comments (reliable boundary)
    html = re.sub(
        r"<!-- EXECUTIVE SUMMARY START -->.*?<!-- EXECUTIVE SUMMARY END -->",
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )
    # Also strip old-style exec summary (no sentinels — from earlier runs)
    html = re.sub(
        r"<!-- EXECUTIVE SUMMARY -->.*?</section>", "", html, count=1, flags=re.DOTALL
    )
    # Remove TOC using sentinel comments
    html = re.sub(
        r"<!-- TOC START -->.*?<!-- TOC END -->",
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )
    # Also strip old-style TOC (no sentinels)
    html = re.sub(
        r"<!-- TABLE OF CONTENTS -->.*?</nav>", "", html, count=1, flags=re.DOTALL
    )
    # Remove stale geographic warning (superseded by plot 27 choropleth map)
    html = re.sub(
        r'<div[^>]*class="[^"]*alert[^"]*"[^>]*>.*?Geographic Visualizations.*?</div>',
        "",
        html,
        flags=re.DOTALL,
    )
    # Strip id= attributes from .section divs (but NOT from other elements)
    html = re.sub(
        r'(<div class="section") id="[^"]+"',
        r"\1",
        html,
    )
    # Remove old report-footer (will be rebuilt)
    html = re.sub(
        r'<footer class="report-footer">.*?</footer>', "", html, flags=re.DOTALL
    )
    return html


def polish():
    html = Path(REPORT_FILE).read_text(encoding="utf-8")

    # ── 0. Strip any leftovers from a previous run (idempotency) ──────────
    html = strip_stale_phase5(html)

    metrics = load_metrics()

    # ── 1. Inject CSS into <head> ──────────────────────────────────────────
    html = html.replace("</head>", INJECTED_CSS + "\n</head>", 1)

    # ── 2. Remove original <header> block (replaced by exec summary) ──────
    html = re.sub(
        r"<header[^>]*>.*?</header>",
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )

    # ── 2b. Remove stale geographic warning (plot 27 choropleth replaces it) ─
    html = re.sub(
        r'<div[^>]*class="[^"]*alert[^"]*"[^>]*>.*?Geographic Visualizations.*?</div>',
        "",
        html,
        flags=re.DOTALL,
    )

    # ── 3. Inject exec summary + TOC right after <body> ───────────────────
    exec_html = build_executive_summary(metrics)
    toc_html = build_toc()
    html = html.replace("<body>", "<body>\n" + exec_html + "\n" + toc_html, 1)

    # ── 4. Add section IDs (fresh, no stale IDs remaining) ─────────────────
    html = add_section_ids(html)

    # ── 5. Replace old <footer> or inject before </body> ──────────────────
    footer_html = build_footer()
    if "<footer" in html:
        html = re.sub(r"<footer[^>]*>.*?</footer>", footer_html, html, flags=re.DOTALL)
    else:
        html = html.replace("</body>", footer_html + "\n</body>", 1)

    Path(REPORT_FILE).write_text(html, encoding="utf-8")

    # ── 6. Verification ────────────────────────────────────────────────────
    final = Path(REPORT_FILE).read_text(encoding="utf-8")
    size_mb = len(final.encode("utf-8")) / (1024 * 1024)
    iframe_count = final.count("<iframe ")
    img_count = final.count("<img ")
    h2_count = len(re.findall(r"<h2", final))
    has_exec = "Executive Summary" in final
    has_toc = 'class="toc-nav"' in final
    has_footer = "report-footer" in final
    section_ids = re.findall(r'class="section" id="([^"]+)"', final)

    print(f"\n{'=' * 50}")
    print("PHASE 5 — VERIFICATION")
    print(f"{'=' * 50}")
    print(f"  File size          : {size_mb:.3f} MB")
    print(f"  <iframe> tags      : {iframe_count}")
    print(f"  <img> tags         : {img_count}")
    print(f"  <h2> tags          : {h2_count}")
    print(f"  Executive Summary  : {'✓' if has_exec else '✗ MISSING'}")
    print(f"  TOC nav            : {'✓' if has_toc else '✗ MISSING'}")
    print(f"  Footer             : {'✓' if has_footer else '✗ MISSING'}")
    print(f"  Section IDs found  : {section_ids}")
    print(f"{'=' * 50}\n")

    if not (has_exec and has_toc and has_footer):
        print("ERROR: one or more required elements missing.")
        sys.exit(1)

    print(f"Report written: {REPORT_FILE}")


if __name__ == "__main__":
    polish()
