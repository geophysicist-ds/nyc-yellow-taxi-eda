"""
phase3_4_inject.py — Inject Phase 3 & 4 sections into eda_report.html

Description:
    Reads eda_report.html, inserts a "Fare & Tip Analysis" section after
    the Visualizations section, and a "Geographic Analysis" section after
    that. Each plot gets a 2-sentence insight caption.

Input:
    - eda_report.html
    - plots/21_*.html … plots/26_*.html

Output:
    - eda_report.html  (overwritten in-place)

Version: 1.0.0
Created: 2026-05-17
"""

import re
import sys
from pathlib import Path

REPORT_FILE = "eda_report.html"
PLOTS_DIR = Path("plots")

# ---------------------------------------------------------------------------
# PLOT METADATA: (filename_stem, caption sentence 1, caption sentence 2)
# ---------------------------------------------------------------------------
PHASE3_PLOTS = [
    (
        "21_tip_pct_distribution",
        "The distribution is strongly right-skewed with a sharp spike at exactly 20%, "
        "confirming that many passengers select the default tip suggestion.",
        "Credit card trips with tips above 25% are rare but present, "
        "suggesting occasional high generosity for short or exceptional rides.",
    ),
    (
        "22_fare_per_mile_by_borough",
        "Manhattan trips command the highest median fare-per-mile, "
        "reflecting dense stop-and-go traffic that inflates metered time charges.",
        "EWR (Newark Airport) shows the widest spread, "
        "driven by the flat-rate surcharge applied to airport runs.",
    ),
    (
        "23_surge_by_hour",
        "Extra charges peak between midnight and 6 AM (red bars), "
        "aligned with the NYC late-night $0.50 surcharge in effect 8 PM–6 AM.",
        "The morning rush (6–9 AM) also shows elevated extras, "
        "while midday hours consistently fall below the $0.30 threshold.",
    ),
]

PHASE4_PLOTS = [
    (
        "24_top20_pickup_zones",
        "Midtown Center and Upper East Side North dominate pickups, "
        "reflecting the concentration of hotels, offices, and tourist destinations.",
        "All top-20 zones are in Manhattan, underscoring the borough's "
        "overwhelming share (91%) of sampled yellow cab activity.",
    ),
    (
        "25_top20_dropoff_zones",
        "Dropoff patterns closely mirror pickup zones, with Midtown and Upper East Side "
        "again at the top, indicating high intra-Manhattan trip circulation.",
        "LaGuardia Airport appears in the top 20 dropoffs but not pickups, "
        "consistent with the airport pickup restriction for yellow cabs.",
    ),
    (
        "26_borough_sankey",
        "The dominant flow is Manhattan→Manhattan, accounting for the vast majority "
        "of trips, with thin but visible cross-borough links to Queens and Brooklyn.",
        "Outbound airport flows (Manhattan→Queens for JFK/LGA) are clearly visible "
        "as the second-largest link category by volume.",
    ),
    (
        "27_borough_trip_density",
        "Manhattan dominates with over 91% of all pickup trips, "
        "dwarfing every other borough in the sample.",
        "EWR (Newark Airport) and Staten Island show near-zero density, "
        "consistent with yellow cab licensing restrictions outside NYC proper.",
    ),
]


def make_section(section_id: str, title: str, plot_list: list[tuple]) -> str:
    """Build an HTML section div with iframe-embedded plots and captions."""
    cards = ""
    for stem, cap1, cap2 in plot_list:
        html_path = PLOTS_DIR / f"{stem}.html"
        png_path = PLOTS_DIR / f"{stem}.png"
        display_name = stem.replace("_", " ").title()

        if html_path.exists():
            embed = (
                f'<iframe src="plots/{html_path.name}" width="100%" height="650px" '
                f'frameborder="0" loading="lazy" style="border-radius:8px;"></iframe>'
            )
        elif png_path.exists():
            embed = f'<img src="plots/{png_path.name}" style="width:100%;border-radius:8px;" alt="{display_name}">'
        else:
            embed = f'<p style="color:#c62828;">Plot file not found: {stem}</p>'

        cards += f"""
  <div class="plot-card" style="margin-bottom:36px;">
    <h3 style="font-size:1.05rem;color:#333;margin-bottom:4px;">{display_name}</h3>
    <p class="plot-desc" style="font-size:0.85rem;color:#777;margin-bottom:10px;">
      {cap1} {cap2}
    </p>
    {embed}
  </div>"""

    return f"""
<div class="section" id="{section_id}">
  <h2>{title}</h2>
  {cards}
</div>"""


def strip_stale_injections(html: str) -> str:
    """Remove previously injected Phase 3/4 sections (idempotency)."""
    for section_id in ("fare-tip-analysis", "geographic-analysis"):
        html = re.sub(
            rf'<div class="section"[^>]*id="{section_id}"[^>]*>.*?</div>\s*',
            "",
            html,
            count=1,
            flags=re.DOTALL,
        )
    return html


def inject():
    html = Path(REPORT_FILE).read_text(encoding="utf-8")

    # ── Idempotency: strip any sections from a previous run ───────────────
    html = strip_stale_injections(html)

    # Build the two new sections
    fare_tip_section = make_section(
        "fare-tip-analysis",
        "Fare &amp; Tip Analysis",
        PHASE3_PLOTS,
    )
    geo_section = make_section(
        "geographic-analysis",
        "Geographic Analysis",
        PHASE4_PLOTS,
    )
    new_sections = "\n" + fare_tip_section + "\n" + geo_section + "\n"

    # ── Insert AFTER the Visualizations section, before footer / </body> ──
    # Anchor on the last container-closing </div> that precedes <footer or </body>
    if "<footer" in html:
        # Place new sections between the last section </div> and the footer
        html = re.sub(
            r"(</div>\s*<footer)",
            new_sections + r"\1",
            html,
            count=1,
            flags=re.DOTALL,
        )
        insertion_method = "after Visualizations section (before footer)"
    else:
        html = html.replace("</body>", new_sections + "</body>", 1)
        insertion_method = "before </body> (fallback)"

    Path(REPORT_FILE).write_text(html, encoding="utf-8")

    # --- Verification ---
    final = Path(REPORT_FILE).read_text(encoding="utf-8")
    sections_found = []
    for marker in [
        "Fare &amp; Tip Analysis",
        "Geographic Analysis",
        "Visualizations",
        "Deep Dive Analysis",
        "Geographic Analysis — Pickup Distribution",
    ]:
        if marker in final:
            sections_found.append(marker)

    iframe_count = final.count("<iframe ")
    img_count = final.count('<img src="plots/')
    total_embedded = iframe_count + img_count

    print(f"\nInsertion method : {insertion_method}")
    print(f"Sections found   : {sections_found}")
    print(f"Total plots embedded (iframe+img): {total_embedded}")
    print(f"Report size      : {len(final) / 1024:.1f} KB")
    print(f"\nReport written   : {REPORT_FILE}")

    if "Fare &amp; Tip Analysis" not in final or "Geographic Analysis" not in final:
        print(
            "\nWARNING: one or more sections not found in output — check insertion logic."
        )
        sys.exit(1)

    print("\nAll sections verified.")


if __name__ == "__main__":
    inject()
