#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the warehouse.

    python dashboard/build_dashboard.py --db data/warehouse.duckdb --out docs/index.html

One file, no dependencies, no server: it opens from disk and publishes to GitHub Pages as-is. That
matters more than interactivity here — a dashboard behind a cold-starting hosted app is a dashboard
nobody looks at, and a reader following a link from the README should see numbers immediately.

Charts are hand-written SVG. The palette is validated for colour-vision deficiency separation and
for contrast against both surfaces, and identity is never carried by colour alone: every series is
also directly labelled.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pae.experiment import delta_method_ratio, naive_ratio, two_proportion, welch_mean  # noqa: E402
from pae.warehouse import DEFAULT_DB, query                                             # noqa: E402

# Validated with the dataviz palette checker: lightness band, chroma floor, CVD separation of
# adjacent pairs, normal-vision floor and contrast against each surface all pass.
LIGHT = ["#3b6ea5", "#c4622d", "#2f8f6a", "#8a5bb1"]
DARK = ["#4f8bc8", "#d4733f", "#2ba173", "#a06fc8"]
CONTROL, TREATMENT = 0, 1


def _svg(content: str, width: int, height: int, label: str) -> str:
    # No height attribute: "auto" is not a valid SVG length and browsers log an error for it.
    # The viewBox plus `height: auto` in CSS is what makes the chart scale.
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
            f'aria-label="{html.escape(label)}">{content}</svg>')


def funnel_chart(frame) -> str:
    """Paired horizontal bars, one step per row. Bars start at a shared baseline so the lengths are
    comparable by eye, which is the only thing a funnel chart is for."""
    steps = list(dict.fromkeys(frame.step))
    width, row_height, pad_left, pad_top = 760, 46, 132, 18
    height = pad_top + row_height * len(steps) + 26
    top = frame.reached.max()
    parts = []
    for i, step in enumerate(steps):
        y = pad_top + i * row_height
        parts.append(f'<text x="{pad_left - 10}" y="{y + 17}" text-anchor="end" '
                     f'class="tick">{html.escape(step.replace("_", " "))}</text>')
        for j, variant in enumerate(("control", "treatment")):
            row = frame[(frame.step == step) & (frame.variant == variant)]
            if row.empty:
                continue
            value = int(row.reached.iloc[0])
            share = row.conversion_from_top.iloc[0]
            bar = max(2.0, (width - pad_left - 86) * value / top)
            bar_y = y + 2 + j * 15
            parts.append(f'<rect x="{pad_left}" y="{bar_y}" width="{bar:.1f}" height="11" rx="4" '
                         f'class="series-{j}"/>')
            if j == 1:
                parts.append(f'<text x="{pad_left + bar + 8:.1f}" y="{bar_y + 10}" '
                             f'class="value">{share:.1%}</text>')
    return _svg("".join(parts), width, height, "funnel conversion by variant")


def cumulative_chart(frame) -> str:
    """Cumulative purchase rate per variant over the run.

    Cumulative because that is how a running experiment is watched, and because the early wander is
    the whole argument against stopping at the first look that crosses.
    """
    width, height = 760, 260
    pad = {"l": 62, "r": 96, "t": 16, "b": 34}
    series = {v: g.sort_values("session_date") for v, g in frame.groupby("variant")}
    values = frame.cumulative_purchase_rate.dropna()
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    lo, hi = lo - span * 0.18, hi + span * 0.18
    dates = sorted(frame.session_date.unique())
    index = {d: i for i, d in enumerate(dates)}

    def x(date):
        return pad["l"] + (width - pad["l"] - pad["r"]) * index[date] / max(len(dates) - 1, 1)

    def y(value):
        return pad["t"] + (height - pad["t"] - pad["b"]) * (1 - (value - lo) / (hi - lo))

    parts = []
    for gridline in range(5):
        value = lo + (hi - lo) * gridline / 4
        parts.append(f'<line x1="{pad["l"]}" x2="{width - pad["r"]}" y1="{y(value):.1f}" '
                     f'y2="{y(value):.1f}" class="grid"/>')
        parts.append(f'<text x="{pad["l"] - 8}" y="{y(value) + 4:.1f}" text-anchor="end" '
                     f'class="tick">{value:.2%}</text>')
    # Direct labels at the right edge, nudged apart when the two series finish close together —
    # which is exactly what a converging experiment does, so without this the labels overlap on
    # precisely the charts worth reading.
    ends = []
    for j, variant in enumerate(("control", "treatment")):
        group = series.get(variant)
        if group is None or group.empty:
            continue
        points = " ".join(f"{x(r.session_date):.1f},{y(r.cumulative_purchase_rate):.1f}"
                          for r in group.itertuples()
                          if r.cumulative_purchase_rate == r.cumulative_purchase_rate)
        parts.append(f'<polyline points="{points}" class="line-{j}"/>')
        ends.append([j, variant, y(group.iloc[-1].cumulative_purchase_rate)])
    ends.sort(key=lambda e: e[2])
    for position in range(1, len(ends)):
        gap = ends[position][2] - ends[position - 1][2]
        if gap < 15:
            ends[position][2] = ends[position - 1][2] + 15
    for j, variant, label_y in ends:
        parts.append(f'<text x="{width - pad["r"] + 8}" y="{label_y + 4:.1f}" '
                     f'class="label-{j}">{variant}</text>')
    for position in (0, len(dates) // 2, len(dates) - 1):
        date = dates[position]
        parts.append(f'<text x="{x(date):.1f}" y="{height - 10}" text-anchor="middle" '
                     f'class="tick">{str(date)[:10]}</text>')
    return _svg("".join(parts), width, height, "cumulative purchase rate by variant over time")


def retention_grid(frame) -> str:
    """Cohort retention as a triangle. One hue, light to dark: the value is a magnitude, so a
    sequential ramp is the only correct encoding — a rainbow would imply categories."""
    cohorts = sorted(frame.cohort_week.unique())
    weeks = sorted(frame.weeks_since_signup.unique())
    cell, pad_left, pad_top = 54, 104, 30
    width = pad_left + cell * len(weeks) + 10
    height = pad_top + 22 * len(cohorts) + 12
    lookup = {(r.cohort_week, r.weeks_since_signup): r.retention for r in frame.itertuples()}
    parts = []
    for j, week in enumerate(weeks):
        parts.append(f'<text x="{pad_left + cell * j + cell / 2}" y="{pad_top - 10}" '
                     f'text-anchor="middle" class="tick">W{week}</text>')
    for i, cohort in enumerate(cohorts):
        y = pad_top + 22 * i
        parts.append(f'<text x="{pad_left - 10}" y="{y + 15}" text-anchor="end" '
                     f'class="tick">{str(cohort)[:10]}</text>')
        for j, week in enumerate(weeks):
            value = lookup.get((cohort, week))
            if value is None or value != value:
                continue
            parts.append(
                f'<rect x="{pad_left + cell * j}" y="{y}" width="{cell - 3}" height="19" rx="3" '
                f'class="heat" fill-opacity="{0.12 + 0.82 * value:.3f}"/>')
            parts.append(f'<text x="{pad_left + cell * j + (cell - 3) / 2}" y="{y + 14}" '
                         f'text-anchor="middle" class="heat-label">{value:.0%}</text>')
    return _svg("".join(parts), width, height, "weekly cohort retention")


def segment_chart(frame) -> str:
    """Revenue share by RFM segment. One series, so no legend — the title names it."""
    width, bar_height, pad_left, pad_top = 760, 38, 168, 10
    height = pad_top + bar_height * len(frame) + 16
    top = float(frame.share_of_revenue_pct.max())
    parts = []
    for i, row in enumerate(frame.itertuples()):
        y = pad_top + bar_height * i
        bar = max(2.0, (width - pad_left - 170) * row.share_of_revenue_pct / top)
        parts.append(f'<text x="{pad_left - 10}" y="{y + 20}" text-anchor="end" '
                     f'class="tick">{html.escape(row.segment)}</text>')
        parts.append(f'<rect x="{pad_left}" y="{y + 7}" width="{bar:.1f}" height="16" rx="4" '
                     f'class="series-0"/>')
        parts.append(f'<text x="{pad_left + bar + 9:.1f}" y="{y + 20}" class="value">'
                     f'{row.share_of_revenue_pct:.0f}% of revenue · {int(row.users):,} users</text>')
    return _svg("".join(parts), width, height, "share of revenue by RFM segment")


CSS_TEMPLATE = """
:root {
  --surface: #fcfcfb; --panel: #ffffff; --ink: #1b1d1f; --ink-soft: #5c6166;
  --ink-faint: #8b9196; --rule: #e4e5e3;
  --s0: __L0__; --s1: __L1__; --s2: __L2__; --s3: __L3__;
  --good: #2f8f6a; --bad: #c4622d;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { __DARK__ }
}
:root[data-theme="dark"] { __DARK__ }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased; }
.wrap { max-width: 880px; margin: 0 auto; padding: 44px 16px 72px; }
h1 { font-size: 27px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.015em; }
h2 { font-size: 17px; margin: 0 0 4px; letter-spacing: -0.005em; }
.sub { color: var(--ink-soft); margin: 0 0 34px; }
.note { color: var(--ink-soft); font-size: 13.5px; margin: 0 0 16px; }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 12px;
  padding: 20px 22px 22px; margin: 0 0 20px; }
.kpis { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  margin: 0 0 22px; }
.kpi { background: var(--panel); border: 1px solid var(--rule); border-radius: 12px;
  padding: 16px 18px; }
.kpi .k { font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-faint); margin-bottom: 7px; }
.kpi .v { font-size: 25px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.kpi .c { font-size: 12.5px; color: var(--ink-soft); font-variant-numeric: tabular-nums;
  margin-top: 4px; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 2px 0 12px;
  font-size: 13px; color: var(--ink-soft); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
svg { display: block; height: auto; max-width: 100%; }
/* A wide table scrolls inside its panel rather than widening the page: a horizontal scrollbar on
   the whole document makes every other section awkward to read on a phone. */
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { width: 100%; min-width: 520px; border-collapse: collapse; font-size: 13.5px;
  font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 600; color: var(--ink-faint); font-size: 11.5px;
  text-transform: uppercase; letter-spacing: 0.06em; padding: 0 10px 8px 0;
  border-bottom: 1px solid var(--rule); }
td { padding: 9px 10px 9px 0; border-bottom: 1px solid var(--rule); }
tr:last-child td { border-bottom: 0; }
.ok { color: var(--good); } .warn { color: var(--bad); font-weight: 600; }
.grid { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--ink-faint); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.value { fill: var(--ink-soft); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.heat { fill: var(--s0); }
.heat-label { fill: var(--ink); font-size: 11px; font-variant-numeric: tabular-nums; }
.series-0 { fill: var(--s0); } .series-1 { fill: var(--s1); }
.line-0 { fill: none; stroke: var(--s0); stroke-width: 2; stroke-linejoin: round; }
.line-1 { fill: none; stroke: var(--s1); stroke-width: 2; stroke-linejoin: round; }
.label-0 { fill: var(--s0); font-size: 12px; font-weight: 600; }
.label-1 { fill: var(--s1); font-size: 12px; font-weight: 600; }
footer { color: var(--ink-faint); font-size: 12.5px; margin-top: 30px; line-height: 1.7; }
a { color: var(--s0); }
@media (max-width: 560px) { .wrap { padding: 26px 16px 56px; } h1 { font-size: 23px; } }
"""

DARK_BLOCK = (
    "--surface: #16181a; --panel: #1d2023; --ink: #eef0f1; --ink-soft: #a8aeb3; "
    "--ink-faint: #767c81; --rule: #2d3135; "
    "--s0: __D0__; --s1: __D1__; --s2: __D2__; --s3: __D3__; "
    "--good: #2ba173; --bad: #d4733f;"
)


def stylesheet() -> str:
    dark = DARK_BLOCK
    for i, colour in enumerate(DARK):
        dark = dark.replace(f"__D{i}__", colour)
    css = CSS_TEMPLATE.replace("__DARK__", dark)
    for i, colour in enumerate(LIGHT):
        css = css.replace(f"__L{i}__", colour)
    return css


def legend(*labels: str) -> str:
    items = "".join(
        f'<span><i class="swatch" style="background:var(--s{i})"></i>{html.escape(label)}</span>'
        for i, label in enumerate(labels))
    return f'<div class="legend">{items}</div>'


def kpi(key: str, value: str, context: str = "") -> str:
    return (f'<div class="kpi"><div class="k">{html.escape(key)}</div>'
            f'<div class="v">{value}</div>'
            f'<div class="c">{html.escape(context)}</div></div>')


def build(db: str | Path) -> str:
    units = query("select * from main_marts.fct_user_metrics", db)
    funnel = query("select * from main_marts.mart_funnel order by variant, step_number", db)
    daily = query("select * from main_marts.mart_experiment_daily", db)
    retention = query("""
        select cohort_week, weeks_since_signup, retention
        from main_marts.mart_retention where weeks_since_signup <= 5
    """, db)
    segments = run_segments(db)

    control = units[units.variant == "control"]
    treatment = units[units.variant == "treatment"]
    clustered = delta_method_ratio(control.purchases, control.sessions,
                                   treatment.purchases, treatment.sessions)
    naive = naive_ratio(control.purchases, control.sessions,
                        treatment.purchases, treatment.sessions)
    conversion = two_proportion(int(control.converted.sum()), len(control),
                                int(treatment.converted.sum()), len(treatment))
    revenue = welch_mean(control.revenue, treatment.revenue)
    sessions_control = welch_mean(control.sessions, treatment.sessions)
    experiment = html.escape(str(units.experiment.iloc[0]))

    def verdict(effect) -> str:
        return ('<span class="warn">separated</span>' if effect.significant
                else '<span class="ok">not separated</span>')

    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{effect.relative:+.2%}</td>"
        f"<td>{effect.rel_low:+.2%} to {effect.rel_high:+.2%}</td>"
        f"<td>{effect.standard_error:.5f}</td><td>{verdict(effect)}</td></tr>"
        for label, effect in (
            ("purchases per session — clustered on user", clustered),
            ("purchases per session — sessions as independent", naive),
            ("users who converted", conversion),
            ("revenue per user", revenue),
            ("sessions per user (negative control)", sessions_control),
        ))

    inflation = clustered.standard_error / naive.standard_error - 1
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment readout</title>
<style>{stylesheet()}</style>
</head><body><div class="wrap">

<h1>Experiment readout — {experiment}</h1>
<p class="sub">{len(control):,} control and {len(treatment):,} treatment users ·
{int(units.sessions.sum()):,} sessions · built from the dbt warehouse in this repository.</p>

<div class="kpis">
{kpi("Purchases / session", f"{clustered.relative:+.1%}",
     f"{clustered.rel_low:+.1%} to {clustered.rel_high:+.1%}")}
{kpi("Revenue / user", f"{revenue.relative:+.1%}",
     f"{revenue.rel_low:+.1%} to {revenue.rel_high:+.1%}")}
{kpi("Users converted", f"{conversion.relative:+.1%}",
     f"{conversion.rel_low:+.1%} to {conversion.rel_high:+.1%}")}
{kpi("SE understated by", f"{inflation:+.0%}",
     "if sessions are treated as independent")}
</div>

<div class="panel">
<h2>Every metric, with the interval that belongs to it</h2>
<p class="note">The first two rows share a point estimate and differ only in the standard error.
Treating a user's sessions as independent observations understates it, which is what makes a
readout call effects significant that are not. The last row is a negative control: the treatment
has no mechanism to change how often someone visits.</p>
<div class="scroll"><table><thead><tr><th>metric</th><th>lift</th><th>95% interval</th>
<th>std. error</th><th>verdict</th></tr></thead><tbody>{rows}</tbody></table></div>
</div>

<div class="panel">
<h2>Funnel, by variant</h2>
<p class="note">Only the purchase step moves. A redesign of checkout that also shifted product
views would be a measurement problem, not a result.</p>
{legend("control", "treatment")}
{funnel_chart(funnel)}
</div>

<div class="panel">
<h2>Cumulative purchase rate over the run</h2>
<p class="note">The early part of both curves wanders, then settles. A rule that stops at the
first day the gap looks significant is reading that wander: with fourteen looks the false positive
rate is about 22%, not 5%.</p>
{legend("control", "treatment")}
{cumulative_chart(daily)}
</div>

<div class="panel">
<h2>Weekly cohort retention</h2>
<p class="note">Denominator is the whole cohort, not the subset still active — the usual way a
retention table ends up looking flat.</p>
<div class="scroll">{retention_grid(retention)}</div>
</div>

<div class="panel">
<h2>Share of revenue by RFM segment</h2>
<p class="note">Recency, frequency and monetary value as population quintiles, so the segments
keep their meaning as the business grows.</p>
{segment_chart(segments)}
</div>

<footer>
Generated by <code>dashboard/build_dashboard.py</code> from <code>data/warehouse.duckdb</code>.
The data is synthetic with a known treatment effect, which is what lets the analysis be checked
against the truth — see the validation report in the repository.
</footer>
</div></body></html>
"""


def run_segments(db: str | Path):
    from pae.warehouse import run_sql_file
    return run_sql_file(Path(__file__).resolve().parents[1] / "sql" / "rfm_segments.sql", db)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default="docs/index.html")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args.db), encoding="utf-8")
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
