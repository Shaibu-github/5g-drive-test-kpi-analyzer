"""Step 7 - report: one HTML template, filled in for one drive or for all drives.

    python -m drivetest report my_log.csv     one drive  -> my_log_report.html
    python -m drivetest                       all drives -> docs/ (the dashboard)

The page layout lives in templates/report.html (a Jinja2 template). This
module only prepares what the template shows: headline tiles, charts, tables
and a map. So a drive-test report always has the same structure, whatever
log it is made from - like the report an RF engineer hands in after a drive.
"""
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import charts, clean, data, handover, kpi, maps

REPO_URL = "https://github.com/Shaibu-github/5g-drive-test-kpi-analyzer"
DATASET_NOTE = (
    'Data: D. Raca, D. Leahy, C. J. Sreenan and J. J. Quinlan, "Beyond Throughput, The Next Generation: a 5G '
    'Dataset with Channel and Context Metrics", ACM MMSys 2020 '
    '(<a href="https://github.com/uccmisl/5Gdataset">uccmisl/5Gdataset</a>, GPL-3.0). '
    "Download-while-driving logs from Cork, Ireland, recorded with G-NetTrack Pro."
)
CODE_NOTE = f'Every number on this page is computed from the raw logs by <a href="{REPO_URL}">this code</a>.'
LOG_NOTE = ("The log is application-level: it shows serving-cell changes, not the handover signalling. "
            "A change can be a handover, a reselection or a re-establishment.")
NSA_NOTE = ("On 5G rows the radio values most likely describe the LTE anchor cell (5G NSA). "
            "That is an inference from the data; the log does not state it.")

TEMPLATES = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                        autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)


# ----------------------------------------------------------------------------
# Small building blocks
# ----------------------------------------------------------------------------

def _number(value, pattern, nothing="-"):
    """Format a number, or show a dash when there is no value."""
    return nothing if pd.isna(value) else pattern.format(value)


def _tile(label, value, note=""):
    return {"label": label, "value": value, "note": note}


def _table(frame, links=None, decimals=1):
    """Turn a table into text for the template. `links` gives the first column a link per row."""
    if frame is None:
        return None
    columns, texts = [], []
    for name in frame.columns:
        values = frame[name]
        is_number = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if is_number:
            whole = (values.dropna() % 1 == 0).all()             # 12 and 7 -> "12", "7";  12.5 and 7 -> "12.5", "7.0"
            pattern = "{:,.0f}" if whole else "{:,." + str(decimals) + "f}"
            texts.append([_number(value, pattern) for value in values])
        else:
            texts.append([str(value) for value in values])
        columns.append({"name": name, "number": is_number})
    rows = [{"cells": list(cells), "href": links[i] if links else None} for i, cells in enumerate(zip(*texts))]
    return {"columns": columns, "rows": rows}


def _chart(title, figure, note="", table=None, wide=False, decimals=1):
    return {"kind": "chart", "title": title, "note": note, "figure": figure,
            "table": _table(table, decimals=decimals), "wide": wide}


def _table_block(title, frame, note="", wide=False, links=None):
    return {"kind": "table", "title": title, "note": note, "table": _table(frame, links), "wide": wide}


def _map_block(title, bins, events, spots, note=""):
    web_map = maps.build_map(bins, events, spots, wheel_zoom=False)
    return {"kind": "map", "title": title, "note": note, "html": web_map.get_root().render(), "wide": True}


def render(page):
    """Fill the template with a page and return the HTML text."""
    number = 0
    for section in page["sections"]:
        for block in section["blocks"]:
            if block["kind"] == "chart":
                # the first chart of a page also loads the Plotly library; the others reuse it
                block["html"] = block["figure"].to_html(
                    full_html=False, include_plotlyjs="cdn" if number == 0 else False,
                    div_id=f"chart-{number}", config={"displayModeBar": False, "responsive": True})
                number += 1
    return TEMPLATES.get_template("report.html").render(page=page)


# ----------------------------------------------------------------------------
# Tables used on the pages
# ----------------------------------------------------------------------------

def kpi_by_technology(df):
    """Samples, signal and throughput per technology, in one small table."""
    share = kpi.tech_share(df)
    radio = kpi.radio_summary(df)
    speed = kpi.throughput_summary(df)
    table = pd.DataFrame({
        "Technology": kpi.TECHS,
        "Samples (%)": share.to_numpy(),
        "Median RSRP or RSCP (dBm)": radio["rsrp_median"].to_numpy(),       # 3G reports RSCP
        "Median SNR (dB)": radio["snr_median"].to_numpy(),
        "Median downlink (Mbps)": speed["median_mbps"].to_numpy(),
        "Best 10% above (Mbps)": speed["p90_mbps"].to_numpy(),
    })
    return table[table["Samples (%)"] > 0]


def changes_by_kind(events):
    table = handover.summarize_by_kind(events).reset_index()
    table.columns = ["Kind of change", "Changes", "In a ping-pong round trip (%)", "Median stay before (s)",
                     "Median RSRP before (dBm)", "Median RSRP gain (dB)"]
    return table


def drive_label(df):
    """A readable name for a drive: its start time."""
    return f"{df['time'].min():%d %b %Y, %H:%M}"


# ----------------------------------------------------------------------------
# Page for ONE drive
# ----------------------------------------------------------------------------

def drive_page(df, events, title, subtitle, nav=None, footer=(LOG_NOTE,)):
    """Everything the template shows for one drive."""
    active = clean.downloading(df)
    share = kpi.tech_share(df).fillna(0)
    minutes = df["elapsed_s"].max() / 60
    km = df["step_m"].sum() / 1000
    radio = df[df["tech"].isin(["5G", "LTE"])]
    round_trips = int(events["ping_pong_out"].sum())
    tiles = [
        _tile("Duration", _number(minutes, "{:.0f} min"), f"{len(df):,} samples, about one per second"),
        _tile("Distance", _number(km, "{:.1f} km"), _number(km / (minutes / 60) if minutes else None, "mean speed {:.0f} km/h")),
        _tile("Time on 5G", f"{share['5G']:.0f}%", f"LTE {share['LTE']:.0f}% · 3G {share['3G']:.0f}%"),
        _tile("Median downlink", _number(active["dl_mbps"].median(), "{:.1f} Mbps"), "while downloading"),
        _tile("Median RSRP", _number(radio["rsrp"].median(), "{:.0f} dBm"), "LTE and 5G samples"),
        _tile("Cell changes", f"{len(events)}",
              f"{_number(len(events) / km if km else None, '{:.1f}')} per km · {round_trips} ping-pong round trips"),
    ]

    # one pass over a road: smaller bins, and every bin with a sample is shown
    bins = maps.road_bins(df, events, size_m=50, min_samples=1)
    spots = maps.weak_spots(bins, min_drives=1)
    coverage_blocks = [_table_block("KPIs per technology", kpi_by_technology(df))]
    if len(radio):                                   # RSRP classes need LTE or 5G samples
        coverage, coverage_table = charts.coverage_classes(df)
        coverage_blocks.insert(0, _chart("RSRP coverage classes", coverage,
                                         table=coverage_table.rename_axis("Technology").reset_index()))
    sections = [
        {"title": "The drive, second by second",
         "note": "Line colour is the serving technology. Point at the chart to read all three values at one moment. "
                 "Dots mark serving-cell changes; a black dot starts a ping-pong round trip (A to B and back within 5 s).",
         "blocks": [_chart("RSRP, SNR and downlink throughput along the drive", charts.timeline(df, events), wide=True)]},
        {"title": "Coverage and throughput",
         "note": "RSRP classes show how much of the drive had weak, fair, good or excellent signal. "
                 "Throughput is measured while the phone was downloading.",
         "blocks": coverage_blocks},
    ]
    if len(events):
        sections.append({
            "title": "Cell changes",
            "note": "Each change interrupts the data flow for a moment. Changes inside a ping-pong round trip are "
                    "the wasteful ones: the phone ends up where it started.",
            "blocks": [_table_block("Serving-cell changes by kind", changes_by_kind(events), wide=True)]})
    if len(bins):                                    # a log without GPS positions has no map
        sections.append({
            "title": "Map",
            "note": "The route in 50 m road bins. Switch layers in the top right corner: RSRP, throughput, "
                    "technology, cell changes, ping-pong round trips. A ring marks a bin with poor RSRP.",
            "blocks": [_map_block("Route map", bins, events, spots)]})
    return {"title": title, "subtitle": subtitle, "nav": nav, "tiles": tiles, "sections": sections,
            "footer": list(footer)}


def report_for_log(log_path, out_path=None):
    """Write the report for one G-NetTrack Pro log file. Returns the path of the report."""
    log_path = Path(log_path)
    df = clean.clean(data.load_log(log_path))
    events = handover.mark_ping_pong(handover.find_cell_changes(df))
    out_path = Path(out_path) if out_path else log_path.with_name(log_path.stem + "_report.html")
    page = drive_page(df, events, title=f"Drive-test report: {log_path.name}",
                      subtitle=f"One drive, started {drive_label(df)}. Made from the raw G-NetTrack Pro log.")
    out_path.write_text(render(page), encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------------
# Page for ALL drives, and the whole site
# ----------------------------------------------------------------------------

def overview_page(df, events, summary, bins, spots, rules, main, borders, scores, baselines, drives_table, nav):
    """Everything the template shows for all drives together (the dashboard front page)."""
    facts, changes, tuned = summary["overview"], summary["cell_changes"], summary["tuning"]
    share = summary["tech_share_pct"]
    speed = kpi.throughput_summary(df)["median_mbps"]
    border_rule, margin_rule = tuned["border_rule"], tuned["margin_rule_3_db"]
    tiles = [
        _tile("Drives", f"{facts['drives']}", f"{facts['hours']} hours · {facts['route_km']:.0f} km · {facts['cells']} cells"),
        _tile("Time on 5G", f"{share['5G']:.0f}%", f"LTE {share['LTE']:.0f}% · 3G {share['3G']:.0f}%"),
        _tile("Median downlink on 5G", f"{speed['5G']:.1f} Mbps", f"LTE {speed['LTE']:.1f} · 3G {speed['3G']:.1f} Mbps"),
        _tile("Cell changes", f"{changes['per_km']} per km", f"{changes['total']} changes in total"),
        _tile("Changes in a ping-pong", f"{changes['changes_in_a_round_trip_pct']:.0f}%",
              f"{changes['ping_pong_round_trips']} round trips (back within 5 s)"),
        _tile("Dip at a cell change", f"{changes['median_drop_pct']:.0f}%",
              f"median throughput drop, {changes['isolated_used_for_dip']} isolated changes"),
    ]

    coverage, coverage_table = charts.coverage_classes(df)
    rsrp, rsrp_table = charts.distribution(df, "rsrp", "RSRP", "dBm")
    snr, snr_table = charts.distribution(df, "snr", "SNR", "dB")
    mbps, mbps_table = charts.distribution(df, "dl_mbps", "Downlink throughput", "Mbps, log scale", techs=kpi.TECHS, log_x=True)
    profile, profile_table, isolated_changes = charts.change_profile(df, events)
    top_borders = borders.head(10).reset_index()
    top_borders["in_drives"] = [f"{with_round_trip} of {crossed}" for with_round_trip, crossed
                                in zip(top_borders["drives_with_round_trip"], top_borders["drives"])]
    top_borders = top_borders[["border", "kind", "changes", "round_trips", "in_drives"]]
    top_borders.columns = ["Border (site-cell | site-cell)", "Kind", "Changes", "Round trips",
                           "Drives with a round trip"]
    rules_table = rules[["rule", "setting", "round_trips_caught_pct", "normal_changes_affected_pct",
                         "advantage_points"]].round(1)
    rules_table.columns = ["Rule", "Setting", "Round trips it acts on (%)", "Normal changes it acts on (%)",
                           "Advantage (points)"]
    main_table = main.round(1).reset_index()
    main_table.columns = ["Round trips used", "Rule or difference", "Advantage (points)", "95% low", "95% high"]

    def percentiles(table):
        return table.rename_axis("Technology").reset_index()

    sections = [
        {"title": "Coverage and signal quality",
         "note": "How strong and how clean the signal was. On 5G rows the phone reports the LTE anchor cell (5G NSA), "
                 "so these values compare the anchor layer with plain LTE.",
         "blocks": [
             _chart("RSRP coverage classes", coverage, wide=True,
                    table=coverage_table.rename_axis("Technology").reset_index()),
             _chart("RSRP distribution", rsrp, "A curve further to the right means stronger signal.", percentiles(rsrp_table)),
             _chart("SNR distribution", snr, "A curve further to the right means a cleaner signal.", percentiles(snr_table)),
         ]},
        {"title": "Throughput",
         "note": "Downlink throughput while the phone was downloading. The axis is logarithmic: each step to the "
                 "right is ten times faster.",
         "blocks": [
             _chart("Downlink throughput distribution", mbps, table=percentiles(mbps_table)),
             _table_block("KPIs per technology", kpi_by_technology(df)),
         ]},
        {"title": "Cell changes",
         "note": "The log shows when the serving cell changes. That can be a handover, a reselection or a "
                 "re-establishment; the log cannot tell them apart.",
         "blocks": [
             _chart("Throughput around a cell change", profile,
                    f"Median over {isolated_changes} isolated changes (no other change within 10 s).", profile_table.reset_index()),
             _table_block("Serving-cell changes by kind", changes_by_kind(events)),
         ]},
        {"title": "Ping-pong: does it belong to particular borders?",
         "note": f"A rule is good when it acts on many ping-pong round trips and on few normal changes. Knowing that a "
                 f"border had a round trip in another drive acts on {border_rule['round_trips_caught_pct']:.0f}% of the "
                 f"round trips and on {border_rule['normal_changes_affected_pct']:.0f}% of the normal changes. A 3 dB "
                 f"signal margin acts on {margin_rule['round_trips_caught_pct']:.0f}% and "
                 f"{margin_rule['normal_changes_affected_pct']:.0f}%. So in this log the border says more than the "
                 f"signal gain. But {tuned['round_trips_of_2_s_or_less']} of the {tuned['round_trips']} round trips last "
                 "2 seconds or less and may be effects of the logger. Without them no difference can be shown. "
                 "The border list shows where to look first; it is not a tuning recommendation.",
         "blocks": [
             _chart("Three rules compared", charts.ping_pong_rules(rules),
                    "Up and to the left is better. The border rule and the road-only rule are judged on drives "
                    "they did not learn from.", rules_table),
             _table_block("The 10 borders with the most round trips", top_borders),
             _table_block("How sure is it?", main_table, wide=True,
                          note="Advantage = round trips acted on minus normal changes acted on, in percentage "
                               "points, with a 95% interval from a bootstrap over drives. An interval that "
                               "includes 0 means: not shown by this data."),
         ]},
        {"title": "Map",
         "note": "All drives in 100 m road bins. Switch layers in the top right corner. A ring marks a weak spot "
                 "that was seen in at least two drives.",
         "blocks": [_map_block("Routes in Cork", bins, events, spots)]},
        {"title": "Throughput prediction",
         "note": "Three models predict the throughput of the next 10 seconds. They are tested on drives they never "
                 "saw. Radio KPIs alone explain little; location and recent history explain much more.",
         "blocks": [_chart("Model scores by input set", charts.model_scores(scores, baselines), wide=True, decimals=2,
                           table=scores.round(2).rename(columns={"features": "Inputs", "model": "Model", "r2_log": "R²",
                                                                 "mae_mbps": "Mean error (Mbps)",
                                                                 "worst_fold_r2": "Worst fold R²",
                                                                 "best_fold_r2": "Best fold R²"}))]},
        {"title": "The drives",
         "note": "Each drive has its own report with the same layout.",
         "blocks": [_table_block("All drives", drives_table.drop(columns="href"), wide=True,
                                 links=drives_table["href"].tolist())]},
    ]
    return {"title": f"Drive-test dashboard: {facts['drives']} drives in Cork",
            "subtitle": f"{facts['samples']:,} samples from a phone downloading while driving, "
                        f"{facts['first_day']} to {facts['last_day']}. One operator, 5G NSA with LTE and 3G fallback.",
            "nav": nav, "tiles": tiles, "sections": sections, "footer": [DATASET_NOTE, CODE_NOTE, LOG_NOTE, NSA_NOTE]}


def write_site(site_dir, df, events, summary, bins, spots, rules, main, borders, scores, baselines):
    """Write the dashboard: index.html for all drives and one page per drive."""
    site_dir = Path(site_dir)
    (site_dir / "drives").mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").write_text("")          # tells GitHub Pages to serve the files as they are

    drives = list(df["drive"].unique())
    labels = {drive: drive_label(df[df["drive"] == drive]) for drive in drives}

    def nav(current, prefix_home, prefix_drive):
        options = [{"label": "All drives", "href": f"{prefix_home}index.html", "selected": current is None}]
        options += [{"label": labels[d], "href": f"{prefix_drive}{d}.html", "selected": d == current} for d in drives]
        links = [{"label": "Code on GitHub", "href": REPO_URL}]
        return {"options": options, "links": links}

    rows = []
    for drive in drives:
        one, its_events = df[df["drive"] == drive], events[events["drive"] == drive]
        page = drive_page(one, its_events, title=f"Drive of {labels[drive]}",
                          subtitle=f"One of the {len(drives)} download-while-driving logs from Cork, Ireland.",
                          nav=nav(drive, "../", ""), footer=[DATASET_NOTE, CODE_NOTE, LOG_NOTE, NSA_NOTE])
        (site_dir / "drives" / f"{drive}.html").write_text(render(page), encoding="utf-8")
        km = one["step_m"].sum() / 1000
        rows.append({
            "Drive": labels[drive],
            "Minutes": round(one["elapsed_s"].max() / 60),
            "Distance (km)": km,
            "Time on 5G (%)": (one["tech"] == "5G").mean() * 100,
            "Median downlink (Mbps)": clean.downloading(one)["dl_mbps"].median(),
            "Cell changes": len(its_events),
            "Ping-pong round trips": int(its_events["ping_pong_out"].sum()),
            "href": f"drives/{drive}.html",
        })
    drives_table = pd.DataFrame(rows).round(1)

    front = overview_page(df, events, summary, bins, spots, rules, main, borders, scores, baselines, drives_table,
                          nav(None, "", "drives/"))
    (site_dir / "index.html").write_text(render(front), encoding="utf-8")
    return site_dir / "index.html"
