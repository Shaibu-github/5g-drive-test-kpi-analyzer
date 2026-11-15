import numpy as np
import pandas as pd
import pytest

from drivetest import charts, handover, report
from drivetest.clean import clean


def lte_cell(site_hex, sector, **more):
    return {"NODEHEX": site_hex, "CellID": sector, "RAWCELLID": int(site_hex, 16) * 256 + sector, **more}


A = lte_cell("A1", 1)
B = lte_cell("A1", 2, NetworkMode="5G", RSRP=-80)
# 10 s on LTE cell A, 2 s on 5G cell B, back to A: one ping-pong round trip, two technologies
ROWS = [A] * 10 + [B] * 2 + [A] * 10


def prepared(make_raw, rows=ROWS):
    df = clean(make_raw(rows))
    return df, handover.mark_ping_pong(handover.find_cell_changes(df))


# --- tables --------------------------------------------------------------------------

def test_table_formats_numbers_per_column():
    frame = pd.DataFrame({"Name": ["a", "b"], "Whole": [12.0, 7.0], "Mixed": [12.5, 7.0], "Gap": [np.nan, 3.25]})
    table = report._table(frame)
    assert [row["cells"] for row in table["rows"]] == [["a", "12", "12.5", "-"], ["b", "7", "7.0", "3.2"]]
    assert [column["number"] for column in table["columns"]] == [False, True, True, True]   # numbers are right-aligned


def test_table_links_belong_to_their_rows():
    table = report._table(pd.DataFrame({"Drive": ["one", "two"]}), links=["one.html", "two.html"])
    assert [(row["cells"][0], row["href"]) for row in table["rows"]] == [("one", "one.html"), ("two", "two.html")]


def test_kpi_table_leaves_out_technologies_that_were_never_used(make_raw):
    df, _ = prepared(make_raw)
    assert report.kpi_by_technology(df)["Technology"].tolist() == ["5G", "LTE"]     # no 3G in this log


# --- charts --------------------------------------------------------------------------

def test_timeline_shows_a_technology_only_while_it_serves(make_raw):
    df, events = prepared(make_raw)
    fig = charts.timeline(df, events)
    rsrp_lines = {trace.name: np.asarray(trace.y, dtype=float) for trace in fig.data
                  if trace.mode == "lines" and trace.yaxis == "y"}
    on_5g = (df["tech"] == "5G").to_numpy()
    assert np.isnan(rsrp_lines["5G"][~on_5g]).all() and (rsrp_lines["5G"][on_5g] == -80).all()
    assert np.isnan(rsrp_lines["LTE"][on_5g]).all() and (rsrp_lines["LTE"][~on_5g] == -90).all()


def test_timeline_marks_cell_changes_and_ping_pong(make_raw):
    df, events = prepared(make_raw)
    marks = {trace.name: len(trace.x) for trace in charts.timeline(df, events).data if trace.mode == "markers"}
    assert marks == {"Cell change": 1, "Start of a ping-pong round trip": 1}     # way back, way out


def test_distribution_reads_the_share_at_or_below_a_value(make_raw):
    rows = [dict(A, RSRP=-100)] * 3 + [dict(A, RSRP=-80)]                # three weak samples, one strong
    df, _ = prepared(make_raw, rows)
    fig, table = charts.distribution(df, "rsrp", "RSRP", "dBm", techs=("LTE",))
    curve = fig.data[0]
    assert curve.y[0] == 75 and curve.y[-1] == 100       # at -100 dBm: 3 of 4 samples; at -80 dBm: all
    assert table.loc["LTE", "P50"] == -100


def test_coverage_chart_has_one_bar_part_per_rsrp_class(make_raw):
    df, _ = prepared(make_raw)
    fig, table = charts.coverage_classes(df)
    assert len(fig.data) == 4
    assert table.loc["5G"].tolist() == [0, 0, 0, 100]    # all 5G samples at -80 dBm: "Excellent"
    assert table.loc["LTE"].tolist() == [0, 0, 100, 0]   # all LTE samples at -90 dBm: "Good"


# --- pages ---------------------------------------------------------------------------

def test_drive_page_renders_to_html(make_raw):
    df, events = prepared(make_raw)
    html = report.render(report.drive_page(df, events, title="My <test> drive", subtitle="A subtitle"))
    assert "My &lt;test&gt; drive" in html                 # text from outside is escaped, never run as HTML
    assert html.count("Plotly.newPlot") == 2               # the timeline and the coverage chart
    assert html.count("plotly-") >= 1 and "<iframe" in html
    assert "2 ping-pong round trips" not in html and "1 ping-pong round trips" in html


def test_a_drive_without_cell_changes_still_gets_a_report(make_raw):
    df, events = prepared(make_raw, [A] * 15)
    page = report.drive_page(df, events, title="Quiet drive", subtitle="")
    assert "Cell changes" not in [section["title"] for section in page["sections"]]
    assert "Quiet drive" in report.render(page)


def test_a_log_without_gps_gets_a_report_without_a_map(make_raw):
    rows = [dict(row, Latitude=None, Longitude=None) for row in ROWS]
    df, events = prepared(make_raw, rows)
    page = report.drive_page(df, events, title="No GPS", subtitle="")
    assert "Map" not in [section["title"] for section in page["sections"]]
    assert "No GPS" in report.render(page)


def test_a_3g_only_log_gets_a_report_without_rsrp_classes(make_raw):
    umts = {"NetworkMode": "HSPA+", "NODEHEX": "C", "CellID": 500, "RAWCELLID": 12 * 65536 + 500}
    df, events = prepared(make_raw, [umts] * 15)
    page = report.drive_page(df, events, title="3G drive", subtitle="")
    titles = [block["title"] for section in page["sections"] for block in section["blocks"]]
    assert "RSRP coverage classes" not in titles and "KPIs per technology" in titles
    assert "on 3G RSCP" in report.render(page)                            # the level axis says what it shows


def test_table_can_show_two_decimals():
    table = report._table(pd.DataFrame({"R2": [0.19, 0.631]}), decimals=2)
    assert [row["cells"][0] for row in table["rows"]] == ["0.19", "0.63"]


def test_report_for_log_reads_a_file_and_writes_a_report(make_raw, tmp_path):
    log = tmp_path / "my_drive.csv"
    make_raw(ROWS).drop(columns="drive").to_csv(log, index=False, na_rep="-")     # the way G-NetTrack writes it
    out = report.report_for_log(log)
    assert out == tmp_path / "my_drive_report.html"
    assert "Drive-test report: my_drive.csv" in out.read_text(encoding="utf-8")


def test_any_log_file_name_is_accepted(make_raw, tmp_path):
    log = tmp_path / "drive.csv"                                                  # no "_" in the name
    make_raw(ROWS).drop(columns="drive").to_csv(log, index=False, na_rep="-")
    assert report.report_for_log(log, tmp_path / "out.html").exists()


def test_a_log_with_missing_columns_gives_a_clear_message(make_raw):
    with pytest.raises(ValueError, match="RSRP"):
        clean(make_raw(ROWS).drop(columns="RSRP"))
