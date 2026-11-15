import pandas as pd

from drivetest import handover
from drivetest.clean import clean


def lte_cell(site_hex, sector):
    """A consistent LTE cell: identity = eNB ID * 256 + cell number."""
    return {"NODEHEX": site_hex, "CellID": sector, "RAWCELLID": int(site_hex, 16) * 256 + sector}


# three cells: A and B are two sectors of site A1, C belongs to site B2
A = lte_cell("A1", 1)
B = lte_cell("A1", 2)
C = lte_cell("B2", 1)                                         # same sector number as A, other site


def events_for(make_raw, rows):
    df = clean(make_raw(rows))
    return df, handover.mark_ping_pong(handover.find_cell_changes(df))


# --- finding the changes ------------------------------------------------------

def test_no_change_gives_no_event(make_raw):
    _, events = events_for(make_raw, [A, A, A])
    assert len(events) == 0


def test_kinds_of_change(make_raw):
    lte_to_5g = dict(C, NetworkMode="5G")
    _, events = events_for(make_raw, [A, A, B, B, C, C, lte_to_5g])
    assert events["kind"].tolist() == ["LTE intra-site", "LTE inter-site", "LTE to 5G"]
    assert events["sample"].tolist() == [2, 4, 6]            # the first sample on the new cell


def test_3g_changes_are_not_split_into_intra_and_inter_site(make_raw):
    first = {"NetworkMode": "HSPA+", "NODEHEX": "C", "CellID": 500, "RAWCELLID": 12 * 65536 + 500}
    second = {"NetworkMode": "HSPA+", "NODEHEX": "C", "CellID": 501, "RAWCELLID": 12 * 65536 + 501}
    _, events = events_for(make_raw, [first, first, second])
    assert events["kind"].tolist() == ["3G cell change"]      # on 3G the "site" is an RNC


def test_same_sector_number_on_another_site_is_a_change(make_raw):
    # A and C both have CellID 1. Watching CellID alone would miss this handover.
    _, events = events_for(make_raw, [A, A, C, C])
    assert len(events) == 1
    assert events.iloc[0]["kind"] == "LTE inter-site"


def test_a_new_drive_is_not_a_cell_change(make_raw):
    _, events = events_for(make_raw, [dict(A, drive="d1"), dict(A, drive="d1"), dict(C, drive="d2"), dict(C, drive="d2")])
    assert len(events) == 0


# --- ping-pong ------------------------------------------------------------------

def test_quick_return_is_ping_pong(make_raw):
    _, events = events_for(make_raw, [A] * 10 + [B] * 2 + [A] * 10)       # 2 s on B, then back
    assert events["ping_pong"].tolist() == [True, True]
    assert events["ping_pong_return"].tolist() == [False, True]           # one round trip


def test_slow_return_is_not_ping_pong(make_raw):
    _, events = events_for(make_raw, [A] * 10 + [B] * 20 + [A] * 10)      # 20 s on B is a real stay
    assert events["ping_pong"].tolist() == [False, False]


def test_moving_on_to_a_third_cell_is_not_ping_pong(make_raw):
    _, events = events_for(make_raw, [A] * 10 + [B] * 2 + [C] * 10)       # A -> B -> C: no return
    assert events["ping_pong"].tolist() == [False, False]


def test_the_time_limit_is_inclusive(make_raw):
    df = clean(make_raw([A] * 10 + [B] * 5 + [A] * 10))                   # exactly 5 s on B
    events = handover.find_cell_changes(df)
    assert handover.mark_ping_pong(events, max_stay_s=5)["ping_pong"].all()
    assert not handover.mark_ping_pong(events, max_stay_s=4)["ping_pong"].any()


def test_ping_pong_does_not_reach_across_two_drives(make_raw):
    # drive d1 ends just after A -> B; drive d2 starts with a quick B -> A.
    # That is not a return, because they are different drives.
    first = [dict(A, drive="d1")] * 10 + [dict(B, drive="d1")] * 2
    second = [dict(B, drive="d2")] * 2 + [dict(A, drive="d2")] * 10
    _, events = events_for(make_raw, first + second)
    assert len(events) == 2
    assert events["ping_pong"].tolist() == [False, False]


def test_sensitivity_table_counts_round_trips(make_raw):
    _, events = events_for(make_raw, [A] * 10 + [B] * 4 + [A] * 10)       # one round trip, 4 s on B
    table = handover.ping_pong_sensitivity(events, thresholds=(2, 5))
    assert table["round_trips"].tolist() == [0, 1]
    assert table["pct_of_changes"].tolist() == [0, 100]


# --- stay time, isolation, windows ------------------------------------------------

def test_stay_time_is_the_time_on_the_previous_cell(make_raw):
    _, events = events_for(make_raw, [A] * 10 + [B] * 20 + [A] * 10)
    assert events["stay_s"].tolist() == [10, 20]


def test_stay_time_restarts_in_a_new_drive(make_raw):
    first = [dict(A, drive="d1")] * 30 + [dict(B, drive="d1")] * 5
    second = [dict(A, drive="d2")] * 7 + [dict(B, drive="d2")] * 5
    _, events = events_for(make_raw, first + second)
    assert events["stay_s"].tolist() == [30, 7]               # 7 s since the start of d2, not since d1


def test_isolated_drops_changes_that_are_close_together(make_raw):
    _, events = events_for(make_raw, [A] * 30 + [B] * 3 + [C] * 30)       # two changes 3 samples apart
    assert len(handover.isolated(events, window=10)) == 0
    _, events = events_for(make_raw, [A] * 30 + [B] * 30)
    assert len(handover.isolated(events, window=10)) == 1


def test_isolation_is_judged_inside_each_drive(make_raw):
    # the change in d1 is the last thing in d1; the change in d2 comes 3 samples into d2.
    # They are in different drives, so both are isolated.
    first = [dict(A, drive="d1")] * 30 + [dict(B, drive="d1")] * 2
    second = [dict(A, drive="d2")] * 3 + [dict(B, drive="d2")] * 30
    _, events = events_for(make_raw, first + second)
    assert len(handover.isolated(events, window=10)) == 2


def test_window_is_centred_on_the_first_sample_of_the_new_cell(make_raw):
    old = dict(A, DL_bitrate=20000)
    new = dict(B, DL_bitrate=5000)
    df, events = events_for(make_raw, [old] * 15 + [new] * 15)
    window = handover.throughput_window(df, events, window=10).set_index("offset")["dl_mbps"]
    assert window.loc[-1] == 20 and window.loc[0] == 5
    dips = handover.dip_per_event(handover.throughput_window(df, events, window=10))
    assert dips.iloc[0]["before_mbps"] == 20 and dips.iloc[0]["after_mbps"] == 5


def test_idle_samples_are_left_out_of_the_throughput_window(make_raw):
    idle = dict(B, State="I", DL_bitrate=0)
    df, events = events_for(make_raw, [A] * 15 + [idle] + [B] * 14)
    window = handover.throughput_window(df, events, window=10)
    assert 0 not in window["offset"].tolist()                 # offset 0 was idle: not a throughput sample


# --- dip statistics ---------------------------------------------------------------

def test_dip_summary_on_known_numbers():
    halved = pd.DataFrame({"before_mbps": [10.0] * 25, "around_mbps": [5.0] * 25, "after_mbps": [20.0] * 25, "kind": "X"})
    no_traffic = pd.DataFrame({"before_mbps": [0.5], "around_mbps": [5.0], "after_mbps": [5.0], "kind": "X"})
    few = pd.DataFrame({"before_mbps": [10.0] * 3, "around_mbps": [10.0] * 3, "after_mbps": [10.0] * 3, "kind": "Rare"})
    summary = handover.dip_summary(pd.concat([halved, no_traffic, few], ignore_index=True))
    assert summary.loc["X", "events"] == 25                   # the event with almost no traffic before is left out
    assert summary.loc["X", "median_drop_pct"] == 50          # 10 -> 5 Mbps is a 50% drop
    assert summary.loc["X", "median_after_vs_before_pct"] == 100
    assert summary.loc["X", "drop_low"] == 50 and summary.loc["X", "drop_high"] == 50
    assert "Rare" not in summary.index                        # too few events for a row of its own
    assert summary.loc["All changes", "events"] == 28


def test_median_interval_contains_the_median_and_narrows_with_more_data():
    few = handover.median_interval([0, 10, 20, 30, 40])
    many = handover.median_interval(list(range(41)) * 20)
    assert few[0] <= 20 <= few[1]
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_rsrp_gain_is_blank_between_3g_and_lte(make_raw):
    cell_3g = {"NetworkMode": "HSPA+", "NODEHEX": "C", "CellID": 500, "RAWCELLID": 12 * 65536 + 500, "RSRP": -70}
    _, events = events_for(make_raw, [dict(A, RSRP=-110)] * 3 + [cell_3g] * 3)
    table = handover.summarize_by_kind(events)
    assert pd.isna(table.loc["LTE to 3G", "median_rsrp_gain_db"])         # RSRP and RSCP cannot be subtracted


def test_rate_per_drive(make_raw):
    df, events = events_for(make_raw, [A] * 60 + [B] * 61)                # 120 s = 2 minutes, one change
    rate = handover.rate_per_drive(df, events)
    assert rate.loc["drive1", "changes"] == 1
    assert rate.loc["drive1", "per_minute"] == 0.5
