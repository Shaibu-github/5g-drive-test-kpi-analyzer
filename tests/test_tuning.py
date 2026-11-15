import pandas as pd

from drivetest import handover, tuning
from drivetest.clean import clean


def lte_cell(site_hex, sector, rsrp=-90):
    """A consistent LTE cell: identity = eNB ID * 256 + cell number."""
    return {"NODEHEX": site_hex, "CellID": sector, "RAWCELLID": int(site_hex, 16) * 256 + sector, "RSRP": rsrp}


A = lte_cell("A1", 1)
B = lte_cell("A1", 2)
C = lte_cell("B2", 1)
PING_PONG_AB = [A] * 10 + [B] * 2 + [A] * 10            # one round trip on border A|B, 2 s on B
PING_PONG_AC = [A] * 10 + [C] * 2 + [A] * 10            # one round trip on border A|C
PLAIN_AB = [A] * 10 + [B] * 20                          # one normal change on border A|B


def changes_for(make_raw, rows):
    events = handover.mark_ping_pong(handover.find_cell_changes(clean(make_raw(rows))))
    return tuning.radio_changes(events)


def in_drive(drive, rows):
    return [dict(row, drive=drive) for row in rows]


# --- names, borders and gain --------------------------------------------------------

def test_cell_name_shows_site_and_cell_number():
    assert tuning.cell_name(0xA1 * 256 + 2, "LTE") == "A1-2"
    assert tuning.cell_name(0xA1 * 256 + 2, "5G") == "A1-2"
    assert tuning.cell_name(12 * 65536 + 500, "3G") == "C-500"         # 3G: RNC ID * 65536 + cell


def test_both_directions_are_the_same_border(make_raw):
    changes = changes_for(make_raw, [A] * 10 + [B] * 20 + [A] * 10)      # A -> B, later B -> A
    assert changes["border"].tolist() == ["A1-1 | A1-2", "A1-1 | A1-2"]


def test_gain_is_new_cell_minus_old_cell(make_raw):
    weak, strong = lte_cell("A1", 1, rsrp=-100), lte_cell("B2", 1, rsrp=-94)
    changes = changes_for(make_raw, [weak] * 5 + [strong] * 5 + [weak] * 5)
    assert changes["gain_db"].tolist() == [6, -6]


def test_changes_from_or_to_3g_are_left_out(make_raw):
    umts = {"NetworkMode": "HSPA+", "NODEHEX": "C", "CellID": 500, "RAWCELLID": 12 * 65536 + 500}
    changes = changes_for(make_raw, [A] * 5 + [umts] * 5 + [B] * 5 + [A] * 5)
    assert changes["border"].tolist() == ["A1-1 | A1-2"]                 # only the LTE to LTE change is kept


# --- the three rules -----------------------------------------------------------------

def test_margin_rule_acts_on_small_gains_only(make_raw):
    old, a_bit_better, much_better = lte_cell("A1", 1, -100), lte_cell("A1", 2, -99), lte_cell("B2", 1, -90)
    changes = changes_for(make_raw, [old] * 5 + [a_bit_better] * 20 + [much_better] * 20)
    assert changes["gain_db"].tolist() == [1, 9]
    assert tuning.margin_acts(changes, 3).tolist() == [True, False]
    assert tuning.margin_acts(changes, 1).tolist() == [False, False]     # "below 1 dB": a gain of 1 is enough


def test_seen_elsewhere_never_counts_the_own_drive(make_raw):
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB) + in_drive("d2", PING_PONG_AB)
                          + in_drive("d3", PING_PONG_AC))
    round_trips_elsewhere = tuning.seen_elsewhere(changes, changes["ping_pong_out"])
    by_drive = round_trips_elsewhere.groupby(changes["drive"]).max()
    assert by_drive.to_dict() == {"d1": 1, "d2": 1, "d3": 0}     # A|C ping-pongs only in d3 itself


def test_border_rule_only_uses_what_other_drives_show(make_raw):
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB) + in_drive("d2", PING_PONG_AB)
                          + in_drive("d3", PING_PONG_AC))
    acts = tuning.border_acts(changes)
    assert acts[changes["drive"] != "d3"].all()       # A|B is known from the other drive
    assert not acts[changes["drive"] == "d3"].any()   # A|C is only known from d3 itself: must not count


def test_border_rule_needs_enough_round_trips(make_raw):
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB) + in_drive("d2", PING_PONG_AB))
    assert tuning.border_acts(changes, min_round_trips=1).all()
    assert not tuning.border_acts(changes, min_round_trips=2).any()      # each other drive has only one


def test_road_rule_knows_the_road_but_not_the_ping_pong(make_raw):
    # d1 and d2 cross A|B without any ping-pong; d3 crosses A|C, which nobody else does
    changes = changes_for(make_raw, in_drive("d1", PLAIN_AB) + in_drive("d2", PLAIN_AB) + in_drive("d3", PING_PONG_AC))
    road = tuning.road_acts(changes)
    assert road[changes["drive"] != "d3"].all() and not road[changes["drive"] == "d3"].any()
    assert not tuning.border_acts(changes).any()      # no border has a round trip in another drive


# --- scoring ---------------------------------------------------------------------------

def score_example(acts):
    """Two drives. Each has one round trip (way out + way back) and two normal changes."""
    changes = pd.DataFrame({
        "drive": ["d1"] * 4 + ["d2"] * 4,
        "ping_pong_out": [True, False, False, False] * 2,
        "ping_pong": [True, True, False, False] * 2,
    })
    return tuning.score_rule(changes, pd.Series(acts))


def test_score_counts_round_trips_and_normal_changes_separately():
    #                     d1: out   back   normal normal  d2: out    back   normal normal
    score = score_example([True, False, True, False, False, False, False, False])
    assert (score["round_trips_caught"], score["round_trips"]) == (1, 2)
    assert (score["normal_changes_affected"], score["normal_changes"]) == (1, 4)
    assert score["round_trips_caught_pct"] == 50 and score["normal_changes_affected_pct"] == 25
    assert score["advantage_points"] == 25


def test_acting_on_the_way_back_is_not_counted():
    score = score_example([False, True, False, False, False, True, False, False])
    assert score["round_trips_caught"] == 0          # the way back is the result, not the decision
    assert score["normal_changes_affected"] == 0


def test_a_rule_that_acts_on_everything_has_no_advantage():
    assert score_example([True] * 8)["advantage_points"] == 0


def test_compare_rules_has_one_row_per_setting(make_raw):
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB + [B] * 20) + in_drive("d2", PING_PONG_AB + [B] * 20))
    table = tuning.compare_rules(changes, margins=(0, 3), min_round_trips=(1,))
    assert table["rule"].tolist() == ["Margin", "Margin", "Border", "Road only"]
    border = table[table["rule"] == "Border"].iloc[0]
    assert border["round_trips_caught_pct"] == 100 and border["normal_changes_affected_pct"] == 100


# --- the checks on the result ------------------------------------------------------------

def test_short_round_trips_can_be_left_out(make_raw):
    long_trip = [A] * 10 + [C] * 4 + [A] * 10                             # 4 s on C
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB) + in_drive("d2", long_trip))
    kept = tuning.without_short_round_trips(changes, max_stay_s=2)
    assert kept["drive"].tolist() == ["d2", "d2"]                         # both legs of the 2 s trip are gone
    assert len(tuning.without_short_round_trips(changes, max_stay_s=1)) == 4


def test_a_drive_drawn_twice_does_not_teach_itself(make_raw):
    once = changes_for(make_raw, in_drive("d1", PING_PONG_AB))
    twice = pd.concat([once, once], ignore_index=True)                    # what a bootstrap draw can look like
    assert not tuning.border_acts(twice).any()


def test_intervals_come_from_repeating_the_whole_procedure(make_raw):
    rows = (in_drive("d1", PING_PONG_AB + [B] * 20) + in_drive("d2", PING_PONG_AB + [B] * 20)
            + in_drive("d3", PLAIN_AB) + in_drive("d4", PING_PONG_AC))
    changes = changes_for(make_raw, rows)
    table = tuning.main_advantages_with_intervals(changes, repeats=200)
    assert table.index.tolist()[-2:] == ["Border rule minus margin rule", "Border rule minus road only"]
    border = table.loc["Border rule (a round trip in another drive)"]
    assert border["low"] <= border["high"]
    assert (table["low"] < table["high"]).any()       # the draws really differ from each other
    # the same seed gives the same interval, a different seed (almost always) another one
    again = tuning.main_advantages_with_intervals(changes, repeats=200)
    assert table.equals(again)
    difference = table.loc["Border rule minus margin rule", "advantage_points"]
    assert difference == (border["advantage_points"] - table.loc["Margin rule (gain below 3 dB)", "advantage_points"])


def test_drive_by_drive_leaves_out_drives_without_round_trips(make_raw):
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB + [B] * 20) + in_drive("d2", PING_PONG_AB + [B] * 20)
                          + in_drive("d3", PLAIN_AB))
    table = tuning.drive_by_drive(changes)
    assert table.index.tolist() == ["d1", "d2"]
    assert table["round_trips"].tolist() == [1, 1]
    # in d1 the border rule acts on its round trip (known from d2) and on its one normal change at the same border
    assert table.loc["d1", "border_advantage"] == 0


# --- tables ----------------------------------------------------------------------------

def test_border_table_counts_round_trips_and_drives(make_raw):
    plain_ac = [A] * 10 + [C] * 20
    changes = changes_for(make_raw, in_drive("d1", PING_PONG_AB) + in_drive("d2", PING_PONG_AB)
                          + in_drive("d3", plain_ac))
    table = tuning.border_table(changes)
    columns = ["changes", "round_trips", "drives", "drives_with_round_trip"]
    assert table.index[0] == "A1-1 | A1-2"                               # most round trips first
    assert table.loc["A1-1 | A1-2", columns].tolist() == [4, 2, 2, 2]
    assert table.loc["A1-1 | B2-1", columns].tolist() == [1, 0, 1, 0]
    assert table.loc["A1-1 | A1-2", "in_round_trip_pct"] == 100
