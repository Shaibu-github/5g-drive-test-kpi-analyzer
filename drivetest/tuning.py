"""Step 4b - handover tuning: does ping-pong belong to particular borders?

About a third of the cell changes belong to a ping-pong round trip
(A -> B -> A within 5 seconds). Before anyone tunes a network for that, there
is a simpler question: what tells a ping-pong apart from a normal change?

* If it is the signal gain (the new cell is hardly stronger than the old one),
  a bigger handover margin for the whole network would be the place to look.
* If it is the border (the same pairs of cells again and again), single cell
  pairs would be the place to look.

This module turns each idea into a rule. A rule says for every cell change:
"I would act on this one". A rule is good when it acts on many round trips
and on few normal changes.

Limits: the log is application-level. It shows the RSRP of the serving cell
about once per second, in 1 dB steps. It does not show the measurement reports
the network decided on, the time-to-trigger, or the frequency of a cell. So
this is a study of which information separates ping-pong from normal changes.
It is not a simulation of the network, and a rule is not a network parameter.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Prepare the cell changes
# ----------------------------------------------------------------------------

def cell_name(cell, tech):
    """Readable name of a cell: site in hex and cell number, like "A81B-12".

    LTE:  cell identity = eNB ID * 256 + cell number. The 5G rows of this
          dataset (5G NSA) carry the identity of the LTE anchor cell, so the
          same formula applies to them.
    3G:   cell identity = RNC ID * 65536 + cell number.
    """
    cells_per_site = 65536 if tech == "3G" else 256
    return f"{int(cell) // cells_per_site:X}-{int(cell) % cells_per_site}"


def radio_changes(events):
    """Keep the LTE and 5G cell changes and add `border` and `gain_db`.

    * border  - the two cells of the change, always in the same order, so
                A -> B and B -> A are the same border.
    * gain_db - RSRP on the new cell minus RSRP on the old cell.
    Changes from or to 3G are left out: 3G reports RSCP, and RSCP cannot be
    compared with RSRP. A change without an RSRP value on both sides is left
    out too (there is one in the dataset).
    """
    changes = events[(events["from_tech"] != "3G") & (events["to_tech"] != "3G")].copy()
    changes["gain_db"] = changes["rsrp_after"] - changes["rsrp_before"]
    changes = changes.dropna(subset="gain_db")

    def border(change):
        cells = sorted([(change.from_cell, change.from_tech), (change.to_cell, change.to_tech)])
        return " | ".join(cell_name(cell, tech) for cell, tech in cells)

    changes["border"] = [border(change) for change in changes.itertuples()]
    return changes


def without_short_round_trips(changes, max_stay_s=2):
    """Leave out the round trips with `max_stay_s` or less on the second cell.

    More than half of the round trips last exactly 2 seconds. That is close to
    how often the logger refreshes its radio values, so some of them may be
    reporting effects and not real handovers. Both legs of such a round trip
    are removed. Run an analysis again on the result to see how much it
    depends on them.
    """
    short_way_out = changes["ping_pong_out"] & (changes["next_stay_s"] <= max_stay_s)
    short_way_back = changes["ping_pong_return"] & (changes["stay_s"] <= max_stay_s)
    return changes[~(short_way_out | short_way_back)]


def border_table(changes):
    """One row per border: how busy it is and how much of that is ping-pong."""
    same_tech = changes["from_tech"] == changes["to_tech"]
    changes = changes.assign(border_kind=changes["kind"].where(same_tech, "between 5G and LTE"))
    table = changes.groupby("border").agg(
        kind=("border_kind", "first"),
        changes=("border", "size"),
        round_trips=("ping_pong_out", "sum"),
        in_round_trip_pct=("ping_pong", lambda flags: round(flags.mean() * 100, 1)),
        drives=("drive", "nunique"),
        lat=("lat", "median"),
        lon=("lon", "median"),
    )
    with_round_trip = changes[changes["ping_pong_out"]].groupby("border")["drive"].nunique()
    table.insert(5, "drives_with_round_trip", with_round_trip.reindex(table.index).fillna(0).astype(int))
    return table.sort_values(["round_trips", "changes"], ascending=False)


# ----------------------------------------------------------------------------
# 2. Three rules. Each one says, for every change: would I act on it?
# ----------------------------------------------------------------------------

def seen_elsewhere(changes, counted):
    """For every change: how many `counted` changes are at its border in the OTHER drives?

    `counted` is a True/False column, for example "starts a round trip".
    The drive of the change itself is always left out. Without that, a drive
    would be judged with knowledge taken from itself: checking the answer
    with the answer.
    """
    per_drive = changes[counted].groupby(["border", "drive"]).size()
    in_all_drives = per_drive.groupby("border").sum().reindex(changes["border"]).fillna(0).to_numpy()
    own_key = pd.MultiIndex.from_frame(changes[["border", "drive"]])
    in_own_drive = per_drive.reindex(own_key).fillna(0).to_numpy()
    return pd.Series(in_all_drives - in_own_drive, index=changes.index)


def margin_acts(changes, margin_db):
    """Margin rule: act on every change that gains less than `margin_db`."""
    return changes["gain_db"] < margin_db


def border_acts(changes, min_round_trips=1):
    """Border rule: act at borders with `min_round_trips` or more round trips in the other drives."""
    return seen_elsewhere(changes, changes["ping_pong_out"]) >= min_round_trips


def road_acts(changes):
    """Control rule: act at every border that another drive also crossed.

    It knows which roads are driven often, and nothing about ping-pong. The
    border rule must beat this one. Otherwise it only knows the roads.
    """
    every_change = pd.Series(True, index=changes.index)
    return seen_elsewhere(changes, every_change) >= 1


# ----------------------------------------------------------------------------
# 3. Score the rules
# ----------------------------------------------------------------------------

def score_rule(changes, acts):
    """How well does a rule separate ping-pong from normal changes?

    * caught    - round trips the rule acts on, counted at their way out
    * affected  - normal changes the rule also acts on
    * advantage - caught % minus affected %, in percentage points. Zero means
                  the rule cannot tell a ping-pong from a normal change.
    The way back of a round trip is in neither group: it is the result of the
    way out, not a separate decision.
    """
    way_out = changes["ping_pong_out"]
    normal = ~changes["ping_pong"]
    round_trips, normal_changes = int(way_out.sum()), int(normal.sum())
    caught, affected = int((way_out & acts).sum()), int((normal & acts).sum())
    caught_pct = caught / round_trips * 100 if round_trips else np.nan
    affected_pct = affected / normal_changes * 100 if normal_changes else np.nan
    return {
        "round_trips": round_trips, "round_trips_caught": caught, "round_trips_caught_pct": caught_pct,
        "normal_changes": normal_changes, "normal_changes_affected": affected, "normal_changes_affected_pct": affected_pct,
        "advantage_points": caught_pct - affected_pct,
    }


def compare_rules(changes, margins=(0, 1, 2, 3, 4, 5, 6, 8), min_round_trips=(1, 2, 3, 5)):
    """Score every rule for several settings (the points of the figure)."""
    rows = []
    for margin_db in margins:
        rows.append({"rule": "Margin", "setting": f"gain below {margin_db} dB",
                     **score_rule(changes, margin_acts(changes, margin_db))})
    for minimum in min_round_trips:
        rows.append({"rule": "Border", "setting": f"{minimum}+ round trips in other drives",
                     **score_rule(changes, border_acts(changes, minimum))})
    rows.append({"rule": "Road only", "setting": "crossed in another drive", **score_rule(changes, road_acts(changes))})
    return pd.DataFrame(rows)


def main_advantages(changes):
    """Advantage of the three main rules, and how far the border rule is ahead of the other two."""
    margin = score_rule(changes, margin_acts(changes, 3))["advantage_points"]
    border = score_rule(changes, border_acts(changes, 1))["advantage_points"]
    road = score_rule(changes, road_acts(changes))["advantage_points"]
    return {
        "Margin rule (gain below 3 dB)": margin,
        "Border rule (a round trip in another drive)": border,
        "Road only (crossed in another drive)": road,
        "Border rule minus margin rule": border - margin,
        "Border rule minus road only": border - road,
    }


def main_advantages_with_intervals(changes, repeats=500, seed=0):
    """The main advantages with a 95% interval: how much do they depend on which drives we have?

    Bootstrap over drives. One draw = pick as many drives as there are, with
    replacement, and run the WHOLE procedure on them again. The borders are
    picked again inside every draw. (Resampling only the final scores would
    hide the uncertainty of picking the borders.) A drive that is drawn twice
    keeps its name, so its two copies never teach each other.
    """
    by_drive = dict(list(changes.groupby("drive")))
    drives = list(by_drive)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        picked = rng.choice(drives, size=len(drives))
        sample = pd.concat([by_drive[drive] for drive in picked], ignore_index=True)
        draws.append(main_advantages(sample))
    draws = pd.DataFrame(draws)                       # a draw without round trips gives NaN and is ignored below
    table = pd.DataFrame({"advantage_points": main_advantages(changes)})
    table["low"] = draws.quantile(0.025)
    table["high"] = draws.quantile(0.975)
    return table


def drive_by_drive(changes):
    """The border rule and the 3 dB margin rule, scored on every drive alone.

    A plain cross-check of the intervals: in how many drives is the border
    rule ahead? Drives without a round trip cannot be scored and are left out.
    """
    border, margin = border_acts(changes, 1), margin_acts(changes, 3)
    rows = {}
    for drive, part in changes.groupby("drive"):
        if part["ping_pong_out"].any():
            rows[drive] = {
                "round_trips": int(part["ping_pong_out"].sum()),
                "border_advantage": score_rule(part, border[part.index])["advantage_points"],
                "margin_advantage": score_rule(part, margin[part.index])["advantage_points"],
            }
    return pd.DataFrame(rows).T.astype({"round_trips": int})
