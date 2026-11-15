"""Step 4 - mobility: find serving-cell changes and measure what they cost.

Important limit: G-NetTrack is an application-level logger. It does not see the
RRC signalling (measurement reports, handover commands), so this module finds
*serving-cell changes*: the moments where the logged cell or technology is
different from the previous sample. A cell change can be a handover, a cell
reselection or a re-establishment after a radio link failure; the log cannot
tell them apart.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Find the changes
# ----------------------------------------------------------------------------

def find_cell_changes(df):
    """Return one row per serving-cell change.

    The cell is identified by `cell` (the full cell identity). The short
    `sector` number is not enough: two sites can both have a sector 12.
    A change is only counted inside a drive, never between two drives.
    A sample without a cell or a technology (no service, for example) says
    nothing about a change: each sample is compared with the last one before
    it that has both.
    """
    df = df.dropna(subset=["cell", "tech"])
    before = df.groupby("drive")[["cell", "site", "tech", "rsrp"]].shift()
    has_previous = before["cell"].notna()          # False on the first sample of a drive
    changed = has_previous & ((df["cell"] != before["cell"]) | (df["tech"] != before["tech"]))

    events = df.loc[changed, ["drive", "sample", "time", "elapsed_s", "lat", "lon", "speed_kmh"]].copy()
    events["from_tech"] = before.loc[changed, "tech"]
    events["to_tech"] = df.loc[changed, "tech"]
    events["from_cell"] = before.loc[changed, "cell"].astype("int64")
    events["to_cell"] = df.loc[changed, "cell"].astype("int64")
    events["from_site"] = before.loc[changed, "site"]
    events["to_site"] = df.loc[changed, "site"]
    events["rsrp_before"] = before.loc[changed, "rsrp"]
    events["rsrp_after"] = df.loc[changed, "rsrp"]
    events["kind"] = [_kind(e) for e in events.itertuples()]

    # Time spent on the previous cell = time since the previous change
    # (or since the start of the drive for the first change).
    previous_change = events.groupby("drive")["elapsed_s"].shift().fillna(0)
    events["stay_s"] = events["elapsed_s"] - previous_change

    events = events.reset_index(drop=True)
    events.index.name = "event"
    return events


def _kind(event):
    """Name the type of change, the way an RF engineer would group them."""
    if event.from_tech != event.to_tech:
        return f"{event.from_tech} to {event.to_tech}"          # technology change
    if event.to_tech == "3G":
        return "3G cell change"       # on 3G the `site` column is the RNC, not a site
    if event.from_site == event.to_site:
        return f"{event.to_tech} intra-site"                    # sector to sector
    return f"{event.to_tech} inter-site"                        # site to site


# ----------------------------------------------------------------------------
# 2. Ping-pong
# ----------------------------------------------------------------------------

def mark_ping_pong(events, max_stay_s=5):
    """Flag ping-pong: A -> B -> A where the stay on B was `max_stay_s` or less.

    Four columns are added:
    * `next_stay_s` - how long the phone then stayed on the new cell (empty
      for the last change of a drive).
    * `ping_pong_out` - this change is the way out (A -> B) of a round trip.
    * `ping_pong_return` - this change is the way back (B -> A). Counting these
      gives the number of round trips.
    * `ping_pong` - this change is part of a round trip (the way out or the
      way back).

    A chain A -> B -> A -> B -> A counts as three round trips: every return
    within the time limit is one.

    Ping-pong wastes signalling and interrupts data twice for no gain. It
    usually points to handover settings (hysteresis, offsets, time-to-trigger)
    that are too loose for that border.
    """
    events = events.copy()
    next_to_cell = events.groupby("drive")["to_cell"].shift(-1)
    next_stay = events.groupby("drive")["stay_s"].shift(-1)
    # this event went A -> B, and the next one goes back to A after a short stay on B
    goes_out = (next_to_cell == events["from_cell"]) & (next_stay <= max_stay_s)
    comes_back = goes_out.groupby(events["drive"]).shift(1, fill_value=False)
    events["next_stay_s"] = next_stay
    events["ping_pong_out"] = goes_out
    events["ping_pong_return"] = comes_back
    events["ping_pong"] = goes_out | comes_back
    return events


def ping_pong_sensitivity(events, thresholds=(2, 3, 5, 10)):
    """How much does the ping-pong share depend on the chosen time limit?"""
    rows = []
    for max_stay_s in thresholds:
        marked = mark_ping_pong(events, max_stay_s)
        rows.append({
            "max_stay_s": max_stay_s,
            "round_trips": int(marked["ping_pong_return"].sum()),
            "changes_in_a_round_trip": int(marked["ping_pong"].sum()),
            "pct_of_changes": round(marked["ping_pong"].mean() * 100, 1),
        })
    return pd.DataFrame(rows).set_index("max_stay_s")


# ----------------------------------------------------------------------------
# 3. Summaries
# ----------------------------------------------------------------------------

def summarize_by_kind(events):
    """Count the changes of each kind, with ping-pong share and RSRP gain."""
    # RSRP gain makes no sense between 3G and LTE/5G: one side is RSCP.
    one_side_is_3g = (events["from_tech"] == "3G") != (events["to_tech"] == "3G")
    gain = (events["rsrp_after"] - events["rsrp_before"]).where(~one_side_is_3g)
    table = events.assign(rsrp_gain_db=gain).groupby("kind").agg(
        changes=("kind", "size"),
        ping_pong_pct=("ping_pong", lambda flags: flags.mean() * 100),
        median_stay_s=("stay_s", "median"),
        median_rsrp_before=("rsrp_before", "median"),
        median_rsrp_gain_db=("rsrp_gain_db", "median"),
    )
    return table.sort_values("changes", ascending=False).round(1)


def rate_per_drive(df, events):
    """Cell changes per minute and per km, for each drive."""
    table = pd.DataFrame({
        "minutes": df.groupby("drive")["elapsed_s"].max() / 60,
        "km": df.groupby("drive")["step_m"].sum() / 1000,
        "changes": events.groupby("drive").size(),
    })
    table["changes"] = table["changes"].fillna(0).astype(int)
    table["per_minute"] = table["changes"] / table["minutes"]
    table["per_km"] = table["changes"] / table["km"]
    return table.round(2)


# ----------------------------------------------------------------------------
# 4. What happens around a change
# ----------------------------------------------------------------------------

def samples_around(df, events, window=10):
    """Collect the samples around every cell change.

    Returns a table with one row per (event, offset): offset 0 is the first
    sample on the new cell, -1 the last sample on the old cell. Samples are
    about one second apart, so offsets are roughly seconds.
    """
    offsets = pd.DataFrame({"offset": np.arange(-window, window + 1)})
    # every event combined with every offset: -10, -9, ..., +10
    grid = events.reset_index()[["event", "drive", "sample", "kind"]].merge(offsets, how="cross")
    grid["sample"] = grid["sample"] + grid["offset"]
    columns = ["drive", "sample", "state", "dl_mbps", "rsrp", "snr", "cqi"]
    # the inner merge drops offsets that fall before the start or after the end of a drive
    return grid.merge(df[columns], on=["drive", "sample"], how="inner")


def throughput_window(df, events, window=10):
    """Like samples_around, but only samples taken while downloading."""
    around = samples_around(df, events, window)
    return around[around["state"] == "D"]


def isolated(events, window=10):
    """Keep the events with no other cell change within `window` samples.

    Around an isolated event the "before" and "after" throughput belong to
    one cell each, so the dip can be measured cleanly.
    """
    previous_sample = events.groupby("drive")["sample"].shift()
    next_sample = events.groupby("drive")["sample"].shift(-1)
    gap_before = (events["sample"] - previous_sample).fillna(np.inf)    # inf: no earlier change in this drive
    gap_after = (next_sample - events["sample"]).fillna(np.inf)
    return events[(gap_before > window) & (gap_after > window)]


def random_moments(df, n=3000, seed=0):
    """Pick random samples as a control group for the dip measurement.

    The same dip calculation is run on these moments. They are plain random
    samples (some are near a real change, most are not). If the dip around
    real cell changes is real, it must not show up here.
    """
    moments = df.sample(n, random_state=seed)[["drive", "sample"]].reset_index(drop=True)
    moments["kind"] = "Random moment (control)"
    moments.index.name = "event"
    return moments


def dip_per_event(windows):
    """For every event: mean throughput before, around and after the change.

    before = offsets -10..-4, around = offsets -1..+1, after = offsets +4..+10.
    Events that miss one of the three parts (drive start or end, no download
    samples) are dropped.
    """
    def mean_between(low, high):
        part = windows[windows["offset"].between(low, high)]
        return part.groupby("event")["dl_mbps"].mean()

    table = pd.DataFrame({
        "before_mbps": mean_between(-10, -4),
        "around_mbps": mean_between(-1, 1),
        "after_mbps": mean_between(4, 10),
    }).dropna()
    table["kind"] = windows.groupby("event")["kind"].first()
    return table


def median_interval(values, repeats=2000, seed=0):
    """95% bootstrap interval of a median: how sure are we about it?

    Draw the same number of events again and again (with replacement), take
    the median every time, and report the range that holds 95% of the results.
    A wide range means: few events, or events that differ a lot.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(repeats, len(values)), replace=True)
    medians = np.median(resampled, axis=1)
    return np.percentile(medians, 2.5), np.percentile(medians, 97.5)


def dip_summary(dips, min_before_mbps=1.0, min_events=20):
    """Typical dip per kind of change, with a 95% interval.

    * drop = how much lower the throughput is around the change than before.
    * after_vs_before = the throughput 4-10 s later, compared with before.
    Both are medians over the events. Only events with at least
    `min_before_mbps` before the change are used (a ratio to almost zero is
    meaningless), and only kinds with at least `min_events` such events.
    """
    dips = dips[dips["before_mbps"] >= min_before_mbps].copy()
    dips["drop_pct"] = (1 - dips["around_mbps"] / dips["before_mbps"]) * 100
    dips["after_vs_before_pct"] = (dips["after_mbps"] / dips["before_mbps"] - 1) * 100

    def one_row(part):
        drop_low, drop_high = median_interval(part["drop_pct"])
        after_low, after_high = median_interval(part["after_vs_before_pct"])
        return {
            "events": len(part),
            "before_mbps": part["before_mbps"].median(),
            "around_mbps": part["around_mbps"].median(),
            "after_mbps": part["after_mbps"].median(),
            "median_drop_pct": part["drop_pct"].median(),
            "drop_low": drop_low,
            "drop_high": drop_high,
            "median_after_vs_before_pct": part["after_vs_before_pct"].median(),
            "after_low": after_low,
            "after_high": after_high,
        }

    rows = {"All changes": one_row(dips)}
    for kind, part in dips.groupby("kind"):
        if len(part) >= min_events:
            rows[kind] = one_row(part)
    table = pd.DataFrame(rows).T
    return table.sort_values("events", ascending=False).round(1)
