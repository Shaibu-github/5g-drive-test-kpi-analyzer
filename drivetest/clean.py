"""Step 2 - clean: turn the raw logs into one tidy table that the analysis can trust.

What cleaning does (each rule comes from a problem found in the raw data):

1. Parse the timestamp text into real times.
2. Group the six network modes into three technologies: 5G, LTE, 3G.
3. Replace impossible radio values (logger placeholders) with NaN.
4. Blank the radio columns that are only placeholders on 3G rows.
5. Convert bitrates from kbps to Mbps.
6. Fix the GPS: remove a stale first position, measure distance moved.
7. Flag rows where the cell identity does not fit the technology.

Rows are never deleted. A bad value becomes NaN, the rest of the row stays.
"""
import numpy as np
import pandas as pd

# G-NetTrack "NetworkMode" -> technology
TECH = {"5G": "5G", "LTE": "LTE", "HSPA+": "3G", "HSDPA": "3G", "HSUPA": "3G", "UMTS": "3G"}

# Raw column name -> name used in the rest of the project
RENAME = {
    "Timestamp": "time",
    "Latitude": "lat",
    "Longitude": "lon",
    "Speed": "speed_kmh",
    "NetworkMode": "network_mode",
    "RAWCELLID": "cell",      # full cell identity: unique for every cell
    "NODEHEX": "site",        # eNB ID (LTE/5G) or RNC ID (3G), in hex
    "CellID": "sector",       # only the cell number inside the site: NOT unique
    "RSRP": "rsrp",
    "RSRQ": "rsrq",
    "SNR": "snr",
    "CQI": "cqi",
    "RSSI": "rssi",
    "NRxRSRP": "nbr_rsrp",    # strongest neighbour cell
    "NRxRSRQ": "nbr_rsrq",
    "DL_bitrate": "dl_kbps",
    "UL_bitrate": "ul_kbps",
    "State": "state",
}

RADIO_COLUMNS = ["rsrp", "rsrq", "snr", "cqi", "rssi", "nbr_rsrp", "nbr_rsrq"]

# A real measurement must be inside these ranges (based on the 3GPP / Android
# reporting ranges). Values outside are logger placeholders, for example
# RSRP = -200 or RSRQ = 2147483647 (the largest 32-bit integer).
VALID_RANGE = {
    "rsrp": (-140, -43),      # dBm
    "rsrq": (-34, 3),         # dB
    "snr": (-20, 40),         # dB
    "nbr_rsrp": (-140, -43),  # dBm
    "nbr_rsrq": (-34, 3),     # dB
}

# Columns of the clean table, in order. The ping columns are dropped because
# they are about 93% empty.
CLEAN_COLUMNS = [
    "drive", "time", "elapsed_s", "sample",
    "lat", "lon", "gps_fix", "step_m", "speed_kmh",
    "tech", "network_mode", "cell", "site", "sector", "identity_ok",
    "rsrp", "rsrq", "snr", "cqi", "rssi", "nbr_rsrp", "nbr_rsrq",
    "dl_mbps", "ul_mbps", "state",
]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two GPS points given in degrees."""
    earth_radius_m = 6_371_000
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * earth_radius_m * np.arcsin(np.sqrt(a))


def clean(raw, max_first_jump_m=300):
    """Return the clean table. `raw` is the output of data.load_logs()."""
    df = raw.rename(columns=RENAME).reset_index(drop=True)

    # --- 1. time ---------------------------------------------------------
    # "2019.12.14_10.16.30" -> timestamp. Some timestamps repeat: the logger
    # samples about once per second but stamps whole seconds, so two samples
    # can land in the same second. They are different measurements, so we keep
    # them and number the samples of each drive 0, 1, 2, ...
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d_%H.%M.%S")
    df["sample"] = df.groupby("drive").cumcount()
    drive_start = df.groupby("drive")["time"].transform("min")
    df["elapsed_s"] = (df["time"] - drive_start).dt.total_seconds()

    # --- 2. technology ---------------------------------------------------
    unknown_modes = set(df["network_mode"]) - set(TECH)
    if unknown_modes:
        # stop early: an unmapped mode would silently become a missing technology
        raise ValueError(f"unknown network mode(s) {sorted(unknown_modes)}: add them to TECH")
    df["tech"] = df["network_mode"].map(TECH)

    # --- 3. impossible radio values -> NaN -------------------------------
    df[RADIO_COLUMNS] = df[RADIO_COLUMNS].astype("float64")  # floats can hold NaN
    # When the serving RSRP is a placeholder (-200), RSRQ and SNR in the same
    # row are placeholders too (-20 and -30), so blank all three together.
    low, high = VALID_RANGE["rsrp"]
    bad_serving = ~df["rsrp"].between(low, high)
    df.loc[bad_serving, ["rsrp", "rsrq", "snr"]] = np.nan
    for column, (low, high) in VALID_RANGE.items():
        df[column] = df[column].where(df[column].between(low, high))

    # --- 4. 3G rows --------------------------------------------------------
    # On 3G the "RSRP" column really holds RSCP, which is a valid signal level.
    # But RSRQ is almost always -2 and the neighbour RSRP always -51 on these
    # rows: they are placeholders, not measurements.
    is_3g = df["tech"] == "3G"
    df.loc[is_3g, ["rsrq", "nbr_rsrp", "nbr_rsrq"]] = np.nan

    # --- 5. units --------------------------------------------------------
    df["dl_mbps"] = df["dl_kbps"] / 1000
    df["ul_mbps"] = df["ul_kbps"] / 1000

    # --- 6. GPS ----------------------------------------------------------
    df = _clean_gps(df, max_first_jump_m)

    # --- 7. does the cell identity fit the technology? -------------------
    # LTE/5G: cell identity = eNB ID * 256 + cell number.
    # 3G:     cell identity = RNC ID * 65536 + cell number.
    # When the phone changes technology, the logger sometimes updates the
    # network mode and the cell identity in different rows. In those rows the
    # formula fails; they are flagged so they can be counted and left out of
    # the cell and site counts.
    site_number = df["site"].apply(lambda hex_text: int(hex_text, 16))
    cells_per_site = np.where(df["tech"] == "3G", 65536, 256)
    df["identity_ok"] = df["cell"] == site_number * cells_per_site + df["sector"]

    return df[CLEAN_COLUMNS]


def _clean_gps(df, max_first_jump_m):
    """Add `step_m` (metres moved since the previous sample) and `gps_fix`.

    The logger only writes a new position about every 100 m. Between two
    position updates, latitude, longitude and speed stay frozen.
    """
    previous_lat = df.groupby("drive")["lat"].shift()
    previous_lon = df.groupby("drive")["lon"].shift()
    df["step_m"] = haversine_m(previous_lat, previous_lon, df["lat"], df["lon"]).fillna(0)

    # A drive can start with a stale position (the phone's last known location
    # before the GPS locks). We see it as a first move that is far too long.
    for _, rows in df.groupby("drive"):
        moves = rows.index[rows["step_m"] > 0]
        if len(moves) > 0 and rows.loc[moves[0], "step_m"] > max_first_jump_m:
            first_move = moves[0]
            stale_rows = rows.index[rows.index < first_move]
            df.loc[stale_rows, ["lat", "lon"]] = np.nan
            df.loc[first_move, "step_m"] = 0

    # gps_fix counts the position updates inside a drive: all samples logged
    # at the same position share one number. Used later to draw the maps.
    previous_lat = df.groupby("drive")["lat"].shift()
    previous_lon = df.groupby("drive")["lon"].shift()
    position_changed = (df["lat"] != previous_lat) | (df["lon"] != previous_lon)
    df["gps_fix"] = position_changed.groupby(df["drive"]).cumsum()
    return df


def downloading(df):
    """Keep only samples taken while the download was running (state "D").

    In the other states (idle, voice call) a low bitrate says nothing about
    the radio link, so throughput analysis must not use those samples.
    """
    return df[df["state"] == "D"]


def cleaning_report(raw, df):
    """Count, for each radio column, how many values cleaning turned into NaN."""
    raw = raw.rename(columns=RENAME)
    rows = []
    for column in RADIO_COLUMNS:
        missing_raw = int(raw[column].isna().sum())
        missing_clean = int(df[column].isna().sum())
        rows.append({
            "column": column,
            "missing_in_raw": missing_raw,
            "blanked_by_cleaning": missing_clean - missing_raw,
            "valid_after_cleaning": len(df) - missing_clean,
            "valid_pct": round(100 * (1 - missing_clean / len(df)), 1),
        })
    return pd.DataFrame(rows).set_index("column")
