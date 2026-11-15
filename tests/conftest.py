"""Shared helper for the tests: build a tiny fake raw log.

The tests never need the real dataset. Each test builds a few rows where the
right answer is known in advance, then checks that the code finds it.
"""
import pandas as pd
import pytest

DEFAULT_ROW = {
    "Longitude": -8.40, "Latitude": 51.90, "Speed": 30, "Operatorname": "B",
    "CellID": 1, "NetworkMode": "LTE", "RSRP": -90, "RSRQ": -10.0, "SNR": 5.0, "CQI": 10.0, "RSSI": -70.0,
    "DL_bitrate": 10000, "UL_bitrate": 100, "State": "D",
    # site A1 (hex) = 161, cell 1 -> LTE cell identity 161 * 256 + 1 = 41217
    "NODEHEX": "A1", "RAWCELLID": 41217, "NRxRSRP": -95.0, "NRxRSRQ": -12.0, "drive": "drive1",
}


def build_raw(rows):
    """Return a raw log. `rows` is a list of dicts that override DEFAULT_ROW.

    Unless a row gives its own Timestamp, rows are one second apart.
    """
    start = pd.Timestamp("2020-01-01 10:00:00")
    full_rows = []
    for number, changes in enumerate(rows):
        row = dict(DEFAULT_ROW)
        row["Timestamp"] = (start + pd.Timedelta(seconds=number)).strftime("%Y.%m.%d_%H.%M.%S")
        row.update(changes)
        full_rows.append(row)
    return pd.DataFrame(full_rows)


@pytest.fixture
def make_raw():
    return build_raw
