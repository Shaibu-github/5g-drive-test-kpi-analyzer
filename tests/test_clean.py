import numpy as np
import pandas as pd
import pytest

from drivetest import data
from drivetest.clean import clean, downloading, haversine_m


def test_dash_is_read_as_missing_but_negative_numbers_survive(tmp_path):
    log_file = tmp_path / "B_2020.01.01_10.00.00.csv"
    log_file.write_text("Timestamp,RSRP,SNR\n2020.01.01_10.00.00,-99,-\n")
    log = data.load_log(log_file)
    assert log.loc[0, "RSRP"] == -99
    assert np.isnan(log.loc[0, "SNR"])
    assert log.loc[0, "drive"] == "2020.01.01_10.00.00"      # drive ID comes from the file name


def test_hex_ids_stay_text(tmp_path):
    log_file = tmp_path / "B_2020.01.01_10.00.00.csv"
    log_file.write_text("Timestamp,NODEHEX\n2020.01.01_10.00.00,1E3\n2020.01.01_10.00.01,0100\n")
    log = data.load_log(log_file)
    assert log["NODEHEX"].tolist() == ["1E3", "0100"]         # not the numbers 1000.0 and 100.0


def test_placeholder_rsrp_blanks_the_whole_serving_measurement(make_raw):
    df = clean(make_raw([{}, {"RSRP": -200, "RSRQ": -20.0, "SNR": -30.0}]))
    assert df.loc[0, "rsrp"] == -90                           # a normal row is untouched
    assert df.loc[1, ["rsrp", "rsrq", "snr"]].isna().all()
    assert df.loc[1, "dl_mbps"] == 10                         # the row itself is kept


def test_garbage_neighbour_rsrq_is_blanked(make_raw):
    df = clean(make_raw([{"NRxRSRQ": 2147483647}, {"NRxRSRQ": -204}, {"NRxRSRQ": -11}]))
    assert df["nbr_rsrq"].isna().tolist() == [True, True, False]


def test_3g_modes_become_3g_and_lose_their_placeholder_columns(make_raw):
    df = clean(make_raw([{"NetworkMode": "HSPA+", "RSRP": -70, "RSRQ": -2.0, "NRxRSRP": -51.0}]))
    assert df.loc[0, "tech"] == "3G"
    assert df.loc[0, "rsrp"] == -70                           # RSCP is a real signal level: keep it
    assert df.loc[0, ["rsrq", "nbr_rsrp", "nbr_rsrq"]].isna().all()


def test_unknown_network_mode_stops_the_cleaning(make_raw):
    with pytest.raises(ValueError, match="EDGE"):
        clean(make_raw([{}, {"NetworkMode": "EDGE"}]))


def test_identity_check_flags_a_cell_id_that_does_not_fit_the_technology(make_raw):
    lte_cell = {"NODEHEX": "A1", "CellID": 1, "RAWCELLID": 161 * 256 + 1}
    cell_3g = {"NetworkMode": "HSPA+", "NODEHEX": "C", "CellID": 500, "RAWCELLID": 12 * 65536 + 500}
    stale = dict(lte_cell, NetworkMode="HSPA+")               # mode says 3G, identity is still the LTE cell
    df = clean(make_raw([lte_cell, cell_3g, stale]))
    assert df["identity_ok"].tolist() == [True, True, False]


def test_repeated_timestamps_are_kept_as_separate_samples(make_raw):
    same_second = "2020.01.01_10.00.00"
    df = clean(make_raw([{"Timestamp": same_second, "DL_bitrate": 7}, {"Timestamp": same_second, "DL_bitrate": 0}]))
    assert len(df) == 2
    assert df["sample"].tolist() == [0, 1]


def test_bitrate_is_converted_to_mbps(make_raw):
    df = clean(make_raw([{"DL_bitrate": 15000, "UL_bitrate": 500}]))
    assert df.loc[0, "dl_mbps"] == 15
    assert df.loc[0, "ul_mbps"] == 0.5


def test_haversine_matches_a_known_distance():
    # one thousandth of a degree of latitude is about 111 m everywhere on Earth
    assert abs(haversine_m(51.900, -8.4, 51.901, -8.4) - 111.2) < 0.5


def test_stale_first_gps_position_is_removed(make_raw):
    far_away = {"Latitude": 51.94, "Longitude": -8.40}        # about 4.4 km from the next position
    df = clean(make_raw([far_away, {}, {}, {"Latitude": 51.901}]))
    assert np.isnan(df.loc[0, "lat"])
    assert df["step_m"].max() < 200                           # the 4.4 km jump is not counted as driving
    assert df["step_m"].sum() > 100                           # the real 111 m move still is


def test_gps_fix_numbers_the_position_updates(make_raw):
    df = clean(make_raw([{}, {}, {"Latitude": 51.901}, {"Latitude": 51.901}]))
    assert df["gps_fix"].tolist() == [1, 1, 2, 2]


def test_downloading_keeps_only_state_d(make_raw):
    df = clean(make_raw([{"State": "D"}, {"State": "I"}, {"State": "VD"}]))
    assert downloading(df)["state"].tolist() == ["D"]


def test_a_new_drive_restarts_sample_number_time_and_distance(make_raw):
    near = {"Latitude": 51.901}                               # drive "b" starts 111 m from where "a" ended
    df = clean(make_raw([{"drive": "a"}, {"drive": "a"}, dict(near, drive="b"), dict(near, drive="b")]))
    assert df["sample"].tolist() == [0, 1, 0, 1]
    assert df["elapsed_s"].tolist() == [0, 1, 0, 1]
    assert df["step_m"].sum() == 0                            # the jump between two drives is not driving
    assert pd.api.types.is_datetime64_any_dtype(df["time"])
