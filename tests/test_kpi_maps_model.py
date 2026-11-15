import numpy as np
import pandas as pd

from drivetest import handover, kpi, maps, model
from drivetest.clean import clean


# --- kpi -------------------------------------------------------------------

def test_cdf_ends_at_100_percent():
    x, y = kpi.cdf([4, 1, 3, 2, np.nan])
    assert x.tolist() == [1, 2, 3, 4]
    assert y.tolist() == [25, 50, 75, 100]


def test_rsrp_class_edges():
    classes = kpi.legend_class(pd.Series([-100.5, -100, -90, -80, -60]), kpi.RSRP_LEGEND).astype(str).tolist()
    assert classes[0].startswith("Poor")
    assert classes[1].startswith("Fair")                      # exactly -100 is already "Fair"
    assert classes[2].startswith("Good")
    assert classes[3].startswith("Excellent") and classes[4].startswith("Excellent")


def test_throughput_summary_ignores_idle_samples(make_raw):
    df = clean(make_raw([{"DL_bitrate": 10000}, {"DL_bitrate": 30000}, {"State": "I", "DL_bitrate": 0}]))
    summary = kpi.throughput_summary(df)
    assert summary.loc["LTE", "samples"] == 2
    assert summary.loc["LTE", "mean_mbps"] == 20


def test_percentiles_and_shares_have_the_right_direction_and_unit(make_raw):
    rows = [{"RSRP": value, "DL_bitrate": 500} for value in range(-110, -60)]      # 50 samples, all 0.5 Mbps
    df = clean(make_raw(rows))
    radio = kpi.radio_summary(df)
    assert radio.loc["LTE", "rsrp_p10"] < radio.loc["LTE", "rsrp_median"]           # p10 is the weak end
    assert kpi.throughput_summary(df).loc["LTE", "pct_below_1mbps"] == 100          # percent, not a 0-1 share
    assert kpi.tech_share(df)["LTE"] == 100
    assert kpi.coverage_classes(df, techs=("LTE",)).loc["LTE"].sum().round() == 100


def test_overview_counts_cells_only_on_rows_with_a_trusted_identity(make_raw):
    good = {"NODEHEX": "A1", "CellID": 1, "RAWCELLID": 161 * 256 + 1}
    glitch = {"NODEHEX": "FF", "CellID": 9, "RAWCELLID": 12345}                    # identity does not fit
    summary = kpi.overview(clean(make_raw([good, good, glitch])))
    assert summary["cells"] == 1 and summary["sites"] == 1
    assert summary["rows_with_identity_mismatch"] == 1


# --- maps ------------------------------------------------------------------

WEAK_PLACE = {"Latitude": 51.9105, "Longitude": -8.4105, "RSRP": -108}


def spots_for(make_raw, rows):
    df = clean(make_raw(rows))
    bins = maps.road_bins(df, handover.find_cell_changes(df))
    return bins, maps.weak_spots(bins)


def test_road_bins_group_nearby_samples_and_find_a_weak_spot(make_raw):
    rows = [dict(WEAK_PLACE, drive="d1")] * 5 + [dict(WEAK_PLACE, drive="d2")] * 5 + [{}] * 5
    bins, spots = spots_for(make_raw, rows)
    assert len(bins) == 2                                     # the weak place and the default place
    assert spots["problem"].tolist() == ["Weak coverage"]
    assert spots.iloc[0]["rsrp_drives"] == 2


def test_a_weak_bin_seen_in_one_drive_is_not_a_weak_spot(make_raw):
    _, spots = spots_for(make_raw, [dict(WEAK_PLACE, drive="d1")] * 10)
    assert len(spots) == 0


def test_a_second_drive_on_3g_does_not_confirm_a_weak_spot(make_raw):
    # d2 passed the place on 3G: it has no RSRP there, so the weak RSRP is still from one drive only
    on_3g = dict(WEAK_PLACE, drive="d2", NetworkMode="HSPA+")
    bins, spots = spots_for(make_raw, [dict(WEAK_PLACE, drive="d1")] * 5 + [on_3g] * 5)
    assert bins.iloc[0]["drives"] == 2 and bins.iloc[0]["rsrp_drives"] == 1
    assert len(spots) == 0


def test_bin_rsrp_needs_enough_samples(make_raw):
    on_3g = dict(WEAK_PLACE, NetworkMode="HSPA+")
    bins, _ = spots_for(make_raw, [dict(WEAK_PLACE)] * 2 + [on_3g] * 8)            # only 2 RSRP samples
    assert np.isnan(bins.iloc[0]["rsrp"])


def test_low_throughput_with_good_coverage_is_its_own_problem(make_raw):
    slow = {"RSRP": -75, "DL_bitrate": 500}                   # strong signal, 0.5 Mbps
    idle = {"RSRP": -75, "DL_bitrate": 0, "State": "I"}       # idle samples must not count as slow
    rows = [dict(slow, drive="d1")] * 5 + [dict(slow, drive="d2")] * 5 + [dict(idle, drive="d2")] * 20
    bins, spots = spots_for(make_raw, rows)
    assert bins.iloc[0]["dl_mbps"] == 0.5
    assert spots["problem"].tolist() == ["Low throughput with good coverage"]


def test_bins_are_100_m_wide_in_both_directions():
    # a fine line of points going east, and one going north
    east = pd.DataFrame({"lat": 51.9, "lon": np.linspace(-8.41, -8.40, 500)})
    north = pd.DataFrame({"lat": np.linspace(51.90, 51.91, 500), "lon": -8.4})
    lon_centres = np.sort(maps.add_bin(east, size_m=100, reference_lat=51.9)["bin_lon"].unique())
    lat_centres = np.sort(maps.add_bin(north, size_m=100, reference_lat=51.9)["bin_lat"].unique())
    east_width_m = np.diff(lon_centres)[0] * maps.METRES_PER_DEGREE * np.cos(np.radians(51.9))
    north_width_m = np.diff(lat_centres)[0] * maps.METRES_PER_DEGREE
    assert abs(east_width_m - 100) < 1                        # without the cos(latitude) fix this would be 62 m
    assert abs(north_width_m - 100) < 1


# --- model -----------------------------------------------------------------

def seconds(drive, start, count, **changes):
    """`count` one-second rows of one drive, starting `start` seconds into it."""
    zero = pd.Timestamp("2020-01-01 10:00:00")
    return [dict(changes, drive=drive, Timestamp=(zero + pd.Timedelta(seconds=start + s)).strftime("%Y.%m.%d_%H.%M.%S"))
            for s in range(count)]


def test_windows_average_ten_seconds(make_raw):
    rows = seconds("d1", 0, 10, DL_bitrate=10000) + seconds("d1", 10, 10, DL_bitrate=30000)
    windows = model.make_windows(clean(make_raw(rows)))
    assert windows["dl_mbps"].tolist() == [10, 30]
    assert np.isnan(windows.loc[0, "previous_mbps"])          # the first window has no history
    assert windows.loc[1, "previous_mbps"] == 10


def test_history_is_missing_after_a_gap(make_raw):
    rows = seconds("d1", 0, 10, DL_bitrate=10000) + seconds("d1", 20, 10, DL_bitrate=30000)    # seconds 10-19 missing
    windows = model.make_windows(clean(make_raw(rows)))
    assert windows["window"].tolist() == [0, 2]
    assert windows["previous_mbps"].isna().all()


def test_history_is_never_taken_from_another_drive(make_raw):
    # d2 is idle for its first 10 s, so its first usable window is number 1,
    # right "after" window 0 of d1 if the drives were not kept apart.
    rows = (seconds("d1", 0, 10, DL_bitrate=10000)
            + seconds("d2", 0, 10, State="I") + seconds("d2", 10, 10, DL_bitrate=50000))
    windows = model.make_windows(clean(make_raw(rows)))
    assert windows["window"].tolist() == [0, 1]
    assert windows["previous_mbps"].isna().all()


def test_windows_leave_out_3g_idle_and_nearly_empty_windows(make_raw):
    rows = seconds("d1", 0, 10, NetworkMode="HSPA+") + seconds("d2", 0, 10, State="I") + seconds("d3", 0, 3)
    assert len(model.make_windows(clean(make_raw(rows)))) == 0


def test_uplink_bitrate_and_speed_are_never_model_inputs():
    # uplink carries the TCP ACKs of the download (it would leak the answer); speed is frozen by the logger
    for features in model.FEATURE_SETS.values():
        assert not {"ul_mbps", "dl_mbps", "speed_kmh"} & set(features)


def test_a_drive_is_never_in_training_and_test_together(make_raw):
    rows = []
    for number in range(6):
        rows += seconds(f"d{number}", 0, 30)
    windows = model.make_windows(clean(make_raw(rows)))
    for train_rows, test_rows in model.drive_folds(windows, n_splits=3):
        train_drives = set(windows["drive"].iloc[train_rows])
        test_drives = set(windows["drive"].iloc[test_rows])
        assert train_drives.isdisjoint(test_drives)


def test_score_of_a_perfect_prediction():
    actual = np.array([1.0, 10.0, 100.0])
    result = model.score(actual, actual)
    assert result["r2_log"] == 1 and result["mae_mbps"] == 0


def test_the_network_can_learn_a_simple_rule():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    y = 2 * x[:, 0] - x[:, 1]                                 # the rule to learn
    predicted = model.predict_network(model.train_network(x, y), x)
    assert np.corrcoef(predicted, y)[0, 1] > 0.98


def test_a_prediction_does_not_depend_on_the_other_test_rows():
    # If scaling or gap filling were learned from the test rows, changing half
    # of the test rows would change the predictions for the other half.
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(200, 3))
    y_train = x_train[:, 0]
    x_test = rng.normal(size=(20, 3))
    changed = x_test.copy()
    changed[10:] = 1000
    for name in model.MODELS:
        first = model.fit_predict(name, x_train, y_train, x_test)[:10]
        second = model.fit_predict(name, x_train, y_train, changed)[:10]
        assert np.allclose(first, second, atol=1e-5), name


def test_baselines_and_fold_spread(make_raw):
    rows = []
    for number in range(4):                                   # 4 drives, throughput 10 then 30 Mbps in each
        rows += seconds(f"d{number}", 0, 10, DL_bitrate=10000) + seconds(f"d{number}", 10, 10, DL_bitrate=30000)
    windows = model.make_windows(clean(make_raw(rows)))
    folds = model.drive_folds(windows, n_splits=2)
    baselines = model.baseline_scores(windows, folds)
    # persistence predicts 10 Mbps for every 30 Mbps window: an error of 20 Mbps
    assert baselines.loc["Persistence (previous window)", "mae_mbps"] == 20
    assert baselines.loc["Median of training drives", "mae_mbps"] == 10
    perfect = windows["dl_mbps"].to_numpy()
    assert model.fold_spread(windows, perfect, folds) == [1.0, 1.0]
