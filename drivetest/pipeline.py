"""Run the whole analysis from raw logs to figures and result tables.

    python -m drivetest

The numbers in the README come from the tables this function writes.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # draw to files; no window needed
import matplotlib.pyplot as plt

from . import clean, data, handover, kpi, maps, model, plots


def run(data_dir="data", figures_dir="figures", results_dir="results"):
    figures_dir, results_dir = Path(figures_dir), Path(results_dir)
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    def save_figure(fig, name):
        fig.savefig(figures_dir / name)
        plt.close(fig)

    def save_table(table, name):
        table.to_csv(results_dir / name)

    # --- 1 + 2: load and clean ------------------------------------------
    print("1/6  loading the logs")
    raw = data.load_logs(data.list_logs(data.download_dataset(data_dir)))
    print("2/6  cleaning")
    df = clean.clean(raw)
    save_table(clean.cleaning_report(raw, df), "cleaning_report.csv")

    # --- 3: KPI statistics ----------------------------------------------
    print("3/6  KPI statistics")
    uplink = kpi.correlation_with_throughput(df, kpis=["ul_mbps"])
    save_table(kpi.radio_summary(df), "radio_summary.csv")
    save_table(kpi.throughput_summary(df), "throughput_summary.csv")
    save_table(kpi.coverage_classes(df), "coverage_classes.csv")
    save_table(kpi.correlation_with_throughput(df), "correlation_with_throughput.csv")
    save_figure(plots.radio_cdfs(df), "radio_cdfs.png")
    save_figure(plots.throughput_cdf(df), "throughput_cdf.png")
    save_figure(plots.coverage_bars(df), "coverage_classes.png")
    save_figure(plots.link_adaptation(df), "link_adaptation.png")

    # --- 4: cell changes --------------------------------------------------
    print("4/6  cell changes")
    events = handover.mark_ping_pong(handover.find_cell_changes(df))
    alone = handover.isolated(events)
    dips = handover.dip_summary(handover.dip_per_event(handover.throughput_window(df, alone)))
    control = handover.dip_summary(handover.dip_per_event(handover.throughput_window(df, handover.random_moments(df))))
    rate = handover.rate_per_drive(df, events)
    save_table(handover.summarize_by_kind(events), "cell_change_kinds.csv")
    save_table(handover.ping_pong_sensitivity(events), "ping_pong_sensitivity.csv")
    save_table(rate, "cell_change_rate.csv")
    save_table(dips, "cell_change_dip.csv")
    save_figure(plots.change_profile(df, events), "cell_change_profile.png")
    save_figure(plots.change_profile_by_kind(df, events), "cell_change_by_kind.png")

    # --- 5: maps ----------------------------------------------------------
    print("5/6  maps")
    bins = maps.road_bins(df, events)
    spots = maps.weak_spots(bins)
    save_table(spots.set_index("problem"), "weak_spots.csv")
    save_figure(plots.coverage_maps(bins, spots), "map_coverage.png")
    save_figure(plots.mobility_maps(bins), "map_mobility.png")
    maps.interactive_map(bins, events, spots, figures_dir / "interactive_map.html")

    # --- 6: throughput prediction -----------------------------------------
    print("6/6  throughput prediction (trains 60 small models, about 15 s)")
    windows = model.make_windows(df)
    folds = model.drive_folds(windows)
    baselines = model.baseline_scores(windows, folds)
    scores, predictions = model.compare_models(windows, folds)
    best_features = "Radio KPIs + location"
    importance = model.feature_importance(windows, model.FEATURE_SETS[best_features])
    leak = model.random_split_score(windows, model.FEATURE_SETS["Radio KPIs"])
    save_table(baselines.round(3), "model_baselines.csv")
    save_table(scores.set_index(["features", "model"]).round(3), "model_scores.csv")
    save_table(importance.rename("r2_drop").to_frame(), "feature_importance.csv")
    save_figure(plots.model_scores(scores, baselines), "model_scores.png")
    save_figure(plots.model_fit(windows, predictions[(best_features, "Gradient boosting")], importance), "model_fit.png")

    # --- headline numbers used in the README --------------------------------
    honest = scores.set_index(["features", "model"]).loc[("Radio KPIs", "Gradient boosting"), "r2_log"]
    # how many cell changes sit on, or right after, a row with a mismatched identity?
    mismatch = df[~df["identity_ok"]]
    mismatch_rows = set(zip(mismatch["drive"], mismatch["sample"]))
    doubtful = sum((e.drive, e.sample) in mismatch_rows or (e.drive, e.sample - 1) in mismatch_rows
                   for e in events.itertuples())
    summary = {
        "overview": kpi.overview(df),
        "tech_share_pct": kpi.tech_share(df).to_dict(),
        "uplink_rank_correlation_with_downlink": uplink.loc["ul_mbps"].to_dict(),
        "cell_changes": {
            "total": len(events),
            "per_minute": round(len(events) / rate["minutes"].sum(), 2),
            "per_km": round(len(events) / rate["km"].sum(), 2),
            "ping_pong_round_trips": int(events["ping_pong_return"].sum()),
            "changes_in_a_round_trip": int(events["ping_pong"].sum()),
            "changes_in_a_round_trip_pct": round(events["ping_pong"].mean() * 100, 1),
            "changes_next_to_identity_mismatch": doubtful,
            "isolated": len(alone),
            "isolated_used_for_dip": int(dips.loc["All changes", "events"]),
            "median_drop_pct": float(dips.loc["All changes", "median_drop_pct"]),
            "control_median_drop_pct": float(control.loc["All changes", "median_drop_pct"]),
        },
        "maps": {"road_bins": len(bins), "weak_spots": spots["problem"].value_counts().to_dict()},
        "model": {
            "windows": len(windows),
            "drives_with_windows": int(windows["drive"].nunique()),
            "r2_radio_kpis_drive_split": round(float(honest), 3),
            "r2_radio_kpis_random_split": round(leak["r2_log"], 3),
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done: figures in {figures_dir}/, tables in {results_dir}/")
    return summary
