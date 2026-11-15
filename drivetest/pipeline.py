"""Run the whole analysis from raw logs to figures and result tables.

    python -m drivetest

The numbers in the README come from the tables this function writes.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # draw to files; no window needed
import matplotlib.pyplot as plt
import pandas as pd

from . import clean, data, handover, kpi, maps, model, plots, report, tuning


def run(data_dir="data", figures_dir="figures", results_dir="results", site_dir="docs"):
    figures_dir, results_dir = Path(figures_dir), Path(results_dir)
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    def save_figure(fig, name):
        fig.savefig(figures_dir / name)
        plt.close(fig)

    def save_table(table, name):
        table.to_csv(results_dir / name)

    # --- 1 + 2: load and clean ------------------------------------------
    print("1/7  loading the logs")
    raw = data.load_logs(data.list_logs(data.download_dataset(data_dir)))
    print("2/7  cleaning")
    df = clean.clean(raw)
    save_table(clean.cleaning_report(raw, df), "cleaning_report.csv")

    # --- 3: KPI statistics ----------------------------------------------
    print("3/7  KPI statistics")
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
    print("4/7  cell changes and ping-pong borders")
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

    # --- 4b: does ping-pong belong to particular borders? ---------------------
    changes = tuning.radio_changes(events)
    not_short = tuning.without_short_round_trips(changes)       # the same test without the 2-second round trips
    rules = tuning.compare_rules(changes)
    borders = tuning.border_table(changes)
    main = pd.concat({
        "All round trips": tuning.main_advantages_with_intervals(changes),
        "Without round trips of 2 s or less": tuning.main_advantages_with_intervals(not_short),
    }, names=["round_trips_used", "rule"])
    by_drive = tuning.drive_by_drive(changes)
    save_table(rules.set_index(["rule", "setting"]).round(1), "ping_pong_rules.csv")
    save_table(main.round(1), "ping_pong_main.csv")
    save_table(by_drive.round(1), "ping_pong_by_drive.csv")
    save_table(borders.round({"lat": 5, "lon": 5}), "ping_pong_borders.csv")
    save_figure(plots.ping_pong_rules(rules, borders), "ping_pong_rules.png")

    # --- 5: maps ----------------------------------------------------------
    print("5/7  maps")
    bins = maps.road_bins(df, events)
    spots = maps.weak_spots(bins)
    save_table(spots.set_index("problem"), "weak_spots.csv")
    save_figure(plots.coverage_maps(bins, spots), "map_coverage.png")
    save_figure(plots.mobility_maps(bins), "map_mobility.png")
    maps.interactive_map(bins, events, spots, figures_dir / "interactive_map.html")

    # --- 6: throughput prediction -----------------------------------------
    print("6/7  throughput prediction (trains 60 small models, about 15 s)")
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
            # a chain A-B-A-B-A is three round trips but one episode
            "ping_pong_episodes": int((events["ping_pong_out"] & ~events["ping_pong_return"]).sum()),
            "changes_in_a_round_trip": int(events["ping_pong"].sum()),
            "changes_in_a_round_trip_pct": round(events["ping_pong"].mean() * 100, 1),
            "changes_next_to_identity_mismatch": doubtful,
            "isolated": len(alone),
            "isolated_used_for_dip": int(dips.loc["All changes", "events"]),
            "median_drop_pct": float(dips.loc["All changes", "median_drop_pct"]),
            "control_median_drop_pct": float(control.loc["All changes", "median_drop_pct"]),
        },
        "tuning": {
            "lte_5g_changes": len(changes),
            "round_trips": int(changes["ping_pong_out"].sum()),
            "round_trips_of_2_s_or_less": int((changes["ping_pong_out"] & (changes["next_stay_s"] <= 2)).sum()),
            # fewer than the difference: in a chain, a longer round trip can start with the way back of a short one
            "round_trips_left_without_them": int(not_short["ping_pong_out"].sum()),
            "normal_changes": int((~changes["ping_pong"]).sum()),
            "borders": len(borders),
            "borders_with_round_trip": int((borders["round_trips"] > 0).sum()),
            "top_8_borders_pct_of_round_trips": round(float(
                borders["round_trips"].head(8).sum() / borders["round_trips"].sum() * 100), 1),
            "top_8_borders_pct_of_changes": round(float(
                borders["changes"].head(8).sum() / borders["changes"].sum() * 100), 1),
            "margin_rule_3_db": _rule_numbers(rules, "gain below 3 dB"),
            "border_rule": _rule_numbers(rules, "1+ round trips in other drives"),
            "road_only": _rule_numbers(rules, "crossed in another drive"),
            "drives_scored": len(by_drive),
            "drives_with_border_rule_ahead": int((by_drive["border_advantage"] > by_drive["margin_advantage"]).sum()),
            "main": {used: part.droplevel(0).round(1).to_dict("index") for used, part in main.groupby(level=0)},
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

    # --- 7: the dashboard -----------------------------------------------------
    print("7/7  dashboard (one page for all drives, one page per drive)")
    report.write_site(site_dir, df, events, summary, bins, spots, rules, main, borders, scores, baselines)
    print(f"done: figures in {figures_dir}/, tables in {results_dir}/, dashboard in {site_dir}/index.html")
    return summary


def _rule_numbers(rules, setting):
    """The scores of one rule setting for summary.json: the counts, and the percentages rounded."""
    row = rules.set_index("setting").loc[setting]
    numbers = {name: int(row[name]) for name in ["round_trips_caught", "normal_changes_affected"]}
    for name in ["round_trips_caught_pct", "normal_changes_affected_pct", "advantage_points"]:
        numbers[name] = round(float(row[name]), 1)
    return numbers
