"""All figures of the report. Every function takes tables and returns a Figure.

One visual style for the whole project: the same colour always means the same
technology, grids are thin and grey, and text is dark grey - never coloured.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from . import handover, kpi
from .clean import downloading
from .kpi import RSRP_LEGEND, THROUGHPUT_LEGEND, legend_class
from .style import AXIS, GRID, INK, INK_SOFT, MODEL_COLOURS, MUTED, RULE_COLOURS, SURFACE, TECH_COLOURS

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "font.size": 10, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _headline(fig, title, note=None):
    """Figure title on the left, with an optional one-line note under it."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    if note:
        fig.text(0.01, 0.925, note, ha="left", va="top", fontsize=9.5, color=INK_SOFT)


# ----------------------------------------------------------------------------
# KPI statistics
# ----------------------------------------------------------------------------

def radio_cdfs(df):
    """CDF of RSRP, RSRQ, SNR and CQI for 5G and LTE."""
    panels = [("rsrp", "RSRP (dBm)"), ("rsrq", "RSRQ (dB)"), ("snr", "SNR (dB)"), ("cqi", "CQI")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    for ax, (column, label) in zip(axes.flat, panels):
        for tech in ["5G", "LTE"]:
            x, y = kpi.cdf(df.loc[df["tech"] == tech, column])
            ax.step(x, y, where="post", color=TECH_COLOURS[tech], label=tech)
        ax.set_xlabel(label)
        ax.set_ylabel("% of samples at or below")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))     # whole dB on the axis
    axes[0, 0].legend(loc="upper left")
    _headline(fig, "Radio KPI distributions: 5G (NSA) vs LTE",
              "Read a CDF like this: a curve further to the right means better values.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def throughput_cdf(df):
    """CDF of downlink throughput per technology, on a log axis."""
    active = downloading(df)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for tech in kpi.TECHS:
        mbps = active.loc[active["tech"] == tech, "dl_mbps"]
        x, y = kpi.cdf(mbps.clip(lower=0.1))          # log axis cannot show 0
        ax.step(x, y, where="post", color=TECH_COLOURS[tech], label=f"{tech}  (median {mbps.median():.1f} Mbps)")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 600)
    ax.set_xticks([0.1, 1, 10, 100, 500], ["≤ 0.1", "1", "10", "100", "500"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Downlink throughput (Mbps, log scale)")
    ax.set_ylabel("% of samples at or below")
    ax.legend(loc="upper left")
    _headline(fig, "Downlink throughput by technology", "Download samples only. Every step right on the axis is 10 times faster.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def coverage_bars(df):
    """Share of samples in each RSRP class, one stacked bar per technology."""
    table = kpi.coverage_classes(df)
    colours = [colour for _, _, colour in RSRP_LEGEND]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    left = np.zeros(len(table))
    for label, colour in zip(table.columns, colours):
        share = table[label].to_numpy()
        ax.barh(table.index, share, left=left, height=0.5, color=colour, edgecolor=SURFACE, linewidth=2, label=label)
        for row, (start, width) in enumerate(zip(left, share)):
            if width >= 6:                             # only label segments that have room
                text_colour = INK if colour == "#fab219" else "#ffffff"
                ax.text(start + width / 2, row, f"{width:.0f}%", ha="center", va="center", fontsize=9, color=text_colour)
        left += share
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of samples")
    ax.invert_yaxis()
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4)
    _headline(fig, "RSRP coverage classes")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def link_adaptation(df):
    """Median throughput for each CQI value and each SNR range."""
    snr_edges = [-20, -5, 0, 5, 10, 15, 20, 40]
    by_cqi = kpi.throughput_by_level(df, "cqi")
    by_snr = kpi.throughput_by_level(df, "snr", edges=snr_edges)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for tech in ["5G", "LTE"]:
        part = by_cqi[by_cqi["tech"] == tech]
        left.plot(part["cqi"], part["median_mbps"], marker="o", markersize=5, color=TECH_COLOURS[tech], label=tech)
        part = by_snr[by_snr["tech"] == tech]
        right.plot(range(len(part)), part["median_mbps"], marker="o", markersize=5, color=TECH_COLOURS[tech], label=tech)
    left.set_xlabel("CQI reported by the phone")
    left.set_ylabel("Median downlink throughput (Mbps)")
    left.set_xticks(range(3, 16, 2))
    left.set_ylim(0)
    left.legend(loc="upper left")
    labels = ["< -5", "-5 to 0", "0 to 5", "5 to 10", "10 to 15", "15 to 20", "> 20"]
    right.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    right.set_xlabel("SNR (dB)")
    _headline(fig, "Does better channel quality give more throughput?",
              "LTE: yes, step by step. 5G rows: no steady rise - the logged CQI and SNR seem to describe the LTE anchor cell.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

def change_profile(df, events, window=10):
    """Median throughput, SNR and CQI in the seconds around a cell change."""
    alone = handover.isolated(events, window)
    around = handover.samples_around(df, alone, window)
    mbps = around[around["state"] == "D"]

    panels = [
        (mbps.groupby("offset")["dl_mbps"].median(), "Throughput (Mbps)"),
        (around.groupby("offset")["snr"].median(), "SNR (dB)"),
        (around.groupby("offset")["cqi"].median(), "CQI"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.8), sharex=True)
    for ax, (series, label) in zip(axes, panels):
        ax.axvline(0, color=AXIS, linewidth=1)                      # the moment of the change
        ax.plot(series.index, series.to_numpy(), marker="o", markersize=4.5, color=TECH_COLOURS["5G"])
        ax.set_ylabel(label)
    axes[0].set_ylim(0)
    axes[0].annotate("first sample on the new cell", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(4, 2), textcoords="offset points", fontsize=9, color=INK_SOFT, va="bottom")
    axes[-1].set_xlabel("Seconds relative to the cell change (about one sample per second)")
    axes[-1].set_xticks(range(-window, window + 1, 2))
    _headline(fig, "What a serving-cell change looks like",
              f"Median over {len(alone)} isolated changes (no other change within {window} s).")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def change_profile_by_kind(df, events, window=10):
    """Median throughput around the change, one small panel per kind of change."""
    kinds = ["5G intra-site", "5G inter-site", "5G to LTE", "LTE intra-site", "LTE inter-site", "LTE to 5G"]
    alone = handover.isolated(events, window)
    mbps = handover.throughput_window(df, alone, window)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, kinds):
        part = mbps[mbps["kind"] == kind]
        profile = part.groupby("offset")["dl_mbps"].median()
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.plot(profile.index, profile.to_numpy(), color=TECH_COLOURS["5G"])
        ax.set_title(f"{kind}  (n = {part['event'].nunique()})", fontsize=10)
        ax.set_ylim(0, 32)
    for ax in axes[-1]:
        ax.set_xlabel("Seconds from the change")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median Mbps")
    _headline(fig, "Throughput around the change, by kind of change",
              "Isolated changes only. The clear, lasting effects are the moves between 5G cells and LTE cells.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def ping_pong_rules(rules, borders, top=10):
    """Left: how well each rule separates ping-pong from normal changes.
    Right: the borders with the most round trips."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})

    left.plot([0, 100], [0, 100], color=AXIS, linewidth=1, linestyle="--")
    left.text(66, 62, "a rule that cannot tell them apart", rotation=45, rotation_mode="anchor",
              ha="center", va="top", fontsize=8.5, color=MUTED)
    for rule in ["Border", "Margin"]:
        part = rules[rules["rule"] == rule]
        x, y = part["normal_changes_affected_pct"], part["round_trips_caught_pct"]
        left.plot(x, y, marker="o", markersize=6, color=RULE_COLOURS[rule], label=f"{rule} rule")
        for setting, px, py in zip(part["setting"], x, y):
            # "5+ round trips in other drives" -> "5+",  "gain below 3 dB" -> "3 dB"
            short = setting.split(" round")[0] if rule == "Border" else setting.replace("gain below ", "")
            if rule == "Margin" and short not in ("0 dB", "3 dB", "8 dB"):
                continue                               # label a few points only, the rest follow the line
            offset = (-8, -4) if short == "0 dB" else (-7, 5)
            left.annotate(short, (px, py), xytext=offset, textcoords="offset points", ha="right",
                          fontsize=8.5, color=INK_SOFT)
    road = rules[rules["rule"] == "Road only"].iloc[0]
    left.plot(road["normal_changes_affected_pct"], road["round_trips_caught_pct"], marker="D", markersize=6,
              linestyle="none", color=MUTED, label="Road only (control)")
    left.set_xlim(0, 100)
    left.set_ylim(0, 100)
    left.set_aspect("equal")
    left.set_xlabel("Normal changes the rule also acts on (%)")
    left.set_ylabel("Ping-pong round trips the rule acts on (%)")
    left.set_title("Up and to the left is better")
    left.legend(loc="upper left")

    worst = borders.head(top).iloc[::-1]               # biggest at the top of the chart
    right.barh(worst.index, worst["round_trips"], height=0.55, color=RULE_COLOURS["Border"])
    for row, (round_trips, drives) in enumerate(zip(worst["round_trips"], worst["drives_with_round_trip"])):
        right.text(round_trips + 0.3, row, f"in {drives} drive{'s' if drives != 1 else ''}", va="center",
                   fontsize=8.5, color=INK_SOFT)
    right.set_xlim(0, worst["round_trips"].max() * 1.22)
    right.xaxis.set_major_locator(MaxNLocator(integer=True))
    right.set_xlabel("Ping-pong round trips (all drives)")
    right.set_title(f"The {top} borders with the most round trips (site-cell | site-cell)")
    right.grid(axis="y", visible=False)
    right.tick_params(axis="y", labelsize=8.5)
    _headline(fig, "What separates a ping-pong from a normal cell change?",
              "Border rule: act at borders with N+ round trips in the other drives.  "
              "Margin rule: act when the RSRP gain is below N dB.\n"
              "Road only: act at every border that another drive also crossed (it knows the road, not the ping-pong).")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------

def _map_axis(ax, bins, title):
    """Shared look of a map panel: true proportions, no grid, light labels."""
    ax.set_aspect(1 / np.cos(np.radians(bins["bin_lat"].mean())))   # 1 km north = 1 km east on screen
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False)
    ax.locator_params(nbins=4)


def _class_map(ax, bins, column, legend, title, legend_title):
    classes = legend_class(bins[column], legend)
    missing = bins[bins[column].isna()]
    if len(missing):
        ax.scatter(missing["bin_lon"], missing["bin_lat"], s=11, color=GRID, label="3G only (no RSRP)")
    for label, _, colour in legend:
        part = bins[classes == label]
        ax.scatter(part["bin_lon"], part["bin_lat"], s=11, color=colour, label=label)
    ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=1.6, title_fontsize=9)
    _map_axis(ax, bins, title)


def coverage_maps(bins, spots):
    """Route coloured by RSRP class (with weak spots ringed) and by throughput."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
    _class_map(left, bins, "rsrp", RSRP_LEGEND, "RSRP (LTE and 5G samples)", "Median RSRP")
    left.scatter(spots["bin_lon"], spots["bin_lat"], s=60, facecolors="none", edgecolors=INK, linewidths=1.1, label="Weak spot")
    left.legend(title="Median RSRP", loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=1.0, title_fontsize=9)
    _class_map(right, bins, "dl_mbps", THROUGHPUT_LEGEND, "Downlink throughput", "Median Mbps")
    _headline(fig, "Drive routes in Cork, in 100 m road bins (all 16 drives combined)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def mobility_maps(bins):
    """Route coloured by serving technology, and where the cell changes happen."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
    for tech, colour in TECH_COLOURS.items():
        part = bins[bins["tech"] == tech]
        left.scatter(part["bin_lon"], part["bin_lat"], s=11, color=colour, label=tech)
    left.legend(title="Most common technology", loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=1.6, title_fontsize=9)
    _map_axis(left, bins, "Serving technology")

    right.scatter(bins["bin_lon"], bins["bin_lat"], s=6, color=GRID)        # the route, as context
    busy = bins[bins["cell_changes"] > 0]
    for label, low, high, size in [("1 to 2", 1, 2, 14), ("3 to 5", 3, 5, 40), ("6 or more", 6, 999, 90)]:
        part = busy[busy["cell_changes"].between(low, high)]
        right.scatter(part["bin_lon"], part["bin_lat"], s=size, color=TECH_COLOURS["5G"], alpha=0.75,
                      edgecolors=SURFACE, linewidths=0.6, label=label)
    right.legend(title="Cell changes in the bin", loc="upper left", bbox_to_anchor=(1.0, 1.0), title_fontsize=9)
    _map_axis(right, bins, "Where the serving cell changes")
    _headline(fig, "Technology and mobility along the routes")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def model_scores(results, baselines):
    """R2 of every model for every set of inputs, with the persistence baseline."""
    feature_sets = list(dict.fromkeys(results["features"]))
    models = list(dict.fromkeys(results["model"]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    height = 0.22
    for position, model in enumerate(models):
        part = results[results["model"] == model].set_index("features").loc[feature_sets]
        y = np.arange(len(feature_sets)) + (position - 1) * (height + 0.03)
        ax.barh(y, part["r2_log"], height=height, color=MODEL_COLOURS[model], label=model)
        for value, row in zip(part["r2_log"], y):
            # the small background box keeps the baseline line from striking through a label
            ax.text(value + 0.008, row, f"{value:.2f}", va="center", fontsize=8.5, color=INK_SOFT, zorder=3,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    persistence = baselines.loc["Persistence (previous window)", "r2_log"]
    ax.axvline(persistence, color=INK_SOFT, linewidth=1, zorder=1)
    ax.text(persistence, -0.62, f"persistence baseline {persistence:.2f}", fontsize=8.5, color=INK_SOFT, ha="center", va="bottom")
    ax.set_yticks(range(len(feature_sets)), feature_sets)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("R² on log throughput, tested on drives the model never saw")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95))    # free space: the top bars are short
    _headline(fig, "Location and history matter more than the choice of model",
              "Each row adds information. 10-second windows, 5-fold cross-validation grouped by drive.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def model_fit(windows, predicted, importance):
    """Predicted vs measured throughput, and the importance of each input."""
    names = {"is_5g": "Technology (5G or LTE)", "lat": "Latitude", "lon": "Longitude", "rsrp": "RSRP", "rsrq": "RSRQ",
             "snr": "SNR", "cqi": "CQI", "rssi": "RSSI", "nbr_rsrp": "Neighbour RSRP",
             "dominance": "Serving minus neighbour RSRP"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    actual = windows["dl_mbps"].clip(lower=0.1)
    left.scatter(actual, np.clip(predicted, 0.1, None), s=7, alpha=0.35, color=TECH_COLOURS["5G"], linewidths=0)
    left.plot([0.1, 500], [0.1, 500], color=INK_SOFT, linewidth=1)          # perfect prediction
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(0.1, 500)
    left.set_ylim(0.1, 500)
    ticks = [0.1, 1, 10, 100]
    left.set_xticks(ticks, ["0.1", "1", "10", "100"])
    left.set_yticks(ticks, ["0.1", "1", "10", "100"])
    left.set_xlabel("Measured throughput (Mbps)")
    left.set_ylabel("Predicted throughput (Mbps)")
    left.set_title("Predicted vs measured (points on the line are perfect)")

    order = importance.sort_values()
    right.barh([names.get(name, name) for name in order.index], order.to_numpy(), height=0.55, color=TECH_COLOURS["5G"])
    right.set_xlabel("Drop in R² when this input is shuffled")
    right.set_title("Which inputs the model needs")
    right.grid(axis="y", visible=False)
    _headline(fig, "Gradient boosting with radio KPIs + location")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
