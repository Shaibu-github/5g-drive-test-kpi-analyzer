"""Step 3 - KPI statistics: coverage, quality and throughput, per technology.

Everything here is a small function that takes the clean table and returns a
small table, so each number in the report can be traced to one function.
"""
import numpy as np
import pandas as pd

from .clean import downloading

TECHS = ["5G", "LTE", "3G"]

# Legend classes, like the legend of a drive-test map: (label, lower edge, colour).
# A value belongs to a class when: lower edge <= value < next lower edge.
RSRP_LEGEND = [
    ("Poor (below -100 dBm)", -141, "#d03b3b"),
    ("Fair (-100 to -90)", -100, "#ec835a"),
    ("Good (-90 to -80)", -90, "#fab219"),
    ("Excellent (-80 or better)", -80, "#0ca30c"),
]
THROUGHPUT_LEGEND = [
    ("Below 2 Mbps", 0, "#86b6ef"),
    ("2 to 5", 2, "#5598e7"),
    ("5 to 20", 5, "#2a78d6"),
    ("20 to 50", 20, "#1c5cab"),
    ("50 Mbps or more", 50, "#0d366b"),
]


def legend_class(values, legend):
    """Give every value the label of its legend class."""
    edges = [low for _, low, _ in legend] + [np.inf]
    labels = [label for label, _, _ in legend]
    return pd.cut(values, bins=edges, labels=labels, right=False)


def overview(df):
    """Headline numbers that describe the whole measurement campaign.

    Cells and sites are counted only on rows whose cell identity fits the
    technology (see cleaning rule 7), so logger glitches do not add fake sites.
    """
    trusted = df[df["identity_ok"]]
    return {
        "drives": int(df["drive"].nunique()),
        "samples": int(len(df)),
        "hours": round(float(df.groupby("drive")["elapsed_s"].max().sum()) / 3600, 1),
        "route_km": round(float(df["step_m"].sum()) / 1000, 1),
        "cells": int(trusted["cell"].nunique()),
        "sites": int(trusted["site"].nunique()),
        "rows_with_identity_mismatch": int((~df["identity_ok"]).sum()),
        "first_day": str(df["time"].min().date()),
        "last_day": str(df["time"].max().date()),
    }


def tech_share(df):
    """Percent of samples served by each technology."""
    share = df["tech"].value_counts(normalize=True) * 100
    return share.reindex(TECHS).round(1)


def radio_summary(df):
    """Median and 10th percentile of each radio KPI, per technology.

    The 10th percentile is the "cell edge" view: 90% of samples are better.
    3G has only a signal level (RSCP, stored in the rsrp column).
    """
    rows = {}
    for tech in TECHS:
        part = df[df["tech"] == tech]
        row = {"samples": len(part)}
        for kpi in ["rsrp", "rsrq", "snr", "cqi"]:
            row[kpi + "_median"] = part[kpi].median()
            row[kpi + "_p10"] = part[kpi].quantile(0.10)
        rows[tech] = row
    return pd.DataFrame(rows).T.round(1)


def throughput_summary(df):
    """Downlink throughput statistics per technology, download samples only."""
    active = downloading(df)
    rows = {}
    for tech in TECHS:
        mbps = active.loc[active["tech"] == tech, "dl_mbps"]
        rows[tech] = {
            "samples": len(mbps),
            "mean_mbps": mbps.mean(),
            "p10_mbps": mbps.quantile(0.10),
            "median_mbps": mbps.median(),
            "p90_mbps": mbps.quantile(0.90),
            "max_mbps": mbps.max(),
            "pct_below_1mbps": (mbps < 1).mean() * 100,
        }
    return pd.DataFrame(rows).T.round(1)


def coverage_classes(df, techs=("5G", "LTE")):
    """Percent of samples in each RSRP class, per technology."""
    part = df[df["tech"].isin(techs)]
    classes = legend_class(part["rsrp"], RSRP_LEGEND)
    table = pd.crosstab(part["tech"], classes, normalize="index") * 100
    return table.reindex(list(techs)).round(1)


def cdf(values):
    """Return x (sorted values) and y (percent of samples <= x) for a CDF plot."""
    x = np.sort(pd.Series(values).dropna().to_numpy())
    y = np.arange(1, len(x) + 1) / len(x) * 100
    return x, y


def correlation_with_throughput(df, kpis=("rsrp", "rsrq", "snr", "cqi"), techs=("5G", "LTE")):
    """Spearman rank correlation between each KPI and downlink throughput.

    Spearman measures "when the KPI goes up, does throughput go up?" without
    assuming a straight line. +1 = always, 0 = no relation.
    """
    active = downloading(df)
    table = {}
    for tech in techs:
        part = active[active["tech"] == tech]
        table[tech] = {kpi: part[kpi].corr(part["dl_mbps"], method="spearman") for kpi in kpis}
    return pd.DataFrame(table).round(2)


def throughput_by_level(df, column, edges=None, techs=("5G", "LTE"), min_samples=30):
    """Median throughput for each level of a KPI (for example each CQI value).

    With `edges` the KPI is first cut into bins (for SNR or RSRP). Levels with
    fewer than `min_samples` samples are dropped: their median is not reliable.
    """
    active = downloading(df)
    active = active[active["tech"].isin(techs)]
    level = active[column] if edges is None else pd.cut(active[column], bins=edges)
    grouped = active.groupby(["tech", level], observed=True)["dl_mbps"]
    table = grouped.agg(samples="size", median_mbps="median").reset_index()
    return table[table["samples"] >= min_samples].round(2)
