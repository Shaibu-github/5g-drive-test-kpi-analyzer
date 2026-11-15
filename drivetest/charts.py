"""Interactive charts for the HTML report and the dashboard (Plotly).

Same data and same colours as the figures in plots.py. The difference: these
charts answer to the mouse. Point at a chart and it shows the values there.
Every function returns a Plotly figure; report.py puts the figures on a page.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import handover, kpi
from .clean import downloading
from .kpi import RSRP_LEGEND
from .style import AXIS, GRID, INK, INK_SOFT, MODEL_COLOURS, MUTED, RULE_COLOURS, SURFACE, TECH_COLOURS
from .tuning import cell_name

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style(fig, height, legend=True):
    """Give a figure the shared look: thin grey grid, grey text, legend on top."""
    fig.update_layout(
        template="none", height=height, showlegend=legend,
        margin=dict(l=60, r=18, t=40 if legend else 14, b=48),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SOFT),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=INK_SOFT)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
    )
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", tickfont=dict(color=MUTED),
                title_font=dict(color=INK_SOFT, size=12), title_standoff=8)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ----------------------------------------------------------------------------
# Coverage and quality
# ----------------------------------------------------------------------------

def coverage_classes(df):
    """Share of samples in each RSRP class: one bar per technology, split into classes."""
    techs = [tech for tech in ["5G", "LTE"] if (df["tech"] == tech).any()]
    labels = [label for label, _, _ in RSRP_LEGEND]
    table = kpi.coverage_classes(df, techs).reindex(columns=labels, fill_value=0).fillna(0)
    fig = go.Figure()
    for label, _, colour in RSRP_LEGEND:
        share = table[label]
        fig.add_trace(go.Bar(
            y=table.index, x=share, orientation="h", name=label, width=0.5,
            marker=dict(color=colour, line=dict(color=SURFACE, width=2)),       # 2 px gap between the parts
            text=[f"{value:.0f}%" if value >= 8 else "" for value in share],    # only where the label fits
            textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color=INK if colour == "#fab219" else "#ffffff", size=12),
            hovertemplate="<b>%{x:.1f}%</b> of the %{y} samples<extra>" + label + "</extra>",
        ))
    _style(fig, height=110 + 45 * len(techs))
    fig.update_layout(barmode="stack", legend=dict(traceorder="normal"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="Share of samples")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_SOFT, size=13))
    return fig, table.round(1)


def distribution(df, column, label, unit, techs=("5G", "LTE"), log_x=False):
    """CDF of one KPI per technology: what share of samples is at or below a value?

    All technologies are read off the same x values, so one pointer position
    compares them directly.
    """
    part = downloading(df) if column == "dl_mbps" else df
    values = {tech: np.sort(part.loc[part["tech"] == tech, column].dropna().to_numpy()) for tech in techs}
    values = {tech: v for tech, v in values.items() if len(v)}
    fig = go.Figure()
    if not values:
        return _style(fig, 300), None
    low = min(v[0] for v in values.values())
    high = max(v[-1] for v in values.values())
    grid = np.geomspace(0.1, max(high, 1), 160) if log_x else np.linspace(low, high, 160)
    for tech, sorted_values in values.items():
        at_or_below = np.searchsorted(sorted_values, grid, side="right") / len(sorted_values) * 100
        fig.add_trace(go.Scatter(
            x=grid, y=at_or_below, mode="lines", name=tech, line=dict(color=TECH_COLOURS[tech], width=2),
            hovertemplate="<b>%{y:.0f}%</b> of samples<extra>" + tech + "</extra>",
        ))
    _style(fig, height=300)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text=f"{label} ({unit})", hoverformat=".1f" if log_x else ".0f")
    if log_x:
        fig.update_xaxes(type="log", tickvals=[0.1, 1, 10, 100], ticktext=["0.1", "1", "10", "100"])
    fig.update_yaxes(title_text="% of samples at or below", range=[0, 100])
    # the numbers behind the curves: the usual percentiles
    table = {tech: {f"P{q}": np.percentile(v, q) for q in (10, 25, 50, 75, 90)} for tech, v in values.items()}
    return fig, pd.DataFrame(table).T.round(1)


# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

def timeline(df, events):
    """RSRP, SNR and throughput along one drive, coloured by serving technology.

    This is the classic drive-test view. Dots on the RSRP line mark the
    serving-cell changes; black dots are the start of a ping-pong round trip.
    The three panels share one time axis, so one pointer position reads all three.
    """
    minutes = (df["elapsed_s"] / 60).astype("float32")
    level = "RSRP (dBm)" if (df["tech"] != "3G").all() else "RSRP, on 3G RSCP (dBm)"
    panels = [("rsrp", level, "dBm", "y"), ("snr", "SNR (dB)", "dB", "y2"), ("dl_mbps", "Downlink (Mbps)", "Mbps", "y3")]
    fig = go.Figure()
    for column, label, unit, y_axis in panels:
        for tech in kpi.TECHS:
            on_tech = df["tech"] == tech
            if not on_tech.any():
                continue
            fig.add_trace(go.Scatter(
                x=minutes, y=df[column].where(on_tech).astype("float32"),    # blank while another technology serves
                xaxis="x", yaxis=y_axis, mode="lines", name=tech, legendgroup=tech, showlegend=y_axis == "y",
                line=dict(color=TECH_COLOURS[tech], width=1.5), connectgaps=False,
                hovertemplate=f"<b>%{{y:.1f}} {unit}</b><extra>{tech}</extra>",
            ))

    marks = [("Cell change", events[~events["ping_pong_out"]], "#ffffff"),
             ("Start of a ping-pong round trip", events[events["ping_pong_out"]], INK)]
    for name, part, fill in marks:
        if part.empty:
            continue
        text = [f"{e.kind}: {cell_name(e.from_cell, e.from_tech)} to {cell_name(e.to_cell, e.to_tech)}"
                for e in part.itertuples()]
        fig.add_trace(go.Scatter(
            x=part["elapsed_s"] / 60, y=part["rsrp_after"].fillna(part["rsrp_before"]), xaxis="x", yaxis="y",
            mode="markers", name=name, marker=dict(size=7, color=fill, line=dict(color=INK, width=1)),
            text=text, hovertemplate="%{text}<extra></extra>",
        ))

    _style(fig, height=560)
    # hoverdistance: only show values within 2 pixels of the pointer. Otherwise the readout
    # would also list the last value of a technology that stopped serving a while ago.
    fig.update_layout(grid=dict(rows=3, columns=1, ygap=0.12), hovermode="x unified", hoversubplots="axis",
                      hoverdistance=2,
                      yaxis=dict(title_text=panels[0][1]), yaxis2=dict(title_text=panels[1][1]),
                      yaxis3=dict(title_text=panels[2][1], rangemode="tozero"))
    _style(fig, height=560)             # again: the axes of panels 2 and 3 exist only now
    fig.update_xaxes(title_text="Minutes since the start of the drive", hoverformat=".1f")
    return fig


def change_profile(df, events, window=10):
    """Median throughput in the seconds around an isolated cell change."""
    alone = handover.isolated(events, window)
    around = handover.throughput_window(df, alone, window)
    profile = around.groupby("offset")["dl_mbps"].median()
    fig = go.Figure(go.Scatter(
        x=profile.index, y=profile.to_numpy(), mode="lines+markers",
        line=dict(color=TECH_COLOURS["5G"], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
        hovertemplate="<b>%{y:.1f} Mbps</b><extra></extra>",
    ))
    _style(fig, height=300, legend=False)
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.add_annotation(x=0, y=1, yref="paper", text="first sample on the new cell", showarrow=False,
                       xanchor="left", xshift=6, yanchor="top", font=dict(color=INK_SOFT, size=11))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Seconds relative to the cell change", dtick=2, hoverformat="+d", ticksuffix=" s")
    fig.update_yaxes(title_text="Median downlink (Mbps)", rangemode="tozero")
    table = profile.rename("median_mbps").to_frame().round(1)
    table.index.name = "seconds_from_change"
    return fig, table, len(alone)


def ping_pong_rules(rules):
    """How well each rule separates ping-pong from normal cell changes."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(color=AXIS, width=1, dash="dash")))
    for rule in ["Border", "Margin"]:
        part = rules[rules["rule"] == rule]
        # "5+ round trips in other drives" -> "5+",  "gain below 3 dB" -> "3 dB"
        short = [s.split(" round")[0] if rule == "Border" else s.replace("gain below ", "") for s in part["setting"]]
        shown = [s if rule == "Border" or s in ("0 dB", "3 dB", "8 dB") else "" for s in short]
        fig.add_trace(go.Scatter(
            x=part["normal_changes_affected_pct"], y=part["round_trips_caught_pct"],
            mode="lines+markers+text", name=f"{rule} rule", text=shown, textposition="top left",
            textfont=dict(color=INK_SOFT, size=11), customdata=part["setting"],
            line=dict(color=RULE_COLOURS[rule], width=2),
            marker=dict(size=10, line=dict(color=SURFACE, width=2)),
            hovertemplate=("<b>%{customdata}</b><br>acts on %{y:.0f}% of the round trips"
                           "<br>and on %{x:.0f}% of the normal changes<extra>" + rule + " rule</extra>"),
        ))
    road = rules[rules["rule"] == "Road only"]
    fig.add_trace(go.Scatter(
        x=road["normal_changes_affected_pct"], y=road["round_trips_caught_pct"], mode="markers",
        name="Road only (control)", customdata=road["setting"],
        marker=dict(size=10, symbol="diamond", color=MUTED, line=dict(color=SURFACE, width=2)),
        hovertemplate=("<b>%{customdata}</b><br>acts on %{y:.0f}% of the round trips"
                       "<br>and on %{x:.0f}% of the normal changes<extra>Road only</extra>"),
    ))
    _style(fig, height=420)
    fig.add_annotation(x=76, y=58, text="a rule that cannot tell them apart", showarrow=False, textangle=-35,
                       font=dict(color=MUTED, size=11))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Normal changes the rule also acts on (%)", range=[0, 100])
    fig.update_yaxes(title_text="Ping-pong round trips the rule acts on (%)", range=[0, 100])
    return fig


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def model_scores(scores, baselines):
    """R2 of every model for every set of inputs, with the persistence baseline."""
    feature_sets = list(dict.fromkeys(scores["features"]))
    fig = go.Figure()
    for model, colour in MODEL_COLOURS.items():
        part = scores[scores["model"] == model].set_index("features").loc[feature_sets]
        fig.add_trace(go.Bar(
            y=feature_sets, x=part["r2_log"], orientation="h", name=model,
            marker=dict(color=colour, line=dict(color=SURFACE, width=2)),
            text=[f"{value:.2f}" for value in part["r2_log"]], textposition="outside",
            textfont=dict(color=INK_SOFT, size=11), cliponaxis=False,
            hovertemplate="<b>R² %{x:.2f}</b><br>%{y}<extra>" + model + "</extra>",
        ))
    persistence = baselines.loc["Persistence (previous window)", "r2_log"]
    _style(fig, height=360)
    fig.add_vline(x=persistence, line=dict(color=INK_SOFT, width=1))
    fig.add_annotation(x=persistence, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"persistence baseline {persistence:.2f}", font=dict(color=INK_SOFT, size=11))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12, barcornerradius=4,
                      margin=dict(l=210), legend=dict(y=1.1, traceorder="normal"))
    fig.update_xaxes(title_text="R² on log throughput, tested on drives the model never saw", range=[0, 0.8])
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_SOFT))
    return fig
