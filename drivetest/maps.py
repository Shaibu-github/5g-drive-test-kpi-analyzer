"""Step 5 - maps: put the KPIs back on the road.

The same roads were driven many times. So instead of drawing 16 drives on top
of each other, every sample is placed in a 100 m x 100 m square ("road bin")
and each bin shows the typical value over all drives. A weak spot that shows
up in several drives is a real network problem, not one bad measurement.
"""
import numpy as np
import pandas as pd

from .clean import downloading
from .kpi import RSRP_LEGEND, THROUGHPUT_LEGEND, legend_class

METRES_PER_DEGREE = 111_320          # one degree of latitude, in metres
TECH_COLOURS = {"5G": "#2a78d6", "LTE": "#eb6834", "3G": "#1baf7a"}


def add_bin(table, size_m=100, reference_lat=51.9):
    """Add the bin a GPS point falls in (columns bin_lat, bin_lon = bin centre)."""
    lat_step = size_m / METRES_PER_DEGREE
    # a degree of longitude gets shorter away from the equator
    lon_step = size_m / (METRES_PER_DEGREE * np.cos(np.radians(reference_lat)))
    table = table.dropna(subset=["lat", "lon"]).copy()
    table["bin_lat"] = (np.floor(table["lat"] / lat_step) + 0.5) * lat_step
    table["bin_lon"] = (np.floor(table["lon"] / lon_step) + 0.5) * lon_step
    return table


def road_bins(df, events, size_m=100, min_samples=5):
    """One row per 100 m road bin, with the typical KPIs over all drives."""
    reference_lat = df["lat"].mean()
    samples = add_bin(df, size_m, reference_lat)
    keys = ["bin_lat", "bin_lon"]

    # RSRP exists only on LTE and 5G samples, so it gets its own counts
    with_rsrp = samples[samples["tech"].isin(["5G", "LTE"])].dropna(subset=["rsrp"])
    bins = pd.DataFrame({
        "samples": samples.groupby(keys).size(),
        "drives": samples.groupby(keys)["drive"].nunique(),
        "rsrp": with_rsrp.groupby(keys)["rsrp"].median(),
        "rsrp_samples": with_rsrp.groupby(keys).size(),
        "rsrp_drives": with_rsrp.groupby(keys)["drive"].nunique(),
        "dl_mbps": downloading(samples).groupby(keys)["dl_mbps"].median(),
        "tech": samples.groupby(keys)["tech"].agg(lambda t: t.mode().iloc[0]),   # most common
        "cell_changes": add_bin(events, size_m, reference_lat).groupby(keys).size(),
    })
    for column in ["rsrp_samples", "rsrp_drives", "cell_changes"]:
        bins[column] = bins[column].fillna(0).astype(int)
    bins = bins[bins["samples"] >= min_samples].copy()
    # a median of two or three RSRP samples is not a measurement of the bin
    bins.loc[bins["rsrp_samples"] < min_samples, "rsrp"] = np.nan
    return bins.reset_index()


def weak_spots(bins, min_drives=2, poor_rsrp=-100, good_rsrp=-90, low_mbps=2):
    """Find the bins an RF engineer would put on the action list.

    Two different problems, with two different fixes:
    * "Weak coverage"   - RSRP is poor. Fix: tilt, power, a new site.
    * "Low throughput with good coverage" - signal is fine but data is slow.
      Fix: look at interference, load, or missing 5G - not at coverage.
    Only bins with RSRP from at least `min_drives` different drives are used:
    one bad pass can be a truck in the way, not a network problem.
    """
    seen = bins[bins["rsrp_drives"] >= min_drives]
    coverage = seen[seen["rsrp"] < poor_rsrp].assign(problem="Weak coverage")
    quality = seen[(seen["rsrp"] >= good_rsrp) & (seen["dl_mbps"] < low_mbps)]
    quality = quality.assign(problem="Low throughput with good coverage")
    return pd.concat([coverage, quality], ignore_index=True)


def interactive_map(bins, events, spots, path):
    """Write an interactive HTML map (open it in a browser) with four layers."""
    import folium     # imported here so the rest of the project works without folium

    centre = [bins["bin_lat"].mean(), bins["bin_lon"].mean()]
    web_map = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap")

    def dot_layer(name, colours, texts, show):
        layer = folium.FeatureGroup(name=name, show=show)
        for lat, lon, colour, text in zip(bins["bin_lat"], bins["bin_lon"], colours, texts):
            if pd.notna(colour):
                folium.CircleMarker([lat, lon], radius=5, stroke=False, fill=True, fill_color=colour,
                                    fill_opacity=0.9, tooltip=text).add_to(layer)
        layer.add_to(web_map)

    rsrp_colour = dict((label, colour) for label, _, colour in RSRP_LEGEND)
    mbps_colour = dict((label, colour) for label, _, colour in THROUGHPUT_LEGEND)
    dot_layer("RSRP (LTE/5G)", legend_class(bins["rsrp"], RSRP_LEGEND).map(rsrp_colour),
              [f"RSRP {v:.0f} dBm" for v in bins["rsrp"]], show=True)
    dot_layer("Downlink throughput", legend_class(bins["dl_mbps"], THROUGHPUT_LEGEND).map(mbps_colour),
              [f"{v:.1f} Mbps" for v in bins["dl_mbps"]], show=False)
    dot_layer("Technology", bins["tech"].map(TECH_COLOURS), list(bins["tech"]), show=False)

    changes = folium.FeatureGroup(name="Cell changes", show=False)
    for event in events.dropna(subset=["lat", "lon"]).itertuples():
        text = f"{event.kind}: RSRP {event.rsrp_before:.0f} to {event.rsrp_after:.0f} dBm"
        folium.CircleMarker([event.lat, event.lon], radius=3, color="#0b0b0b", weight=1, fill=True,
                            fill_color="#ffffff", fill_opacity=1, tooltip=text).add_to(changes)
    changes.add_to(web_map)

    problems = folium.FeatureGroup(name="Weak spots", show=True)
    for spot in spots.itertuples():
        text = f"{spot.problem}: RSRP {spot.rsrp:.0f} dBm, {spot.dl_mbps:.1f} Mbps, {spot.rsrp_drives} drives"
        folium.CircleMarker([spot.bin_lat, spot.bin_lon], radius=9, color="#0b0b0b", weight=2,
                            fill=False, tooltip=text).add_to(problems)
    problems.add_to(web_map)

    folium.LayerControl(collapsed=False).add_to(web_map)
    web_map.get_root().html.add_child(folium.Element(_legend_html()))
    web_map.save(str(path))
    return path


def _legend_html():
    """A small fixed legend box for the interactive map."""
    def rows(legend):
        return "".join(
            f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{colour};margin-right:6px"></span>{label}</div>'
            for label, _, colour in reversed(legend)
        )
    return (
        '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#fcfcfb;'
        'padding:10px 12px;border:1px solid #c3c2b7;border-radius:6px;'
        'font:12px system-ui,sans-serif;color:#0b0b0b;line-height:1.5">'
        f'<b>RSRP</b>{rows(RSRP_LEGEND)}'
        f'<div style="margin-top:6px"><b>Throughput</b></div>{rows(THROUGHPUT_LEGEND)}'
        '<div style="margin-top:6px">Black ring = weak spot</div></div>'
    )
