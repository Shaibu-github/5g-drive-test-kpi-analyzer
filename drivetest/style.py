"""One visual style for the whole project: figures, maps and the dashboard.

The same colour always means the same thing. Text is never coloured: titles,
labels and numbers use the ink colours, and only the marks carry a data colour.
"""
SURFACE = "#fcfcfb"      # chart background
PAGE = "#f9f9f7"         # page background behind the charts
INK = "#0b0b0b"          # titles and values
INK_SOFT = "#52514e"     # axis labels, notes
MUTED = "#898781"        # tick labels
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TECH_COLOURS = {"5G": "#2a78d6", "LTE": "#eb6834", "3G": "#1baf7a"}
MODEL_COLOURS = {"Linear regression": "#2a78d6", "Gradient boosting": "#eb6834", "Neural network": "#1baf7a"}
RULE_COLOURS = {"Border": "#2a78d6", "Margin": "#eb6834"}
