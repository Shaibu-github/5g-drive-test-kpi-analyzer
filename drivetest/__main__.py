"""Entry point for `python -m drivetest`."""
import argparse

from .pipeline import run

parser = argparse.ArgumentParser(description="Run the 5G drive-test KPI analysis from raw logs to figures.")
parser.add_argument("--data-dir", default="data", help="where the dataset is downloaded (default: data)")
parser.add_argument("--figures-dir", default="figures", help="where the figures are written (default: figures)")
parser.add_argument("--results-dir", default="results", help="where the tables are written (default: results)")
args = parser.parse_args()

run(args.data_dir, args.figures_dir, args.results_dir)
