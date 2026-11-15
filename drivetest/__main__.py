"""Entry point for `python -m drivetest`.

    python -m drivetest                     run the whole analysis on the public dataset
    python -m drivetest report my_log.csv   write an HTML report for one log file
"""
import argparse

parser = argparse.ArgumentParser(prog="python -m drivetest", description="5G drive-test KPI analysis: from raw logs to figures, tables and a dashboard.")
parser.add_argument("--data-dir", default="data", help="where the dataset is downloaded (default: data)")
parser.add_argument("--figures-dir", default="figures", help="where the figures are written (default: figures)")
parser.add_argument("--results-dir", default="results", help="where the tables are written (default: results)")
parser.add_argument("--site-dir", default="docs", help="where the dashboard is written (default: docs)")
commands = parser.add_subparsers(dest="command")
one_log = commands.add_parser("report", help="write an HTML report for one G-NetTrack Pro log file")
one_log.add_argument("log", help="path of the log file (.csv)")
one_log.add_argument("--out", help="path of the report (default: next to the log, <name>_report.html)")
args = parser.parse_args()

if args.command == "report":
    from .report import report_for_log
    print(f"report written to {report_for_log(args.log, args.out)}")
else:
    from .pipeline import run
    run(args.data_dir, args.figures_dir, args.results_dir, args.site_dir)
