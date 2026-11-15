"""Step 1 - get the data: download the 5G dataset and load the raw drive-test logs.

Dataset: Raca, Leahy, Sreenan, Quinlan, "Beyond Throughput, The Next Generation:
A 5G Dataset with Channel and Context Metrics", ACM MMSys 2020.
https://github.com/uccmisl/5Gdataset (GPL-3.0). The data is not stored in this
repository; this module downloads it.
"""
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

DATASET_URL = "https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip"
DATASET_FOLDER = "5G-production-dataset"

# The dataset has several test types. We use the file-download test measured
# from a moving car, because it is the closest to a classic drive test.
SCENARIO = "Download/Driving"


def download_dataset(data_dir="data"):
    """Download and unzip the dataset once. Return the dataset folder."""
    data_dir = Path(data_dir)
    root = data_dir / DATASET_FOLDER
    if root.exists():
        return root  # already downloaded: nothing to do

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / (DATASET_FOLDER + ".zip")
    if not zip_path.exists():
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        # The zip also contains a "__MACOSX" folder (Finder metadata). Skip it.
        members = [m for m in archive.namelist() if not m.startswith("__MACOSX")]
        archive.extractall(data_dir, members=members)
    return root


def list_logs(root, scenario=SCENARIO):
    """Return the log files of one scenario, sorted by start time."""
    return sorted((Path(root) / scenario).glob("*.csv"))


def load_log(path):
    """Read one G-NetTrack Pro log file into a table.

    G-NetTrack writes "-" when it has no value, so "-" is read as missing (NaN).
    This only matches a cell that is exactly "-"; negative numbers such as -99
    are not affected.
    """
    # The three hex ID columns must stay text. Otherwise pandas could read an
    # ID such as "1E3" as the number 1000.
    hex_columns = {"NODEHEX": str, "CELLHEX": str, "LACHEX": str}
    log = pd.read_csv(path, na_values="-", dtype=hex_columns)
    # File name: B_2019.12.14_10.16.30.csv -> operator "B" + start time of the drive.
    # The start time is a natural ID for the drive. A file name without "_"
    # (your own log, for example) is used as it is.
    log["drive"] = Path(path).stem.split("_", 1)[-1]
    return log


def load_logs(paths):
    """Read many log files and stack them into one table (one row per sample)."""
    return pd.concat([load_log(p) for p in paths], ignore_index=True)
