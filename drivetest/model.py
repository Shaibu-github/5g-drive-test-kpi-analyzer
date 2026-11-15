"""Step 6 - prediction: estimate downlink throughput from what the phone can measure.

Design choices (each one is a lesson from the data):

* Target = mean throughput of a 10-second window, not of one sample. One-second
  throughput is mostly TCP and scheduler noise, and the radio KPIs only update
  every 2-3 seconds.
* The model learns log(1 + Mbps). Throughput goes from 0 to 500 Mbps; on a log
  scale a miss from 2 to 4 Mbps counts like a miss from 100 to 200 Mbps.
* Test data = whole drives the model has never seen (grouped cross-validation).
  Neighbouring windows of one drive look alike, so a random split would leak
  the answer and give a score that is too good.
* Uplink bitrate is NOT used as an input. During a download the uplink carries
  the TCP acknowledgements, so it is a result of the downlink, not a cause.
* Speed is NOT used either: the logger freezes it between two position updates
  (it never reads 0 at a red light), so it is not a trustworthy input.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .clean import downloading

RADIO = ["rsrp", "rsrq", "snr", "cqi", "rssi", "nbr_rsrp", "dominance", "is_5g"]

# Four sets of inputs, from "knows almost nothing" to "knows the recent past".
FEATURE_SETS = {
    "Technology only": ["is_5g"],
    "Radio KPIs": RADIO,
    "Radio KPIs + location": RADIO + ["lat", "lon"],
    "Radio KPIs + location + history": RADIO + ["lat", "lon", "previous_mbps"],
}

MODELS = ["Linear regression", "Gradient boosting", "Neural network"]


# ----------------------------------------------------------------------------
# 1. Build the table the models learn from
# ----------------------------------------------------------------------------

def make_windows(df, seconds=10):
    """Average the samples of each drive into windows of `seconds` seconds.

    Only LTE and 5G download samples are used: the radio KPIs do not exist on 3G.
    """
    active = downloading(df)
    active = active[active["tech"].isin(["5G", "LTE"])].copy()
    active["is_5g"] = (active["tech"] == "5G").astype(float)
    # dominance: how many dB the serving cell is above the best neighbour
    active["dominance"] = active["rsrp"] - active["nbr_rsrp"]
    active["window"] = (active["elapsed_s"] // seconds).astype(int)

    inputs = ["rsrp", "rsrq", "snr", "cqi", "rssi", "nbr_rsrp", "dominance", "is_5g", "lat", "lon"]
    grouped = active.groupby(["drive", "window"])
    windows = grouped[inputs + ["dl_mbps"]].mean()
    windows["samples"] = grouped.size()
    windows = windows.reset_index()

    # history: the throughput of the window just before this one (same drive)
    previous = windows.groupby("drive")[["window", "dl_mbps"]].shift()
    is_neighbour = previous["window"] == windows["window"] - 1
    windows["previous_mbps"] = previous["dl_mbps"].where(is_neighbour)

    # keep windows that are at least half full
    windows = windows[windows["samples"] >= seconds / 2]
    return windows.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. The models
# ----------------------------------------------------------------------------

def make_sklearn_model(name):
    """Return a fresh, untrained scikit-learn model."""
    if name == "Linear regression":
        # fill gaps with the median, put all inputs on the same scale, then fit a line
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "Gradient boosting":
        # many small decision trees; handles missing values by itself
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
    raise ValueError(f"unknown model: {name}")


class ThroughputNet(nn.Module):
    """A small fully connected network: inputs -> 32 -> 16 -> 1."""

    def __init__(self, n_inputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


def train_network(x_train, y_train, epochs=300, learning_rate=0.01, seed=0):
    """Train the network with the classic PyTorch loop. Returns the model."""
    torch.manual_seed(seed)                       # same result on every run
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    model = ThroughputNet(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_function = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()                     # 1. forget the old gradients
        loss = loss_function(model(x), y)         # 2. how wrong is the model now?
        loss.backward()                           # 3. gradient of the error for each weight
        optimizer.step()                          # 4. move the weights a small step
    return model


def predict_network(model, x):
    model.eval()
    with torch.no_grad():                         # no gradients needed to predict
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def fit_predict(name, x_train, y_train, x_test):
    """Train model `name` on the training rows and predict the test rows.

    `y_train` and the prediction are in log(1 + Mbps).
    """
    if name == "Neural network":
        # the network needs complete, scaled inputs; learn the scaling on the
        # training rows only, then apply it to the test rows
        prepare = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        model = train_network(prepare.fit_transform(x_train), y_train)
        return predict_network(model, prepare.transform(x_test))
    model = make_sklearn_model(name)
    model.fit(x_train, y_train)
    return model.predict(x_test)


# ----------------------------------------------------------------------------
# 3. Honest evaluation: test on drives the model has never seen
# ----------------------------------------------------------------------------

def drive_folds(windows, n_splits=5):
    """Split the windows into folds so that a drive is never in both sides."""
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(windows, groups=windows["drive"]))


def cross_validated_prediction(windows, features, name, folds):
    """Predict every window with a model that was trained on other folds."""
    x = windows[features].to_numpy(dtype=float)
    y_log = np.log1p(windows["dl_mbps"].to_numpy())
    predicted = np.zeros(len(windows))
    for train_rows, test_rows in folds:
        log_prediction = fit_predict(name, x[train_rows], y_log[train_rows], x[test_rows])
        predicted[test_rows] = np.expm1(log_prediction)      # back to Mbps
    return np.clip(predicted, 0, None)                       # throughput cannot be negative


def score(actual_mbps, predicted_mbps):
    """R2 on the log scale (share of variation explained) and MAE in Mbps."""
    return {
        "r2_log": r2_score(np.log1p(actual_mbps), np.log1p(predicted_mbps)),
        "mae_mbps": mean_absolute_error(actual_mbps, predicted_mbps),
    }


def baseline_scores(windows, folds):
    """Two predictions that need no machine learning. A model must beat them.

    Persistence can only be scored on windows that have a previous window
    (all but the first window of each drive and the windows after a gap).
    """
    actual = windows["dl_mbps"].to_numpy()
    median = np.zeros(len(windows))
    for train_rows, test_rows in folds:
        median[test_rows] = np.median(actual[train_rows])
    # persistence: "the next 10 seconds will be like the last 10 seconds"
    has_previous = windows["previous_mbps"].notna().to_numpy()
    rows = {
        "Median of training drives": score(actual, median),
        "Persistence (previous window)": score(actual[has_previous], windows["previous_mbps"].to_numpy()[has_previous]),
    }
    return pd.DataFrame(rows).T


def fold_spread(windows, predicted, folds):
    """R2 of every test fold on its own.

    The overall score hides how much the result depends on which drives are
    in the test set. A wide spread means: small differences between two
    models are noise.
    """
    actual_log = np.log1p(windows["dl_mbps"].to_numpy())
    predicted_log = np.log1p(predicted)
    return [r2_score(actual_log[test_rows], predicted_log[test_rows]) for _, test_rows in folds]


def compare_models(windows, folds):
    """Score every model on every feature set. Returns (results, predictions)."""
    actual = windows["dl_mbps"].to_numpy()
    rows, predictions = [], {}
    for set_name, features in FEATURE_SETS.items():
        for name in MODELS:
            predicted = cross_validated_prediction(windows, features, name, folds)
            predictions[(set_name, name)] = predicted
            per_fold = fold_spread(windows, predicted, folds)
            rows.append({"features": set_name, "model": name, **score(actual, predicted),
                         "worst_fold_r2": min(per_fold), "best_fold_r2": max(per_fold)})
    return pd.DataFrame(rows), predictions


def random_split_score(windows, features, name="Gradient boosting", n_splits=5):
    """The WRONG way, kept to show the size of the mistake.

    Windows are shuffled before splitting, so the model is tested on windows
    whose neighbours (same drive, same minute) it has seen in training.
    """
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    folds = list(splitter.split(windows))
    predicted = cross_validated_prediction(windows, features, name, folds)
    return score(windows["dl_mbps"].to_numpy(), predicted)


def feature_importance(windows, features, n_splits=5, repeats=5):
    """Permutation importance of each input for the gradient boosting model.

    For each test fold: shuffle one input column, predict again, and measure how
    much R2 drops. A big drop means the model really needs that input.
    """
    x = windows[features].to_numpy(dtype=float)
    y_log = np.log1p(windows["dl_mbps"].to_numpy())
    rng = np.random.default_rng(0)
    drops = np.zeros(len(features))
    folds = drive_folds(windows, n_splits)
    for train_rows, test_rows in folds:
        model = make_sklearn_model("Gradient boosting").fit(x[train_rows], y_log[train_rows])
        base = r2_score(y_log[test_rows], model.predict(x[test_rows]))
        for column in range(len(features)):
            for _ in range(repeats):
                shuffled = x[test_rows].copy()
                shuffled[:, column] = rng.permutation(shuffled[:, column])
                drops[column] += base - r2_score(y_log[test_rows], model.predict(shuffled))
    importance = pd.Series(drops / (len(folds) * repeats), index=features)
    return importance.sort_values(ascending=False).round(3)
