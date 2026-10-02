"""Phase 1: test the assumptions of a single-Gaussian-emission HMM.

Checks, on the training split projected onto principal components:
  * Gaussianity of each activity's emissions (Shapiro-Wilk, Q-Q plots)
  * Stationarity within the longest segment of each activity (Augmented Dickey-Fuller)
  * Residual structure around the per-activity mean

Outputs:
  outputs/tables/gaussianity_tests.csv
  outputs/tables/stationarity_tests.csv
  outputs/figures/gaussianity_qq_plots.png
  outputs/figures/linearity_residuals.png
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from har.config import RANDOM_STATE
from har.data import load_dataset
from har.evaluation import save_table
from har.plotting import save_figure
from har.sequences import segment_bounds

N_PCS_TO_CHECK = 3
SHAPIRO_MAX_SAMPLES = 5000


def check_gaussianity(X, y, classes):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    fig, axes = plt.subplots(len(classes), N_PCS_TO_CHECK, figsize=(15, 20))
    for c, name in enumerate(classes):
        X_c = X[y == c]
        for pc in range(N_PCS_TO_CHECK):
            values = X_c[:, pc]
            sample = values
            if len(values) > SHAPIRO_MAX_SAMPLES:
                sample = rng.choice(values, SHAPIRO_MAX_SAMPLES, replace=False)
            stat, p = stats.shapiro(sample)
            rows.append({"Activity": name, "PC": pc + 1, "Shapiro-Stat": stat,
                         "p-value": p, "Gaussian": p > 0.05})
            ax = axes[c, pc]
            stats.probplot(values, dist="norm", plot=ax)
            ax.set_title(f"{name} - PC{pc + 1}\np={p:.2e}")
    fig.suptitle(f"Q-Q Plots for Top {N_PCS_TO_CHECK} PCA Components per Activity", fontsize=16)
    save_figure(fig, "gaussianity_qq_plots.png")
    return pd.DataFrame(rows)


def check_stationarity(X, y, subjects, classes):
    """ADF test on PC1 over the longest contiguous segment of each activity."""
    starts, ends = segment_bounds(y, subjects)
    rows = []
    for c, name in enumerate(classes):
        own = np.flatnonzero(y[starts] == c)
        k = own[np.argmax((ends - starts)[own])]
        length = ends[k] - starts[k]
        if length < 20:
            print(f"Skipping {name}: longest segment too short ({length})")
            continue
        with warnings.catch_warnings():
            # statsmodels >= 0.15 warns about a future change of return type.
            warnings.simplefilter("ignore", FutureWarning)
            stat, p = adfuller(X[starts[k]:ends[k], 0])[:2]
        rows.append({"Activity": name, "Subject": subjects[starts[k]], "Segment_Length": length,
                     "ADF_Stat": stat, "p-value": p, "Stationary": p < 0.05})
    return pd.DataFrame(rows)


def plot_residuals(X, y, classes):
    """Residuals of PC1 around each activity mean, plotted against the fitted mean."""
    fitted, residuals = [], []
    for c in range(len(classes)):
        values = X[y == c, 0]
        fitted.append(np.full(len(values), values.mean()))
        residuals.append(values - values.mean())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(np.concatenate(fitted), np.concatenate(residuals), alpha=0.1)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Fitted Values (Class Means)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values (PC1)")
    save_figure(fig, "linearity_residuals.png")


def main():
    train, _, encoder = load_dataset()
    classes = encoder.classes_

    print("\n--- Gaussianity (Shapiro-Wilk) ---")
    save_table(check_gaussianity(train.X, train.y, classes), "gaussianity_tests.csv")

    print("\n--- Stationarity (Augmented Dickey-Fuller) ---")
    save_table(check_stationarity(train.X, train.y, train.subjects, classes),
               "stationarity_tests.csv")

    print("\n--- Residual analysis ---")
    plot_residuals(train.X, train.y, classes)


if __name__ == "__main__":
    main()
