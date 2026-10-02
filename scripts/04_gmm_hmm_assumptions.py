"""Phase 2: test the assumptions of the GMM-HMM.

  * Goodness of fit of a 2-component mixture versus a single Gaussian (AIC)
  * Conditional independence of observations given the state (residual ACF)
  * Effect of downsampling on the autocorrelation
  * First-order Markov property of the frame-level label sequence (LRT)

Outputs:
  outputs/tables/gmm_aic_comparison.csv
  outputs/tables/residual_autocorrelation.csv
  outputs/tables/markov_test_frame_level.csv
  outputs/figures/gmm_vs_gaussian_fit.png
  outputs/figures/residual_acf.png
  outputs/figures/downsampling_acf.png
"""

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture

from har.config import RANDOM_STATE
from har.data import load_dataset
from har.evaluation import save_table
from har.plotting import save_figure
from har.sequences import markov_order_test, segment_bounds, split_by_group

DOWNSAMPLING_FACTORS = (1, 2, 4, 8)


def compare_gmm_fit(X, y, classes):
    """AIC of a single Gaussian versus a 2-component GMM on PC1 of each activity."""
    rows = []
    fig = plt.figure(figsize=(15, 10))
    for c, name in enumerate(classes):
        values = X[y == c, 0].reshape(-1, 1)
        g1 = GaussianMixture(n_components=1, random_state=RANDOM_STATE).fit(values)
        g2 = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(values)
        aic1, aic2 = g1.aic(values), g2.aic(values)
        rows.append({"Activity": name, "AIC_Gaussian": aic1, "AIC_GMM2": aic2,
                     "Improvement": aic1 - aic2})

        ax = fig.add_subplot(2, 3, c + 1)
        sns.histplot(values.ravel(), stat="density", bins=50, alpha=0.4, label="Data", ax=ax)
        grid = np.linspace(values.min(), values.max(), 1000).reshape(-1, 1)
        ax.plot(grid, np.exp(g1.score_samples(grid)), "r--", label="Single", linewidth=2)
        ax.plot(grid, np.exp(g2.score_samples(grid)), "b-", label="GMM(2)", linewidth=2)
        ax.set_title(f"{name}\nAIC Imp: {aic1 - aic2:.0f}")
        if c == 0:
            ax.legend()
    fig.suptitle("Goodness of Fit: Single Gaussian vs GMM (2 Components)", fontsize=16)
    save_figure(fig, "gmm_vs_gaussian_fit.png")
    return pd.DataFrame(rows)


def pooled_acf(segments, nlags):
    """Autocorrelation of residuals pooled over segments; lags never span two segments.

    Each segment is a 1-D array of residuals around the activity mean.
    """
    variance = sum(np.sum(e ** 2) for e in segments)
    acf = [1.0]
    for k in range(1, nlags + 1):
        acf.append(sum(np.sum(e[:-k] * e[k:]) for e in segments if len(e) > k) / variance)
    return np.array(acf)


def activity_residual_segments(X, y, subjects, label, step=1):
    """PC1 residuals around the activity mean, one array per contiguous segment.

    ``step`` keeps every ``step``-th window of each segment (downsampling).
    """
    starts, ends = segment_bounds(y, subjects)
    mean = X[y == label, 0].mean()
    return [X[s:e:step, 0] - mean for s, e in zip(starts, ends) if y[s] == label]


def check_residual_autocorrelation(X, y, subjects, classes):
    """ACF of PC1 residuals within each activity's contiguous segments."""
    rows = []
    fig = plt.figure(figsize=(15, 10))
    for c, name in enumerate(classes):
        acf = pooled_acf(activity_residual_segments(X, y, subjects, c), nlags=20)
        rows.append({"Activity": name, "Lag1_ACF": acf[1]})

        ax = fig.add_subplot(2, 3, c + 1)
        ax.bar(range(len(acf)), acf)
        ax.axhline(0.05, color="r", linestyle="--")
        ax.axhline(-0.05, color="r", linestyle="--")
        ax.set_title(f"{name}\nLag-1 ACF: {acf[1]:.2f}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_ylim(-0.2, 1.0)
    fig.suptitle("Autocorrelation of Residuals (Conditional Independence Check)", fontsize=16)
    save_figure(fig, "residual_acf.png")
    return pd.DataFrame(rows)


def check_downsampling(X, y, subjects, classes, activity="WALKING"):
    """Lag-1 autocorrelation of PC1 for one activity after keeping every k-th window."""
    label = list(classes).index(activity)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, factor in zip(axes.ravel(), DOWNSAMPLING_FACTORS):
        segments = activity_residual_segments(X, y, subjects, label, step=factor)
        acf = pooled_acf(segments, nlags=5)
        n_windows = sum(len(e) for e in segments)
        print(f"Downsample factor {factor}x: {n_windows} windows, lag-1 ACF = {acf[1]:.4f}")
        ax.bar(range(len(acf)), acf)
        ax.axhline(0.05, color="r", linestyle="--")
        ax.set_title(f"{activity}, factor {factor}x (Lag-1 ACF: {acf[1]:.2f})")
        ax.set_xlabel("Lag (in downsampled windows)")
        ax.set_ylim(-0.2, 1.0)
    save_figure(fig, "downsampling_acf.png")


def main():
    train, _, encoder = load_dataset()
    classes = encoder.classes_

    print("\n--- Goodness of fit: Gaussian vs GMM (AIC, higher improvement favours GMM) ---")
    save_table(compare_gmm_fit(train.X, train.y, classes), "gmm_aic_comparison.csv")

    print("\n--- Conditional independence (residual autocorrelation) ---")
    save_table(check_residual_autocorrelation(train.X, train.y, train.subjects, classes),
               "residual_autocorrelation.csv")

    print("\n--- Downsampling ---")
    check_downsampling(train.X, train.y, train.subjects, classes)

    print("\n--- Markov property of the frame-level sequence (order 1 vs order 2) ---")
    result = markov_order_test(split_by_group(train.y, train.subjects))
    verdict = "first-order Markov" if result.is_first_order else "NOT first-order Markov"
    print(f"G = {result.g_statistic:.2f}, dof = {result.dof}, p = {result.p_value:.4g} "
          f"-> {verdict}")
    save_table(pd.DataFrame([{"Sequence": "frame-level", **asdict(result)}]),
               "markov_test_frame_level.csv")


if __name__ == "__main__":
    main()
