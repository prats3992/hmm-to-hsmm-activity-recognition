"""Exploratory data analysis and justification for PCA.

Outputs: outputs/figures/eda/*.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from har.config import RANDOM_STATE
from har.data import LABEL_COLUMN, SUBJECT_COLUMN, feature_columns, read_split
from har.plotting import save_figure


def plot_dataset_overview(train_df, test_df, features):
    print(f"Train shape: {train_df.shape}, test shape: {test_df.shape}")
    print(f"Feature columns: {len(features)}")
    missing = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
    print(f"Missing values: {int(missing)}")

    meta = [SUBJECT_COLUMN, LABEL_COLUMN]
    all_df = pd.concat([train_df[meta], test_df[meta]], keys=["train", "test"], names=["dataset"])
    all_df = all_df.reset_index(level="dataset")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=all_df, x=LABEL_COLUMN, hue="dataset", ax=ax)
    ax.set_title("Activity Distribution in Train and Test Sets")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "eda/class_distribution.png")

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.countplot(data=all_df, x=SUBJECT_COLUMN, hue="dataset", ax=ax)
    ax.set_title("Windows per Subject")
    save_figure(fig, "eda/subject_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, features[:3]):
        sns.histplot(data=train_df, x=col, hue=LABEL_COLUMN, element="step",
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {col}")
    save_figure(fig, "eda/feature_distributions_top3.png")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(train_df[features[:20]].corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix (First 20 Features)")
    save_figure(fig, "eda/correlation_matrix_top20.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=train_df, x=LABEL_COLUMN, y=features[0], ax=ax)
    ax.set_title(f"Boxplot of {features[0]} by Activity")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "eda/boxplot_feature1.png")


def plot_pca_justification(train_df, features, subset_size=50):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(train_df[features[:subset_size]].corr(), cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Correlation Matrix (First {subset_size} Features)\n"
                 "Red blocks indicate high redundancy")
    save_figure(fig, "eda/pca_justification_correlation.png")

    X = StandardScaler().fit_transform(train_df[features].values)
    cumulative = np.cumsum(PCA(random_state=RANDOM_STATE).fit(X).explained_variance_ratio_)
    n = {q: int(np.argmax(cumulative >= q)) + 1 for q in (0.90, 0.95, 0.99)}
    for q, k in n.items():
        print(f"Components for {q:.0%} variance: {k}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative, linewidth=2)
    for q, color in ((0.90, "r"), (0.95, "g")):
        ax.axhline(q, color=color, linestyle="--", label=f"{q:.0%} Variance ({n[q]} components)")
        ax.axvline(n[q], color=color, linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title(f"PCA Explained Variance Analysis\n"
                 f"Why we can reduce {len(features)} features to ~{n[0.90]}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "eda/pca_explained_variance.png")


def main():
    train_df = read_split("train")
    test_df = read_split("test")
    features = feature_columns(train_df)
    plot_dataset_overview(train_df, test_df, features)
    plot_pca_justification(train_df, features)


if __name__ == "__main__":
    main()
