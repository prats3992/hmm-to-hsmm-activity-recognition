"""Static baselines (Gaussian Naive Bayes, Random Forest) and the final model comparison.

Run after 03_gaussian_hmm.py, 05_gmm_hmm.py and 07_hsmm.py: their accuracy
tables are combined with the baselines into a single comparison.

Outputs:
  outputs/tables/baseline_accuracy.csv
  outputs/tables/model_comparison.csv
  outputs/figures/model_comparison_bar.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from har.config import FOCUS_SUBJECTS, RANDOM_STATE, TABLES_DIR
from har.data import load_dataset
from har.evaluation import accuracy_rows, save_table
from har.plotting import save_figure

MODEL_TABLES = ("gaussian_hmm_accuracy.csv", "gmm_hmm_accuracy.csv", "hsmm_accuracy.csv")
PLOTTED_MODELS = ("Naive Bayes", "Random Forest", "GMM-HMM (Viterbi)", "HSMM")


def train_baselines(train, test):
    rows = []
    for name, model in (("Naive Bayes", GaussianNB()),
                        ("Random Forest", RandomForestClassifier(n_estimators=100,
                                                                 random_state=RANDOM_STATE))):
        print(f"Training {name}...")
        model.fit(train.X, train.y)
        rows += accuracy_rows(name, test.y, model.predict(test.X), test.subjects)
    return pd.DataFrame(rows)


def plot_comparison(results):
    table = results.pivot(index="Subject", columns="Model", values="Accuracy")
    groups = [f"Subject {s}" for s in FOCUS_SUBJECTS] + ["Overall"]
    models = [m for m in PLOTTED_MODELS if m in table.columns]

    x = np.arange(len(groups))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, model in enumerate(models):
        offset = (k - (len(models) - 1) / 2) * width
        ax.bar(x + offset, table.loc[groups, model], width, label=model, alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison: Baselines vs GMM-HMM vs HSMM")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(models), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "model_comparison_bar.png")


def main():
    train, test, _ = load_dataset()
    baselines = train_baselines(train, test)
    save_table(baselines, "baseline_accuracy.csv")

    tables = [baselines]
    for name in MODEL_TABLES:
        path = TABLES_DIR / name
        if path.exists():
            tables.append(pd.read_csv(path))
        else:
            print(f"Warning: {path.name} not found; run the corresponding model script first.")
    results = pd.concat(tables, ignore_index=True)

    print("\n--- Model comparison ---")
    columns = ["Overall"] + [f"Subject {s}" for s in FOCUS_SUBJECTS]
    wide = results.pivot(index="Model", columns="Subject", values="Accuracy")[columns]
    save_table(wide.reset_index(), "model_comparison.csv")
    plot_comparison(results)


if __name__ == "__main__":
    main()
