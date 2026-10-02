"""Shared plotting helpers. All figures are written under ``outputs/figures``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from har.config import FIGURES_DIR  # noqa: E402


def save_figure(fig, name):
    """Save ``fig`` as ``outputs/figures/<name>`` and close it."""
    path = FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path.relative_to(FIGURES_DIR.parent.parent)}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, name, cmap="Blues", figsize=(10, 8)):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    save_figure(fig, name)


def plot_decoding(y_true, y_pred, class_names, title, name, style="r--", pred_label="Prediction"):
    fig, ax = plt.subplots(figsize=(15, 6))
    t = np.arange(len(y_true))
    ax.plot(t, y_true, "k-", label="True Label", linewidth=2, alpha=0.6)
    ax.plot(t, y_pred, style, label=pred_label, linewidth=1.5)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Window Index")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)


def plot_transition_matrix(probs, class_names, title, name, cmap="Blues", fmt=".3f",
                           xlabel="Next Activity", ylabel="Current Activity"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(probs, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_figure(fig, name)
