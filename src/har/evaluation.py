"""Accuracy bookkeeping shared by the training scripts."""

import pandas as pd
from sklearn.metrics import accuracy_score

from har.config import FOCUS_SUBJECTS, TABLES_DIR


def accuracy_rows(model_name, y_true, y_pred, subjects, focus_subjects=FOCUS_SUBJECTS):
    """Overall accuracy plus accuracy on each focus subject, as table rows."""
    rows = [{"Model": model_name, "Subject": "Overall",
             "Accuracy": accuracy_score(y_true, y_pred)}]
    for s in focus_subjects:
        mask = subjects == s
        if mask.any():
            rows.append({"Model": model_name, "Subject": f"Subject {s}",
                         "Accuracy": accuracy_score(y_true[mask], y_pred[mask])})
    return rows


def save_table(df, name):
    """Write ``df`` to ``outputs/tables/<name>`` and print it."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / name
    df.to_csv(path, index=False)
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(df.to_string(index=False))
    print(f"Saved {path.relative_to(TABLES_DIR.parent.parent)}")
