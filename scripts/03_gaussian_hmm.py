"""Phase 1 model: one Gaussian-emission HMM per activity, frame-level classification.

Each test window is assigned the activity whose model gives it the highest
likelihood; no temporal smoothing across windows is applied.

Outputs:
  outputs/tables/gaussian_hmm_accuracy.csv
  outputs/figures/confusion_matrix_hmm.png
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from har.data import load_dataset
from har.evaluation import accuracy_rows, save_table
from har.models import fit_class_models, frame_log_likelihoods
from har.plotting import plot_confusion_matrix


def main():
    train, test, encoder = load_dataset()

    print("Training one Gaussian HMM per activity...")
    models = fit_class_models(train.X, train.y, train.subjects, emission="gaussian")
    y_pred = np.argmax(frame_log_likelihoods(models, test.X), axis=1)

    print(classification_report(test.y, y_pred, target_names=encoder.classes_))
    rows = accuracy_rows("Gaussian HMM (frame-level)", test.y, y_pred, test.subjects)
    save_table(pd.DataFrame(rows), "gaussian_hmm_accuracy.csv")
    plot_confusion_matrix(test.y, y_pred, encoder.classes_, "HMM Confusion Matrix",
                          "confusion_matrix_hmm.png")


if __name__ == "__main__":
    main()
