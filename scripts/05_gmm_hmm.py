"""Phase 2 model: GMM-emission HMM per activity, with and without Viterbi decoding.

1. Frame-level: each window is assigned the activity with the highest GMM-HMM likelihood.
2. Viterbi: the per-activity likelihoods are used as emissions of a 6-state HMM
   whose transition matrix is estimated from training labels, and each test
   subject's sequence is decoded jointly.

Outputs:
  outputs/tables/gmm_hmm_accuracy.csv
  outputs/figures/confusion_matrix_gmm_hmm.png
  outputs/figures/confusion_matrix_gmm_hmm_viterbi.png
  outputs/figures/viterbi_decoding_subj{2,9,12}.png
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from har.config import FOCUS_SUBJECTS
from har.data import load_dataset
from har.decoding import frame_transition_log_probs, viterbi
from har.evaluation import accuracy_rows, save_table
from har.models import fit_class_models, frame_log_likelihoods
from har.plotting import plot_confusion_matrix, plot_decoding


def main():
    train, test, encoder = load_dataset()
    classes = encoder.classes_
    n_classes = len(classes)

    print("Training one GMM-HMM per activity...")
    models = fit_class_models(train.X, train.y, train.subjects, emission="gmm")
    log_lik = frame_log_likelihoods(models, test.X)

    # Frame-level classification.
    y_frame = np.argmax(log_lik, axis=1)
    print(classification_report(test.y, y_frame, target_names=classes))
    plot_confusion_matrix(test.y, y_frame, classes, "GMM-HMM Confusion Matrix",
                          "confusion_matrix_gmm_hmm.png", cmap="Greens")

    # Viterbi decoding of each test subject's sequence.
    log_trans = frame_transition_log_probs(train.y, n_classes, train.subjects)
    log_start = np.full(n_classes, -np.log(n_classes))
    y_viterbi = np.empty_like(test.y)
    for subject in np.unique(test.subjects):
        mask = test.subjects == subject
        y_viterbi[mask] = viterbi(log_lik[mask], log_trans, log_start)
        if subject in FOCUS_SUBJECTS:
            acc_frame = accuracy_score(test.y[mask], y_frame[mask])
            acc_viterbi = accuracy_score(test.y[mask], y_viterbi[mask])
            print(f"Subject {subject}: frame-level {acc_frame:.4f}, Viterbi {acc_viterbi:.4f}")
            plot_decoding(test.y[mask], y_viterbi[mask], classes,
                          f"Activity Sequence Decoding (Subject {subject})\n"
                          f"Viterbi Accuracy: {acc_viterbi:.2%}",
                          f"viterbi_decoding_subj{subject}.png",
                          style="r--", pred_label="Viterbi Prediction")

    plot_confusion_matrix(test.y, y_viterbi, classes, "GMM-HMM (Viterbi) Confusion Matrix",
                          "confusion_matrix_gmm_hmm_viterbi.png", cmap="Greens")

    rows = (accuracy_rows("GMM-HMM (frame-level)", test.y, y_frame, test.subjects)
            + accuracy_rows("GMM-HMM (Viterbi)", test.y, y_viterbi, test.subjects))
    save_table(pd.DataFrame(rows), "gmm_hmm_accuracy.csv")


if __name__ == "__main__":
    main()
