"""Phase 3 model: hidden semi-Markov model with duration-explicit Viterbi decoding.

GMM-HMM likelihoods serve as emissions, activity durations follow a truncated
Gaussian estimated from training segments, and transitions are modelled by the
embedded chain (transitions between distinct activities only).

Outputs:
  outputs/tables/hsmm_accuracy.csv
  outputs/figures/duration_distributions.png
  outputs/figures/hsmm_confusion_matrix_overall.png
  outputs/figures/hsmm_confusion_matrix_subj{2,9,12}.png
  outputs/figures/hsmm_decoding_subj{2,9,12}.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from har.config import FOCUS_SUBJECTS, HSMM_MAX_DURATION
from har.data import load_dataset
from har.decoding import embedded_transition_log_probs, fit_duration_params, hsmm_viterbi
from har.evaluation import accuracy_rows, save_table
from har.models import fit_class_models, frame_log_likelihoods
from har.plotting import plot_confusion_matrix, plot_decoding, save_figure
from har.sequences import durations_by_state


def plot_duration_histograms(y, subjects, classes, duration_params):
    fig = plt.figure(figsize=(15, 10))
    for c, durs in durations_by_state(y, subjects, range(len(classes))).items():
        mu, std = duration_params[c]
        ax = fig.add_subplot(2, 3, c + 1)
        sns.histplot(durs, kde=True, stat="density", ax=ax)
        ax.set_title(f"{classes[c]} Duration\nMean={mu:.1f}, Std={std:.1f}")
        ax.set_xlim(0, np.percentile(durs, 95) * 1.5)
    save_figure(fig, "duration_distributions.png")


def main():
    train, test, encoder = load_dataset()
    classes = encoder.classes_
    n_classes = len(classes)

    print("Training GMM-HMM emission models...")
    models = fit_class_models(train.X, train.y, train.subjects, emission="gmm")
    duration_params = fit_duration_params(train.y, n_classes, train.subjects)
    log_trans = embedded_transition_log_probs(train.y, n_classes, train.subjects)
    plot_duration_histograms(train.y, train.subjects, classes, duration_params)

    y_pred = np.empty_like(test.y)
    for subject in np.unique(test.subjects):
        mask = test.subjects == subject
        log_lik = frame_log_likelihoods(models, test.X[mask])
        y_pred[mask] = hsmm_viterbi(log_lik, log_trans, duration_params, HSMM_MAX_DURATION)
        acc = accuracy_score(test.y[mask], y_pred[mask])
        print(f"Subject {subject}: HSMM accuracy {acc:.4f}")

        if subject in FOCUS_SUBJECTS:
            plot_decoding(test.y[mask], y_pred[mask], classes,
                          f"HSMM Decoding (Subject {subject}) - Accuracy: {acc:.2%}",
                          f"hsmm_decoding_subj{subject}.png",
                          style="g--", pred_label="HSMM Prediction")
            plot_confusion_matrix(test.y[mask], y_pred[mask], classes,
                                  f"Confusion Matrix (Subject {subject})",
                                  f"hsmm_confusion_matrix_subj{subject}.png", figsize=(8, 6))

    plot_confusion_matrix(test.y, y_pred, classes, "HSMM Overall Confusion Matrix",
                          "hsmm_confusion_matrix_overall.png")
    save_table(pd.DataFrame(accuracy_rows("HSMM", test.y, y_pred, test.subjects)),
               "hsmm_accuracy.csv")


if __name__ == "__main__":
    main()
