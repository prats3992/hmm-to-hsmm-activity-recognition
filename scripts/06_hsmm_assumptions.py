"""Phase 3: test the assumptions of the hidden semi-Markov model.

  * Markov property of the embedded chain (sequence of distinct activities), LRT
  * Fit of activity durations to Normal, Log-Normal and Geometric distributions (KS)
  * Frame-level versus embedded transition matrices
  * Segment-length percentiles, used to choose the decoder's maximum duration

Outputs:
  outputs/tables/markov_test_embedded.csv
  outputs/tables/hsmm_duration_fits.csv
  outputs/figures/hsmm_duration_assumptions.png
  outputs/figures/tpm_frame_level_gmm_hmm.png
  outputs/figures/tpm_embedded_hsmm.png
"""

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from har.data import load_labels
from har.evaluation import save_table
from har.plotting import plot_transition_matrix, save_figure
from har.sequences import (durations_by_state, embedded_sequences, markov_order_test,
                           normalize_rows, run_lengths, transition_counts)


def test_embedded_markov(y, subjects):
    embedded = embedded_sequences(y, subjects)
    print(f"Frame-level length: {len(y)}, embedded length: {sum(map(len, embedded))} "
          f"over {len(embedded)} subjects")
    result = markov_order_test(embedded)
    verdict = "first-order Markov" if result.is_first_order else "NOT first-order Markov"
    print(f"G = {result.g_statistic:.2f}, dof = {result.dof}, p = {result.p_value:.4g} "
          f"-> {verdict}")
    return pd.DataFrame([{"Sequence": "embedded", **asdict(result)}])


def fit_duration_distributions(y, subjects, classes):
    """KS distance of each activity's segment lengths to Normal, Log-Normal and Geometric fits."""
    rows = []
    fig = plt.figure(figsize=(18, 12))
    for c, durs in durations_by_state(y, subjects, range(len(classes))).items():
        mu, std = stats.norm.fit(durs)
        shape, loc, scale = stats.lognorm.fit(durs, floc=0)
        p_geom = 1.0 / np.mean(durs)

        ks = {
            "Normal": stats.kstest(durs, stats.norm(mu, std).cdf).statistic,
            "LogNorm": stats.kstest(durs, stats.lognorm(shape, loc, scale).cdf).statistic,
            "Geometric": stats.kstest(durs, stats.geom(p_geom).cdf).statistic,
        }
        best = min(ks, key=ks.get)
        rows.append({"Activity": classes[c], "Count": len(durs), "Mean_Dur": np.mean(durs),
                     "Best_Fit": best, "Normal_KS": ks["Normal"],
                     "LogNorm_KS": ks["LogNorm"], "Geom_KS": ks["Geometric"]})

        ax = fig.add_subplot(2, 3, c + 1)
        sns.histplot(durs, stat="density", alpha=0.4, label="Data", ax=ax)
        x = np.linspace(durs.min(), durs.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), "r-", label=f"Normal (KS={ks['Normal']:.2f})")
        ax.plot(x, stats.lognorm.pdf(x, shape, loc, scale), "g--",
                label=f"LogNorm (KS={ks['LogNorm']:.2f})")
        ax.set_title(f"{classes[c]} Duration\nBest Fit: {best}")
        ax.legend()
    save_figure(fig, "hsmm_duration_assumptions.png")
    return pd.DataFrame(rows)


def plot_transition_matrices(y, subjects, classes):
    n = len(classes)
    embedded = np.zeros((n, n))
    for seq in embedded_sequences(y, subjects):
        embedded += transition_counts(seq, n)
    plot_transition_matrix(
        normalize_rows(transition_counts(y, n, subjects)), classes,
        "Frame-Level Transition Probability Matrix (GMM-HMM)\n"
        "Note the dominant diagonal (Self-Transitions)",
        "tpm_frame_level_gmm_hmm.png", cmap="Blues", fmt=".3f",
        xlabel="Next Activity (t+1)", ylabel="Current Activity (t)")
    plot_transition_matrix(
        normalize_rows(embedded), classes,
        "Embedded Transition Probability Matrix (HSMM)\n"
        "Transitions GIVEN that the activity changes",
        "tpm_embedded_hsmm.png", cmap="Greens", fmt=".2f")


def main():
    y, subjects, classes = load_labels("train")

    print("\n--- Markov property of the embedded chain (order 1 vs order 2) ---")
    save_table(test_embedded_markov(y, subjects), "markov_test_embedded.csv")

    print("\n--- Duration distribution fits (KS statistic, lower is better) ---")
    save_table(fit_duration_distributions(y, subjects, classes), "hsmm_duration_fits.csv")

    print("\n--- Transition matrices ---")
    plot_transition_matrices(y, subjects, classes)

    print("\n--- Segment lengths (windows) ---")
    for split in ("train", "test"):
        _, lengths = run_lengths(*load_labels(split)[:2])
        print(f"{split}: max {lengths.max()}, 95th percentile {np.percentile(lengths, 95):.1f}, "
              f"99th percentile {np.percentile(lengths, 99):.1f}")


if __name__ == "__main__":
    main()
