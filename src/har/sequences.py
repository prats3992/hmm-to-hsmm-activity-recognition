"""Label-sequence statistics: segments, transition matrices and Markov-order tests.

Most functions take an optional ``groups`` array (the subject id of each
window). Runs and transitions never cross a change of group, so the end of one
subject's recording is not joined to the start of the next.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _same_group(groups, n):
    """Boolean mask over consecutive pairs (t, t+1) that belong to the same group."""
    if groups is None:
        return np.ones(n - 1, dtype=bool)
    groups = np.asarray(groups)
    return groups[1:] == groups[:-1]


def segment_bounds(y, groups=None):
    """Start and end (exclusive) indices of maximal constant runs of ``y`` within each group."""
    y = np.asarray(y)
    breaks = (y[1:] != y[:-1]) | ~_same_group(groups, len(y))
    change = np.flatnonzero(breaks) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(y)]))
    return starts, ends


def run_lengths(y, groups=None):
    """Return ``(labels, lengths)`` of the segments of ``y``."""
    starts, ends = segment_bounds(y, groups)
    return np.asarray(y)[starts], ends - starts


def durations_by_state(y, groups=None, states=None):
    """Map each state to the array of its segment lengths (in windows)."""
    labels, lengths = run_lengths(y, groups)
    if states is None:
        states = np.unique(y)
    return {s: lengths[labels == s] for s in states}


def split_by_group(y, groups=None):
    """Split ``y`` into one array per contiguous group."""
    y = np.asarray(y)
    if groups is None:
        return [y]
    cuts = np.flatnonzero(~_same_group(groups, len(y))) + 1
    return np.split(y, cuts)


def embedded_sequences(y, groups=None):
    """Per group, the sequence of distinct activities (consecutive repeats collapsed)."""
    return [run_lengths(seq)[0] for seq in split_by_group(y, groups)]


def transition_counts(y, n_states, groups=None, exclude_self=False):
    y = np.asarray(y)
    keep = _same_group(groups, len(y))
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (y[:-1][keep], y[1:][keep]), 1)
    if exclude_self:
        np.fill_diagonal(counts, 0)
    return counts


def normalize_rows(counts, pseudocount=0.0):
    counts = counts + pseudocount
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return counts / totals


@dataclass
class MarkovTestResult:
    n_triples: int
    g_statistic: float
    dof: int
    p_value: float

    @property
    def is_first_order(self):
        return self.p_value >= 0.05


def markov_order_test(sequences):
    """Likelihood-ratio (G) test of a first-order against a second-order Markov chain.

    H0: P(S_t | S_{t-1}, S_{t-2}) = P(S_t | S_{t-1}). Both models are evaluated
    on the same triples (s_{t-2}, s_{t-1}, s_t) pooled over ``sequences``, giving

        G = 2 sum_{ijk} n_ijk log( n_ijk n_j / (n_ij n_jk) ),

    the conditional-independence statistic of Anderson and Goodman (1957).
    Degrees of freedom count only transitions that are actually observed, so
    structurally impossible transitions do not inflate them.
    """
    triples = [np.stack([s[:-2], s[1:-1], s[2:]], axis=1) for s in sequences if len(s) >= 3]
    triples = np.concatenate(triples)
    n = int(triples.max()) + 1
    counts = np.zeros((n, n, n))
    np.add.at(counts, tuple(triples.T), 1)

    n_ij = counts.sum(axis=2)
    n_jk = counts.sum(axis=0)
    n_j = counts.sum(axis=(0, 2))

    i, j, k = np.nonzero(counts)
    g = 2 * np.sum(counts[i, j, k] * np.log(counts[i, j, k] * n_j[j] / (n_ij[i, j] * n_jk[j, k])))

    # Per middle state j: (rows observed - 1) * (columns observed - 1), restricted to
    # the observed cells; summed over j.
    dof = 0
    for m in range(n):
        dof += (np.count_nonzero(counts[:, m, :]) - np.count_nonzero(n_ij[:, m])
                - np.count_nonzero(n_jk[m]) + (n_j[m] > 0))
    dof = int(max(dof, 1))
    return MarkovTestResult(len(triples), g, dof, stats.chi2.sf(g, dof))
