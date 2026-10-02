"""Viterbi decoding for the HMM and duration-explicit Viterbi decoding for the HSMM."""

import numpy as np
from scipy import stats

from har.sequences import durations_by_state, normalize_rows, transition_counts


def frame_transition_log_probs(y, n_states, groups=None):
    """Frame-level transition matrix with add-one smoothing (standard HMM)."""
    counts = transition_counts(y, n_states, groups)
    return np.log(normalize_rows(counts, pseudocount=1.0))


def embedded_transition_log_probs(y, n_states, groups=None, pseudocount=1e-5):
    """Transition matrix of the embedded chain, i.e. given that the activity changes (HSMM).

    Self-transitions are impossible by construction, so the diagonal is ``-inf``.
    """
    counts = transition_counts(y, n_states, groups, exclude_self=True)
    counts = counts + pseudocount
    np.fill_diagonal(counts, 0)
    with np.errstate(divide="ignore"):
        return np.log(normalize_rows(counts))


def viterbi(log_emissions, log_trans, log_start):
    """Most likely state sequence under a first-order HMM."""
    T, N = log_emissions.shape
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0] = log_start + log_emissions[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(N)] + log_emissions[t]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def fit_duration_params(y, n_states, groups=None):
    """Mean and standard deviation of segment length for each state."""
    params = {}
    for state, durs in durations_by_state(y, groups, range(n_states)).items():
        std = np.std(durs) if len(durs) >= 2 else 1.0
        params[state] = (np.mean(durs), std)
    return params


def duration_log_probs(duration_params, max_dur):
    """Discretised Gaussian duration distribution truncated to ``1..max_dur``.

    Returns an ``(N, max_dur + 1)`` array; column 0 (zero duration) is ``-inf``.
    """
    n_states = len(duration_params)
    log_probs = np.full((n_states, max_dur + 1), -np.inf)
    d = np.arange(1, max_dur + 1)
    for j in range(n_states):
        mu, std = duration_params[j]
        probs = stats.norm.pdf(d, loc=mu, scale=max(std, 1e-3))
        probs = probs / probs.sum()
        with np.errstate(divide="ignore"):
            log_probs[j, 1:] = np.log(probs)
    return log_probs


def hsmm_viterbi(log_emissions, log_trans, duration_params, max_dur):
    """Duration-explicit (segment-level) Viterbi decoding for a hidden semi-Markov model.

    ``delta[t, j]`` is the best log-score of a segmentation of windows ``0..t``
    whose last segment is in state ``j`` and ends at ``t``. A segment of length
    ``d`` either starts the sequence (uniform initial distribution) or follows a
    segment of another state ending at ``t - d``:

        delta[t, j] = max_d log p_j(d) + sum_{tau = t-d+1}^{t} log b_j(x_tau)
                      + ( -log N                               if d = t + 1
                          max_i delta[t - d, i] + log a_ij     otherwise )

    The first and last segments may be cut off by the start or end of the
    recording, so their durations are scored with the survival function
    P(D_j >= d) instead of p_j(d).
    """
    T, N = log_emissions.shape
    log_dur = duration_log_probs(duration_params, max_dur)
    # log P(D >= d), used for segments censored by the start or end of the recording.
    with np.errstate(divide="ignore"):
        log_surv = np.log(np.cumsum(np.exp(log_dur[:, ::-1]), axis=1)[:, ::-1])
    cum_emissions = np.vstack([np.zeros(N), np.cumsum(log_emissions, axis=0)])

    delta = np.full((T, N), -np.inf)
    psi_state = np.full((T, N), -1)
    psi_dur = np.zeros((T, N), dtype=int)

    for t in range(T):
        for d in range(1, min(max_dur, t + 1) + 1):
            start = t - d + 1
            emission = cum_emissions[t + 1] - cum_emissions[start]
            if start == 0:
                prev, prev_state = np.full(N, -np.log(N)), np.full(N, -1)
            else:
                scores = delta[start - 1][:, None] + log_trans
                prev_state = np.argmax(scores, axis=0)
                prev = scores[prev_state, np.arange(N)]
            censored = start == 0 or t == T - 1
            total = prev + (log_surv if censored else log_dur)[:, d] + emission
            better = total > delta[t]
            delta[t, better] = total[better]
            psi_state[t, better] = prev_state[better]
            psi_dur[t, better] = d

    path = np.zeros(T, dtype=int)
    state, t = int(np.argmax(delta[-1])), T - 1
    while t >= 0:
        d = psi_dur[t, state]
        path[t - d + 1:t + 1] = state
        state, t = psi_state[t, state], t - d
    return path
