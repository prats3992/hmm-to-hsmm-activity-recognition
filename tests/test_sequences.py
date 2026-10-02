import numpy as np

from har.sequences import (durations_by_state, embedded_sequences, markov_order_test,
                           run_lengths, transition_counts)


def test_runs_break_at_subject_boundaries():
    y = np.array([0, 0, 1, 1, 1, 1, 2])
    subjects = np.array([1, 1, 1, 2, 2, 2, 2])
    labels, lengths = run_lengths(y, subjects)
    assert labels.tolist() == [0, 1, 1, 2]
    assert lengths.tolist() == [2, 1, 3, 1]
    assert durations_by_state(y, subjects)[1].tolist() == [1, 3]


def test_transitions_ignore_subject_boundaries():
    y = np.array([0, 1, 1, 0])
    subjects = np.array([1, 1, 2, 2])
    counts = transition_counts(y, 2, subjects)
    assert counts.tolist() == [[0, 1], [1, 0]]


def test_embedded_sequences_per_subject():
    y = np.array([0, 0, 1, 1, 2, 2])
    subjects = np.array([1, 1, 1, 2, 2, 2])
    assert [s.tolist() for s in embedded_sequences(y, subjects)] == [[0, 1], [1, 2]]


def _simulate(next_state, n, rng):
    s = [0, 1]
    for _ in range(n):
        s.append(next_state(s, rng))
    return np.array(s)


def test_markov_test_accepts_first_order_chain():
    rng = np.random.default_rng(0)
    P = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.3, 0.3, 0.4]])
    seq = _simulate(lambda s, r: r.choice(3, p=P[s[-1]]), 5000, rng)
    assert markov_order_test([seq]).p_value > 0.05


def test_markov_test_rejects_second_order_chain():
    rng = np.random.default_rng(0)
    seq = _simulate(lambda s, r: (s[-1] + s[-2]) % 3 if r.random() < 0.7 else r.integers(3),
                    5000, rng)
    assert markov_order_test([seq]).p_value < 1e-6
