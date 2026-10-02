import numpy as np

from har.decoding import embedded_transition_log_probs, hsmm_viterbi, viterbi


def _noisy_emissions(true, n_states, flip_rate, rng):
    """Log-emissions that favour the true state, with a fraction of windows misleading."""
    log_em = np.log(np.full((len(true), n_states), 0.2))
    log_em[np.arange(len(true)), true] = np.log(0.6)
    flip = rng.random(len(true)) < flip_rate
    log_em[flip] = log_em[flip][:, ::-1]
    return log_em


def test_viterbi_recovers_clean_sequence():
    true = np.repeat([0, 1, 2], 10)
    log_em = _noisy_emissions(true, 3, 0.0, np.random.default_rng(0))
    log_trans = np.log(np.full((3, 3), 0.05) + np.eye(3) * 0.85)
    path = viterbi(log_em, log_trans, np.log(np.full(3, 1 / 3)))
    assert (path == true).all()


def test_hsmm_viterbi_smooths_noisy_segments():
    rng = np.random.default_rng(0)
    true = np.repeat([0, 1, 2, 0, 1], [15, 20, 25, 20, 12])
    log_em = _noisy_emissions(true, 3, 0.3, rng)
    log_trans = np.log(np.full((3, 3), 0.5))
    np.fill_diagonal(log_trans, -np.inf)
    durations = {j: (20.0, 6.0) for j in range(3)}

    path = hsmm_viterbi(log_em, log_trans, durations, max_dur=50)
    assert (path == true).mean() > 0.9
    assert (path == true).mean() > (log_em.argmax(axis=1) == true).mean()


def test_hsmm_viterbi_single_segment():
    # The whole sequence is one segment, so it must be decoded from the initial state alone.
    true = np.zeros(30, dtype=int)
    log_em = _noisy_emissions(true, 3, 0.0, np.random.default_rng(0))
    log_trans = np.log(np.full((3, 3), 0.5))
    np.fill_diagonal(log_trans, -np.inf)
    path = hsmm_viterbi(log_em, log_trans, {j: (30.0, 5.0) for j in range(3)}, max_dur=50)
    assert (path == 0).all()


def test_embedded_transitions_forbid_self_loops():
    y = np.array([0, 0, 1, 1, 2, 0])
    log_trans = embedded_transition_log_probs(y, 3)
    assert np.isneginf(np.diag(log_trans)).all()
    assert np.allclose(np.exp(log_trans).sum(axis=1), 1)
