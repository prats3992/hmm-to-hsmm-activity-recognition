"""Class-conditional HMM emission models.

Each activity gets its own small HMM (a Gaussian HMM or a GMM-HMM) fit on that
activity's windows. The models act as emission densities P(x_t | activity) for
frame-level classification and for the Viterbi and HSMM decoders.
"""

import numpy as np
from hmmlearn import hmm

from har.config import N_MIXTURES, N_SUBSTATES, RANDOM_STATE
from har.sequences import segment_bounds


def fit_class_models(X, y, subjects, emission="gmm", n_substates=N_SUBSTATES,
                     n_mix=N_MIXTURES, n_iter=100):
    """Fit one HMM per activity label.

    ``emission`` is ``"gaussian"`` for a single Gaussian per sub-state or
    ``"gmm"`` for a Gaussian mixture with ``n_mix`` components per sub-state.
    Each contiguous segment of an activity (within one subject) is treated as a
    separate observation sequence. Returns a dict mapping label index to fitted model.
    """
    starts, ends = segment_bounds(y, subjects)
    models = {}
    for label in np.unique(y):
        mask = y == label
        lengths = (ends - starts)[y[starts] == label]
        if emission == "gaussian":
            model = hmm.GaussianHMM(n_components=n_substates, covariance_type="diag",
                                    n_iter=n_iter, random_state=RANDOM_STATE)
        elif emission == "gmm":
            model = hmm.GMMHMM(n_components=n_substates, n_mix=n_mix, covariance_type="diag",
                               n_iter=n_iter, random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown emission type: {emission!r}")
        model.fit(X[mask], lengths)
        models[label] = model
    return models


def frame_log_likelihoods(models, X):
    """Return a ``(T, n_classes)`` matrix of per-window log-likelihoods log P(x_t | class)."""
    log_lik = np.empty((len(X), len(models)))
    for label, model in models.items():
        log_lik[:, label] = [model.score(x.reshape(1, -1)) for x in X]
    return log_lik
