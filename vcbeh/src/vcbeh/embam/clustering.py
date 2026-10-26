"""Temporal clustering factor (EMBAM ``temp_fact``).

Polyn, Norman & Kahana (2009).  For each recall transition, the actual lag is
ranked against the lags of all still-available items; 1.0 means the nearest
available item was chosen every time, 0.5 is chance.
"""

from __future__ import annotations

import numpy as np

from ._walk import candidate_mask, iter_transitions


def temp_fact(
    recalls: np.ndarray,
    subjects: np.ndarray,
    list_length: int,
    rec_mask: np.ndarray,
    pres_mask: np.ndarray,
) -> np.ndarray:
    """Temporal clustering score, one row per subject (shape ``(nsub, 1)``).

    Scores are pooled over **all transitions of all of a subject's trials** and
    averaged once -- not averaged per trial and then across trials.

    Ties use **midrank** scoring, ``(gt + (eq-1)/2) / (ncand - 1)``.  This was
    determined empirically: midrank reproduces the saved MATLAB values to
    2e-16, while both strict ``>`` and ``>=`` are off by 0.024.  ``eq`` is
    always at least 1 because the chosen transition is itself a candidate, so
    ``(eq-1)/2`` is half credit for the *other* equidistant candidates.

    Transitions with one or zero candidates are skipped entirely; a subject
    with no scored transitions yields NaN.
    """
    subjects = np.asarray(subjects).ravel()
    recalls = np.asarray(recalls)
    rec_mask = np.asarray(rec_mask, dtype=bool)
    pres_mask = np.asarray(pres_mask, dtype=bool)
    ll = int(list_length)
    positions = np.arange(1, ll + 1)

    subs = np.unique(subjects)
    out = np.full((subs.size, 1), np.nan)

    for i, s in enumerate(subs):
        idx = np.nonzero(subjects == s)[0]
        total = 0.0
        nscored = 0

        for t, a, b, already in iter_transitions(recalls, idx, rec_mask, rec_mask, ll):
            cand = candidate_mask(pres_mask[t], already, a, ll)
            ncand = int(cand.sum())
            if ncand <= 1:
                continue
            dists = np.abs(positions[cand] - a)
            actual = abs(b - a)
            greater = int((dists > actual).sum())
            equal = int((dists == actual).sum())
            total += (greater + (equal - 1) / 2) / (ncand - 1)
            nscored += 1

        if nscored:
            out[i, 0] = total / nscored

    return out
