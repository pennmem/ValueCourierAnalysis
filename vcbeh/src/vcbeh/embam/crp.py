"""Lag conditional response probability (EMBAM ``crp``).

Kahana (1996).  The output is ``2*list_length - 1`` columns wide, spanning lags
``-(ll-1) .. +(ll-1)``; use `lag_index` / `lag_axis` rather than arithmetic on
raw column numbers -- MATLAB's ``crps(:, lag+ll)`` becomes a *negative* index in
Python and would silently wrap to the end of the array.
"""

from __future__ import annotations

import numpy as np

from ._walk import candidate_mask, iter_transitions


def lag_index(lag: int | np.ndarray, list_length: int) -> int | np.ndarray:
    """Column holding ``lag`` in a CRP matrix (0-based)."""
    return lag + int(list_length) - 1


def lag_axis(list_length: int) -> np.ndarray:
    """The lags labelling each CRP column, ``-(ll-1) .. +(ll-1)``."""
    ll = int(list_length)
    return np.arange(-(ll - 1), ll)


def crp(
    recalls: np.ndarray,
    subjects: np.ndarray,
    list_length: int,
    from_mask_rec: np.ndarray,
    to_mask_rec: np.ndarray,
    from_mask_pres: np.ndarray,
    to_mask_pres: np.ndarray,
) -> np.ndarray:
    """Conditional response probability by lag, one row per subject.

    Actual and possible transition counts are pooled across **all of a
    subject's trials first** and divided once at the end -- not averaged over
    per-trial ratios.  Lag 0 is always NaN.

    ``from_mask_pres`` is accepted for signature compatibility with the MATLAB
    original but is unused: only ``to_mask_pres`` gates the denominator.
    """
    subjects = np.asarray(subjects).ravel()
    recalls = np.asarray(recalls)
    from_mask_rec = np.asarray(from_mask_rec, dtype=bool)
    to_mask_rec = np.asarray(to_mask_rec, dtype=bool)
    to_mask_pres = np.asarray(to_mask_pres, dtype=bool)
    ll = int(list_length)
    nbins = 2 * ll - 1
    positions = np.arange(1, ll + 1)

    subs = np.unique(subjects)
    out = np.full((subs.size, nbins), np.nan)

    for i, s in enumerate(subs):
        idx = np.nonzero(subjects == s)[0]
        actual = np.zeros(nbins)
        possible = np.zeros(nbins)

        for t, a, b, already in iter_transitions(
            recalls, idx, from_mask_rec, to_mask_rec, ll
        ):
            actual[b - a + ll - 1] += 1
            cand = candidate_mask(to_mask_pres[t], already, a, ll)
            if cand.any():
                possible += np.bincount(
                    positions[cand] - a + ll - 1, minlength=nbins
                )

        nz = possible > 0
        out[i, nz] = actual[nz] / possible[nz]
        out[i, ll - 1] = np.nan  # lag 0 is undefined

    return out
