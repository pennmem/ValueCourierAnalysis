"""Recall masks and by-subject dispatch (EMBAM equivalents)."""

import numpy as np


def make_clean_recalls_mask2d(recalls):
    """Mark entries that are correct, non-repeated recalls.

    An entry is kept when it is a positive serial position (so intrusions,
    coded <= 0, are dropped) and that serial position has not already appeared
    earlier in the same row.

    Parameters
    ----------
    recalls : (n_trials, n_ops) array
        Serial position recalled at each output position, 1-based; 0 for empty
        slots and negative for intrusions.

    Returns
    -------
    (n_trials, n_ops) bool array
    """
    recalls = np.asarray(recalls)
    mask = np.zeros(recalls.shape, dtype=bool)

    for row in range(recalls.shape[0]):
        seen = set()
        for col in range(recalls.shape[1]):
            sp = recalls[row, col]
            if sp > 0 and sp not in seen:
                seen.add(sp)
                mask[row, col] = True
    return mask


def unique_subjects(subjects):
    """Sorted unique subject identifiers, matching MATLAB's ``unique``."""
    return np.unique(np.asarray(subjects).ravel())


def apply_by_index(func, subjects, _dim, mats, *extra):
    """Split rows by subject, apply ``func``, stack the results.

    Mirrors EMBAM's ``apply_by_index``. ``_dim`` exists only to keep the call
    signature recognizable against the MATLAB source; rows are always the split
    dimension here.

    ``func`` receives the per-subject slice of each array in ``mats``, followed
    by ``extra``, and returns either a scalar or a 1-D vector. Results are
    stacked into ``(n_subjects, ...)``.
    """
    subjects = np.asarray(subjects).ravel()
    out = []
    for subj in unique_subjects(subjects):
        rows = subjects == subj
        sliced = [np.asarray(m)[rows] for m in mats]
        out.append(np.asarray(func(*sliced, *extra), dtype=np.float64))

    stacked = np.stack(out, axis=0)
    # A scalar-returning func gives (n_subjects, 1); flatten to (n_subjects,)
    # so it lines up with how MATLAB stores e.g. temp_clusts.
    if stacked.ndim == 2 and stacked.shape[1] == 1:
        stacked = stacked.ravel()
    return stacked
