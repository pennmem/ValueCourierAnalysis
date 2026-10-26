"""Per-list value aggregation helpers (lab-local MATLAB equivalents).

Neither of these functions was available on disk. ``value_from_mask`` was
recovered by fitting candidate definitions against the saved
res_083026_xs0.mat; see scripts/reverse_value_from_mask.py for that derivation.
"""

import numpy as np

from .masks import unique_subjects


def value_from_mask(values, subjects, mask, maxnlist):
    """Per-subject, per-list weighted mean of ``values`` under ``mask``.

    The mask is not required to be boolean: ana_vavi_e2frop.m passes integer
    output-position weights, in which case this is a weighted mean with the
    mask entries as weights.

    Masked-out entries are excluded rather than multiplied by zero, because
    ``rec_values_act``/``rec_values_est`` carry NaN outside the recalled slots
    and ``NaN * 0`` would propagate.

    A NaN *inside* the mask contributes zero to the numerator but still counts
    in the denominator, matching MATLAB's ``sum(...,'omitnan')`` over the
    masked values divided by the full mask weight. This is not the same as a
    nanmean: ``rec_values_est`` is NaN for recalled items the participant left
    unestimated, and those lists are averaged as if the estimate were 0.

    An empty mask for a list yields ``inf``, a sentinel the caller uses to tell
    "this list had no recalls" apart from "this subject had no such list"
    (which stays NaN). See ana_vavi_e2frop.m:155-177, which converts these Infs
    to NaN once it has checked that they line up across measures.

    Parameters
    ----------
    values : (n_trials, n_cols) array
    subjects : (n_trials, 1) array
    mask : (n_trials, n_cols) array
        Boolean or numeric weights.
    maxnlist : int
        Column count of the output; lists beyond a subject's count stay NaN.

    Returns
    -------
    (n_subjects, maxnlist) array
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    subjects = np.asarray(subjects).ravel()

    subs = unique_subjects(subjects)
    out = np.full((len(subs), maxnlist), np.nan)

    for si, subj in enumerate(subs):
        rows = np.flatnonzero(subjects == subj)
        if rows.size > maxnlist:
            raise ValueError(
                f"subject {subj} has {rows.size} lists, exceeding maxnlist={maxnlist}"
            )
        for li, row in enumerate(rows):
            weights = mask[row]
            denom = weights.sum()
            if denom == 0:
                out[si, li] = np.inf
                continue
            kept = np.where(
                (weights != 0) & ~np.isnan(values[row]), values[row], 0.0
            )
            out[si, li] = np.sum(kept * weights) / denom
    return out


def mean_abserr_sess(abserr_list, session_matrix, sublist, maxnsess):
    """Average a per-list measure within each session.

    Parameters
    ----------
    abserr_list : (n_subjects, maxnlist) array
        Per-list values, NaN where the subject has no such list.
    session_matrix : (n_subjects, maxnlist) array
        Session number of each list, NaN-padded to match.
    sublist : array
        Subject identifiers; only its length is used.
    maxnsess : int
        Column count of the output.

    Returns
    -------
    (n_subjects, maxnsess) array

    Notes
    -----
    Sessions are numbered **per subject and positionally**: column ``j`` holds
    a subject's j-th distinct session in ascending order, not "session j".
    A subject missing a session therefore has its later sessions shifted left
    relative to other subjects. That is what the MATLAB helper does, and the
    saved results were produced with it.

    The average is a plain mean, not a NaN-omitting one, so a single undefined
    list (e.g. one with no recalls at all) makes that whole session NaN.
    """
    abserr_list = np.asarray(abserr_list, dtype=np.float64)
    session_matrix = np.asarray(session_matrix, dtype=np.float64)

    out = np.full((len(sublist), maxnsess), np.nan)
    for si in range(len(sublist)):
        sess_row = session_matrix[si]
        sessions = np.unique(sess_row[~np.isnan(sess_row)])
        if len(sessions) > maxnsess:
            raise ValueError(
                f"subject index {si} has {len(sessions)} sessions, "
                f"exceeding maxnsess={maxnsess}"
            )
        for sj, sess in enumerate(sessions):
            picked = abserr_list[si][sess_row == sess]
            if picked.size:
                out[si, sj] = picked.mean()
    return out
