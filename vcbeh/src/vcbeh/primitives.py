"""Project-specific per-subject reductions.

Ports of ``value_from_mask.m``, ``value_from_mask_weightedop.m``,
``mean_abserr_sess.m`` and ``intru_types.m``.
"""

from __future__ import annotations

import numpy as np

from .embam.groupby import group_apply

# Sentinel written when a list has no valid items, kept deliberately distinct
# from NaN so "no recalls" can be told apart from "undefined for another
# reason".  ana_vavi converts these to NaN only after checking they align.
NO_ITEMS = np.inf


def value_from_mask(
    values_matrix: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
    maxnlist: int,
) -> np.ndarray:
    """Mask-weighted average value per list; ``(nsub, maxnlist)``.

    Each list contributes ``sum(values * mask) / sum(mask)``.  The mask may be
    boolean or **weighted** (``ana_vavi`` passes output-position weights), so
    the denominator is a sum of weights, not a count.  NaN values are dropped
    from the numerator but the denominator still counts their weight, matching
    MATLAB's ``sum(..., "omitnan")`` over the product.

    Lists beyond a subject's list count stay NaN; a list whose mask sums to
    zero gets `NO_ITEMS` (Inf).
    """
    return group_apply(
        _vfm_for_subj, subjects, (values_matrix, mask), int(maxnlist)
    )


def _vfm_for_subj(
    values_matrix: np.ndarray, mask: np.ndarray, maxnlist: int
) -> np.ndarray:
    out = np.full(maxnlist, np.nan)
    nlists = values_matrix.shape[0]
    if nlists > maxnlist:
        raise ValueError(f"subject has {nlists} lists, maxnlist is {maxnlist}")

    weights = np.asarray(mask, dtype=np.float64)
    values = np.asarray(values_matrix, dtype=np.float64)
    for li in range(nlists):
        w = weights[li]
        denom = w.sum()
        out[li] = np.nansum(values[li] * w) / denom if denom else NO_ITEMS
    return out


def value_from_mask_weightedop(
    values_matrix: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
    maxnlist: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean-mask average value per list; ``(nsub, maxnlist)``.

    The MATLAB original took a ``weights`` argument that its inner function
    never read, so passing one is rejected rather than silently ignored.  Use
    `value_from_mask` with a weighted mask for output-position weighting.
    """
    if weights is not None:
        raise TypeError(
            "the `weights` argument was dead code in the MATLAB original; "
            "pass a weighted mask to value_from_mask instead"
        )
    mask = np.asarray(mask, dtype=bool)
    return group_apply(_vfm_bool_for_subj, subjects, (values_matrix, mask), int(maxnlist))


def _vfm_bool_for_subj(
    values_matrix: np.ndarray, mask: np.ndarray, maxnlist: int
) -> np.ndarray:
    out = np.full(maxnlist, np.nan)
    for li in range(values_matrix.shape[0]):
        sel = mask[li]
        n = int(sel.sum())
        out[li] = values_matrix[li][sel].sum() / n if n else NO_ITEMS
    return out


def mean_abserr_sess(
    err_matrix: np.ndarray,
    sess_matrix: np.ndarray,
    subjects: np.ndarray,
    maxnsess: int,
) -> np.ndarray:
    """Average a per-list quantity within each session; ``(nsub, maxnsess)``.

    Columns are session **ordinals**, not session numbers: a subject missing
    session 3 has their session-4 mean land in column 3.

    The average is a plain mean, so a session containing any NaN list yields
    NaN.  That is the MATLAB behaviour and it is what the saved results
    contain -- do not substitute nanmean.
    """
    return group_apply(
        _err_for_subj, subjects, (err_matrix, sess_matrix), int(maxnsess)
    )


def _err_for_subj(
    err_vec: np.ndarray, sess_vec: np.ndarray, maxnsess: int
) -> np.ndarray:
    out = np.full(maxnsess, np.nan)
    err = np.asarray(err_vec).ravel()
    sess = np.asarray(sess_vec, dtype=np.float64).ravel()
    sesslist = np.unique(sess[~np.isnan(sess)])
    for i, s in enumerate(sesslist):
        if i >= maxnsess:
            raise ValueError(f"subject has more than {maxnsess} sessions")
        out[i] = err[sess == s].mean()
    return out


def intru_types(
    rec_itemnos: np.ndarray,
    intrusions: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
    list_length: int,
) -> np.ndarray:
    """Intrusion rates per subject as ``(nsub, 3)``: WLI, PLI, ELI.

    Encoding, per the MATLAB original:

    * **WLI** (within-list): ``intrusions == -1`` and ``rec_itemnos > 0``
    * **PLI** (prior-list):  ``intrusions > 0``
    * **ELI** (extra-list):  ``intrusions == -1`` and ``rec_itemnos == -1``

    Each count is divided by the number of words the subject studied
    (``ntrials * list_length``).
    """
    shapes = {a.shape for a in (rec_itemnos, intrusions, mask)}
    if len(shapes) != 1:
        raise ValueError(f"matrix size misalignment: {shapes}")
    return group_apply(
        _int_for_subj,
        subjects,
        (rec_itemnos, intrusions, np.asarray(mask, dtype=bool)),
        int(list_length),
    )


def _int_for_subj(
    rec_itemnos: np.ndarray,
    intrusions: np.ndarray,
    mask: np.ndarray,
    list_length: int,
) -> np.ndarray:
    denom = mask.shape[0] * list_length
    wli = np.count_nonzero(mask & (rec_itemnos > 0) & (intrusions == -1))
    pli = np.count_nonzero(mask & (intrusions > 0))
    eli = np.count_nonzero(mask & (rec_itemnos == -1) & (intrusions == -1))
    return np.array([wli, pli, eli], dtype=np.float64) / denom
