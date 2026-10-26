"""Serial position curve and probability of first recall (EMBAM ``spc``, ``pfr``)."""

from __future__ import annotations

import numpy as np


def spc(
    recalls: np.ndarray,
    subjects: np.ndarray,
    list_length: int,
    rec_mask: np.ndarray,
    pres_mask: np.ndarray,
) -> np.ndarray:
    """Recall probability per serial position, one row per subject.

    For subject *s* and serial position *p*::

        numerator   = number of *cells* where recalls == p and rec_mask
        denominator = number of trials where pres_mask[:, p-1]
        result      = numerator / denominator,  or NaN when denominator == 0

    Counting cells rather than trials only coincides with the usual definition
    because the clean-recalls mask removes repeats.  It matters here because
    ``prec_byval`` passes a value-filtered mask, and the ``denominator == 0 ->
    NaN`` branch is what lets its ``nanmean`` skip values a subject never saw.
    """
    subjects = np.asarray(subjects).ravel()
    recalls = np.asarray(recalls)
    rec_mask = np.asarray(rec_mask, dtype=bool)
    pres_mask = np.asarray(pres_mask, dtype=bool)
    ll = int(list_length)

    subs = np.unique(subjects)
    out = np.full((subs.size, ll), np.nan)

    for i, s in enumerate(subs):
        sel = subjects == s
        counted = np.where(rec_mask[sel] & (recalls[sel] > 0), recalls[sel], 0)
        num = np.bincount(counted.ravel(), minlength=ll + 1)[1 : ll + 1]
        den = pres_mask[sel].sum(axis=0)
        nz = den > 0
        out[i, nz] = num[nz] / den[nz]

    return out


def pfr(
    recalls: np.ndarray,
    subjects: np.ndarray,
    list_length: int,
    rec_mask: np.ndarray,
    pres_mask: np.ndarray,
) -> np.ndarray:
    """Probability of first recall per serial position, one row per subject.

    "First" means **output position 1 literally**, not the first valid recall.
    A trial whose first output is an intrusion therefore contributes to the
    denominator but to no numerator, so rows can sum to less than 1 (observed
    range in this dataset: 0.78 to 1.0).
    """
    subjects = np.asarray(subjects).ravel()
    recalls = np.asarray(recalls)
    rec_mask = np.asarray(rec_mask, dtype=bool)
    pres_mask = np.asarray(pres_mask, dtype=bool)
    ll = int(list_length)

    subs = np.unique(subjects)
    out = np.full((subs.size, ll), np.nan)

    for i, s in enumerate(subs):
        sel = subjects == s
        first = recalls[sel, 0]
        ok = rec_mask[sel, 0] & (first > 0)
        num = np.bincount(np.where(ok, first, 0), minlength=ll + 1)[1 : ll + 1]
        den = pres_mask[sel].sum(axis=0)
        nz = den > 0
        out[i, nz] = num[nz] / den[nz]

    return out
