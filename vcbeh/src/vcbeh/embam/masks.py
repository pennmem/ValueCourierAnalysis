"""Recall masks (EMBAM ``make_clean_recalls_mask2d``)."""

from __future__ import annotations

import numpy as np


def make_clean_recalls_mask2d(recalls: np.ndarray) -> np.ndarray:
    """True at each *first* correct recall of a serial position.

    An entry qualifies when it is a real serial position (``> 0``; ``0`` is
    padding and ``-1`` an intrusion) **and** that position has not already
    appeared earlier in the same row.  Repeats keep the first occurrence and
    drop every later one.
    """
    recalls = np.asarray(recalls)
    if recalls.ndim != 2:
        raise ValueError(f"recalls must be 2-D, got shape {recalls.shape}")
    ntrials, ncols = recalls.shape

    mask = np.zeros((ntrials, ncols), dtype=bool)
    max_sp = int(recalls.max()) if recalls.size else 0
    seen = np.zeros((ntrials, max(max_sp, 0) + 1), dtype=bool)
    rows = np.arange(ntrials)

    for j in range(ncols):
        col = recalls[:, j]
        positive = col > 0
        already = np.zeros(ntrials, dtype=bool)
        already[positive] = seen[rows[positive], col[positive]]
        ok = positive & ~already
        mask[:, j] = ok
        seen[rows[ok], col[ok]] = True

    return mask
