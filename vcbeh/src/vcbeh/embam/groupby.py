"""Replacement for EMBAM's ``apply_by_index``.

Every call site in this project used ``dim=1`` and passed trailing scalars, so
the cell-array/``varargin``/``dim`` machinery buys nothing.  What *does* need a
single documented home is the row-ordering contract: results are stacked in
ascending ``unique(index)`` order, and that ordering is the row key of every
matrix in ``res``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def group_apply(
    fn: Callable[..., np.ndarray],
    index: np.ndarray,
    arrays: Sequence[np.ndarray],
    *args,
) -> np.ndarray:
    """Apply ``fn`` to the rows of ``arrays`` grouped by ``index``.

    Groups are visited in ascending ``np.unique(index)`` order and each result
    is stacked as one row, so output row *i* corresponds to subject
    ``np.unique(index)[i]``.

    ``fn`` is called as ``fn(sub_array_1, ..., sub_array_k, *args)`` and must
    return a 1-D row (or a 2-D single row) of the same width for every group.
    """
    index = np.asarray(index).ravel()
    arrays = [np.asarray(a) for a in arrays]
    for k, a in enumerate(arrays):
        if a.shape[0] != index.shape[0]:
            raise ValueError(
                f"array {k} has {a.shape[0]} rows but index has {index.shape[0]}"
            )

    rows = []
    width = None
    for g in np.unique(index):
        sel = index == g
        row = np.atleast_2d(fn(*[a[sel] for a in arrays], *args))
        if width is None:
            width = row.shape[1]
        elif row.shape[1] != width:
            raise ValueError(
                f"group {g!r} produced width {row.shape[1]}, expected {width}"
            )
        rows.append(row)

    if not rows:
        raise ValueError("index is empty; nothing to group")
    return np.vstack(rows)
