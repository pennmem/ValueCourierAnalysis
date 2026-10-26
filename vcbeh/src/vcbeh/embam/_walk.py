"""The output-order transition walk shared by ``crp`` and ``temp_fact``.

Both measures iterate adjacent recall pairs and, for each, ask which serial
positions were still available to transition *to*.  The rules are subtle enough
that duplicating them would be a real risk, so they live here once.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_transitions(
    recalls: np.ndarray,
    trial_indices: np.ndarray,
    from_mask: np.ndarray,
    to_mask: np.ndarray,
    list_length: int,
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Yield ``(trial, from_sp, to_sp, already_output)`` per scored transition.

    ``already_output`` is a boolean array indexed by serial position (``1..ll``)
    marking positions output earlier in the trial.  It is **reused and mutated
    between yields**, so consume it before advancing the iterator.

    Two rules that are easy to get wrong, both required to match MATLAB:

    * ``already_output`` is updated for *any* positive entry, even when the
      transition was skipped because an endpoint failed its mask, and even when
      the entry was itself a masked-out repeat.
    * A transition is scored only when the ``from`` end passes ``from_mask``
      *and* the ``to`` end passes ``to_mask``; otherwise it contributes nothing
      to either the numerator or the denominator.
    """
    ncols = recalls.shape[1]
    ll = int(list_length)

    for t in trial_indices:
        t = int(t)
        row = recalls[t]
        already = np.zeros(ll + 1, dtype=bool)
        for j in range(ncols - 1):
            a = int(row[j])
            b = int(row[j + 1])
            if from_mask[t, j] and to_mask[t, j + 1]:
                yield t, a, b, already
            if 0 < a <= ll:
                already[a] = True


def candidate_mask(
    to_pres_row: np.ndarray,
    already: np.ndarray,
    from_sp: int,
    list_length: int,
) -> np.ndarray:
    """Serial positions still available as transition targets, as a bool array.

    A position counts when it was presented, has not already been output, and
    is not the item being transitioned *from*.  Note the last two together
    exclude every previously output position *including* the current ``from``
    item, which has not yet been added to ``already``.
    """
    ll = int(list_length)
    positions = np.arange(1, ll + 1)
    return to_pres_row[:ll] & ~already[1 : ll + 1] & (positions != from_sp)
