"""Recall-probability and transition analyses (EMBAM equivalents).

Serial positions are kept 1-based throughout, matching how they are stored in
``recalls``; the conversion to 0-based happens only at the point of indexing.
"""

import numpy as np

from .masks import apply_by_index


# ------------------------------------------------------------------ helpers


def _recalled_positions(recalls_row, mask_row, list_length):
    """Set of studied serial positions the row recalled under the mask."""
    return {
        int(recalls_row[col])
        for col in range(len(recalls_row))
        if mask_row[col] and 1 <= recalls_row[col] <= list_length
    }


# ---------------------------------------------------------------------- spc


def _spc_for_subj(recalls, mask, pres_mask, list_length):
    counts = np.zeros(list_length)
    for tri in range(recalls.shape[0]):
        for sp in _recalled_positions(recalls[tri], mask[tri], list_length):
            counts[sp - 1] += 1  # 1-based SP -> 0-based column

    # The denominator is the number of lists in which that serial position was
    # PRESENTED, not simply the trial count. With an all-true pres_mask the two
    # agree, which is every call in ana_vavi_e2frop.m; they diverge when a
    # value-filtered mask is passed, which is what prec_byval does.
    denom = pres_mask.sum(axis=0).astype(np.float64)
    out = np.full(list_length, np.nan)
    nz = denom > 0
    out[nz] = counts[nz] / denom[nz]
    return out


def spc(recalls, subjects, list_length, rec_mask, pres_mask):
    """Serial position curve.

    Returns
    -------
    (n_subjects, list_length) array
        Recalls at each serial position over the number of lists in which that
        position was presented, per ``pres_mask``. NaN where the denominator is
        zero -- which is what lets prec_byval average over positions a subject
        never saw at a given value.
    """
    return apply_by_index(
        _spc_for_subj, subjects, 1, (recalls, rec_mask, pres_mask), list_length
    )


# ---------------------------------------------------------------------- pfr


def _pfr_for_subj(recalls, mask, pres_mask, list_length):
    counts = np.zeros(list_length)
    for tri in range(recalls.shape[0]):
        # Output position 1 only. A list whose first response is an intrusion
        # contributes to the denominator but to no serial position -- it is not
        # credited to whatever correct item came next.
        sp = int(recalls[tri, 0])
        if mask[tri, 0] and 1 <= sp <= list_length:
            counts[sp - 1] += 1

    denom = pres_mask.sum(axis=0).astype(np.float64)
    out = np.full(list_length, np.nan)
    nz = denom > 0
    out[nz] = counts[nz] / denom[nz]
    return out


def pfr(recalls, subjects, list_length, rec_mask, pres_mask):
    """Probability of first recall.

    Returns
    -------
    (n_subjects, list_length) array
        Proportion of the subject's lists in which each serial position was the
        first correct recall. Lists whose first correct recall does not exist
        (no correct recalls at all) contribute to the denominator only.
    """
    return apply_by_index(
        _pfr_for_subj, subjects, 1, (recalls, rec_mask, pres_mask), list_length
    )


# ---------------------------------------------------------------------- crp


def _crp_for_subj(
    recalls, from_mask, to_mask, list_length, pres_mask_from, pres_mask_to
):
    n_lags = 2 * list_length - 1
    actual = np.zeros(n_lags)
    possible = np.zeros(n_lags)

    for tri in range(recalls.shape[0]):
        row = recalls[tri]
        # Transitions are between adjacent output positions, gated by the
        # masks; a repeat or intrusion in between breaks the pair rather than
        # being skipped over.
        recalled = set()
        for col in range(len(row) - 1):
            sp_from = int(row[col])
            if 1 <= sp_from <= list_length and from_mask[tri, col]:
                recalled.add(sp_from)

                sp_to = int(row[col + 1])
                if to_mask[tri, col + 1] and 1 <= sp_to <= list_length:
                    available = [
                        sp
                        for sp in range(1, list_length + 1)
                        if sp not in recalled
                    ]
                    for sp in available:
                        possible[sp - sp_from + list_length - 1] += 1
                    if sp_to not in recalled:
                        actual[sp_to - sp_from + list_length - 1] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        out = actual / possible
    out[possible == 0] = np.nan
    out[list_length - 1] = np.nan  # lag 0 is undefined
    return out


def crp(
    recalls,
    subjects,
    list_length,
    rec_mask_from,
    rec_mask_to,
    pres_mask_from,
    pres_mask_to,
):
    """Lag conditional response probability.

    Returns
    -------
    (n_subjects, 2 * list_length - 1) array
        Column ``lag + list_length - 1`` holds that lag; the centre column
        (lag 0) is NaN.
    """
    return apply_by_index(
        _crp_for_subj,
        subjects,
        1,
        (recalls, rec_mask_from, rec_mask_to),
        list_length,
        pres_mask_from,
        pres_mask_to,
    )


# ---------------------------------------------------------------- temp_fact


def _temp_fact_for_subj(recalls, mask, list_length, pres_mask):
    scores = []
    for tri in range(recalls.shape[0]):
        row = recalls[tri]
        recalled = set()
        # Transitions are between ADJACENT output positions, as in crp: an
        # intrusion or repeat breaks the pair rather than being skipped over.
        for col in range(len(row) - 1):
            sp_from = int(row[col])
            if not (mask[tri, col] and 1 <= sp_from <= list_length):
                continue
            recalled.add(sp_from)

            sp_to = int(row[col + 1])
            if not (mask[tri, col + 1] and 1 <= sp_to <= list_length):
                continue
            if sp_to in recalled:
                continue

            available = [
                sp for sp in range(1, list_length + 1) if sp not in recalled
            ]
            if len(available) < 2:
                continue

            actual_lag = abs(sp_to - sp_from)
            other_lags = [abs(sp - sp_from) for sp in available if sp != sp_to]
            # Percentile rank among the lags to the still-available items,
            # oriented so a shorter-than-average lag scores high. Absolute lags
            # tie often (sp_from +/- k), and ties split evenly.
            greater = sum(1 for lag in other_lags if lag > actual_lag)
            tied = sum(1 for lag in other_lags if lag == actual_lag)
            scores.append((greater + 0.5 * tied) / len(other_lags))
    # Pooled over every transition the subject made, not averaged by list.
    return np.mean(scores) if scores else np.nan


def temp_fact(recalls, subjects, list_length, rec_mask, pres_mask):
    """Temporal clustering factor.

    Returns
    -------
    (n_subjects,) array
        Mean percentile rank of the lag actually transitioned to, among the
        lags to all still-available items. 0.5 is chance; higher means more
        temporally clustered recall.
    """
    return apply_by_index(
        _temp_fact_for_subj, subjects, 1, (recalls, rec_mask), list_length, pres_mask
    )
