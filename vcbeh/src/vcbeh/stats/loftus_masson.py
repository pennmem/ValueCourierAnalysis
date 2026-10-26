"""Loftus-Masson within-subject confidence intervals.

Port of ``LoftusMasson.m``.  Loftus & Masson (1994), *Psychonomic Bulletin &
Review* 1(4):476-490, following the Kahana lab's Pascal implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import t as student_t


def loftus_masson(data: np.ndarray) -> np.ndarray:
    """Within-subject 95% CI half-widths, one per column.

    Rows are subjects, columns are conditions (a within-subject design).
    Between-subject variance is removed by normalizing each subject to the
    grand mean before the interval is derived from the subject-by-condition
    interaction, so the intervals support comparisons *between columns* --
    unlike ordinary standard errors, which are inflated by between-subject
    differences.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a subjects-by-conditions matrix, got {data.shape}")
    nsubj_rows, ncond_cols = data.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        grand_mean = np.nanmean(data)
        subj_mean = np.nanmean(data, axis=1, keepdims=True)
        centered = data - (subj_mean - grand_mean)

        total_by_cond = np.nansum(centered, axis=0)
        n_by_cond = np.sum(~np.isnan(centered), axis=0)
        total_by_subj = np.nansum(centered, axis=1)
        n_by_subj = np.sum(~np.isnan(centered), axis=1)

        grand_total = np.nansum(centered)
        n_valid = np.sum(~np.isnan(centered))

        ss_total = np.nansum(centered**2) - grand_total**2 / n_valid
        ss_cond = np.sum(total_by_cond**2 / n_by_cond) - grand_total**2 / n_valid
        ss_subj = np.sum(total_by_subj**2 / n_by_subj) - grand_total**2 / n_valid
        ss_interaction = ss_total - ss_subj - ss_cond

        mean_n_subj = n_by_cond.sum() / ncond_cols
        mean_n_cond = n_by_subj.sum() / nsubj_rows

        ms_interaction = ss_interaction / (n_valid - mean_n_subj - mean_n_cond - 1)
        ms_within = (
            np.nansum(centered**2, axis=0) - total_by_cond**2 / n_by_cond
        ) / (n_by_cond - 1)

        estimator = (mean_n_cond / (mean_n_cond - 1)) * (
            ms_within - ms_interaction / mean_n_cond
        )
        return np.sqrt(estimator / n_by_cond) * student_t.ppf(0.975, nsubj_rows - 1)
