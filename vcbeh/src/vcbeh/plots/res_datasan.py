"""Per-subject sanity plots (port of ``plot_res_datasan.m``).

Not results figures: every subject is drawn individually so outliers and
data-collection problems are visible.
"""

from __future__ import annotations

import numpy as np

from ..embam import lag_index
from ..types import AnaResult
from .style import RAND_COLOR, TEMP_COLOR, new_axes

_CONDITION_COLORS = {"both": "k", "rand": RAND_COLOR, "temp": TEMP_COLOR}


def plot_curves_by_subject(res: AnaResult, measure: str, condition: str):
    """Every subject's curve for one measure, with the mean overlaid."""
    data = getattr(res.conditions()[condition], measure)
    positions = np.arange(1, data.shape[1] + 1)

    fig, ax = new_axes()
    ax.plot(positions, data.T, "x-", linewidth=1, alpha=0.6)
    ax.plot(positions, np.nanmean(data, axis=0), ":",
            color=_CONDITION_COLORS[condition], linewidth=3)
    ax.set(xlabel="Serial Position",
           ylabel={"spcs": "Recall Probability", "pfrs": "P(first recall)"}[measure],
           xlim=(0.5, positions[-1] + 0.5), ylim=(0, 1),
           title=f"{measure} - {condition}")
    return fig


def plot_crp_by_subject(res: AnaResult, condition: str):
    cond = res.conditions()[condition]
    ll = (cond.crps.shape[1] + 1) // 2
    backward = np.arange(-(ll - 1), 0)
    forward = np.arange(1, ll)

    fig, ax = new_axes()
    for lags in (backward, forward):
        cols = lag_index(lags, ll)
        ax.plot(lags, cond.crps[:, cols].T, "x-", linewidth=1, alpha=0.6)
        ax.plot(lags, np.nanmean(cond.crps[:, cols], axis=0), ":o",
                color=_CONDITION_COLORS[condition], linewidth=3)
    ax.set(xlabel="Lag", ylabel="Cond. Resp. Prob.",
           xlim=(-ll + 0.5, ll - 0.5), ylim=(0, 1), title=f"lag-CRP - {condition}")
    return fig
