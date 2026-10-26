"""How many sessions each participant completed (port of ``count_sub_sess.m``)."""

from __future__ import annotations

import numpy as np

from ..types import Trials
from .style import new_axes

MAX_SESSIONS = 6


def plot_session_counts(trials: Trials, max_sessions: int = MAX_SESSIONS):
    subjects = np.unique(trials.subject)
    per_subject = np.array(
        [np.unique(trials.session[trials.subject == s]).size for s in subjects]
    )
    counts = [int((per_subject == n).sum()) for n in range(1, max_sessions + 1)]

    fig, ax = new_axes()
    ax.bar(range(1, max_sessions + 1), counts, color="k")
    ax.set(xlabel="Number of sessions", ylabel="Number of participants",
           xlim=(0.5, max_sessions + 0.5))
    return fig
