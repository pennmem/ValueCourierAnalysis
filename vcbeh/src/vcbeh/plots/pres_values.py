"""Presented-value sequences per subject and session.

Port of ``plot_pres_values.m``: one figure per subject-session showing every
list's value sequence, so the random and autocorrelated generators can be
eyeballed against each other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import Trials
from .style import new_axes, publication_style

# Five shades per list type, so individual lists stay distinguishable.
RAND_SHADES = np.array(
    [[140, 26, 17], [169, 30, 19], [206, 26, 24], [240, 42, 28], [255, 91, 79]]
) / 255
TEMP_SHADES = np.array(
    [[9, 30, 88], [13, 46, 135], [19, 66, 196], [23, 84, 247], [106, 145, 255]]
) / 255

VALUE_MIDPOINT = 25
# Boundaries of the value "blocks" within a list.
BLOCK_EDGES = (2.5, 7.5, 12.5)


def plot_pres_values(trials: Trials, subject: int, session: int):
    """Value sequence of every list this subject saw in this session."""
    rows = (trials.subject == subject) & (trials.session == session)
    values = trials.pres_values[rows]
    types = trials.pres_types[rows]
    positions = np.arange(1, values.shape[1] + 1)

    fig, ax = new_axes()
    ax.plot([1, values.shape[1]], [VALUE_MIDPOINT] * 2, "-.", color="y", linewidth=4)
    for edge in BLOCK_EDGES:
        ax.plot([edge, edge], [1, 50], ":", color=(0.5, 0.5, 0.5), linewidth=4)

    for list_type, shades in ((1, RAND_SHADES), (2, TEMP_SHADES)):
        for i, row in enumerate(np.nonzero(types == list_type)[0]):
            color = shades[i % len(shades)]
            ax.plot(positions, values[row], "x-", linewidth=2, color=color)
            ax.plot([0.5, positions[-1]], [values[row].mean()] * 2, "--",
                    linewidth=1, color=color)

    ax.set(xlabel="Serial position", ylabel="Value", xlim=(0, 16), ylim=(0, 51),
           xticks=range(1, 16, 2), title=f"sub-LTP{subject}, session {session}")
    return fig


def save_all(trials: Trials, outdir: str | Path, fmt: str = "png") -> list[Path]:
    """One figure per subject-session.  This makes a lot of files."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    with publication_style():
        for subject in np.unique(trials.subject):
            for session in np.unique(trials.session[trials.subject == subject]):
                fig = plot_pres_values(trials, subject, session)
                path = outdir / f"pres_values_sub{subject}_sess{session}.{fmt}"
                fig.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                written.append(path)
    return written
