"""Figure styling.

Replaces ``publishFig.m``, which enlarged fonts by walking figure handles after
the fact.  A context manager setting rcParams up front does the same job
without mutating objects, and cannot miss an artist that was added later.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# The two list-type colours used throughout the original figures.
RAND_COLOR = (140 / 255, 26 / 255, 17 / 255)
TEMP_COLOR = (19 / 255, 66 / 255, 196 / 255)

RAND_LABEL = "Random"
TEMP_LABEL = "Autocorr"

# publishFig used 21pt for ticks and labels, 28pt for multi-panel titles.
BASE_FONT_SIZE = 21
TITLE_FONT_SIZE = 24

_STYLE = {
    "font.size": BASE_FONT_SIZE,
    "axes.labelsize": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE,
    "ytick.labelsize": BASE_FONT_SIZE,
    "legend.fontsize": BASE_FONT_SIZE - 4,
    "lines.linewidth": 2,
    "axes.linewidth": 1.5,
    "figure.figsize": (8, 6),
    "figure.autolayout": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@contextmanager
def publication_style(**overrides):
    """Apply the publication rcParams for the duration of the block."""
    settings = dict(_STYLE)
    settings.update(overrides)
    with mpl.rc_context(settings):
        yield


def new_axes(**kwargs):
    fig, ax = plt.subplots(**kwargs)
    return fig, ax


def series_style(list_type: str) -> dict:
    """Marker/line/colour for one list type, matching the original figures."""
    if list_type == "rand":
        return {
            "color": RAND_COLOR,
            "marker": "o",
            "linestyle": ":",
            "markerfacecolor": RAND_COLOR,
            "label": RAND_LABEL,
        }
    return {
        "color": TEMP_COLOR,
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": TEMP_COLOR,
        "label": TEMP_LABEL,
    }
