"""Figures for the per-condition value-estimation models.

Three views of the same four models: what the raw relationship looks like,
what each predictor contributes once both are in the model, and whether any
coefficient actually differs between list types.
"""

from __future__ import annotations

import numpy as np

from ..stats.linreg import (
    CONDITION_LABELS,
    build_list_table,
    condition_models,
)
from ..types import AnaResult
from .style import RAND_COLOR, TEMP_COLOR, new_axes

_COLORS = {"rand": RAND_COLOR, "temp": TEMP_COLOR}
_MARKERS = {"rand": "o", "temp": "s"}
_LINESTYLES = {"rand": ":", "temp": "--"}


def _list_length(res: AnaResult, list_length: int | None) -> int:
    """List length, taken from the result itself unless overridden.

    The SPC has one column per serial position, so the result already knows
    this -- no need for a hardcoded default that could silently be wrong.
    """
    return int(list_length) if list_length is not None else int(res.both.spcs.shape[1])


def plot_estimate_vs_actual(res: AnaResult, list_length: int | None = None):
    """Round 1: every list's estimate against its true average value.

    The dashed diagonal is perfect accuracy.  A fitted line that crosses it --
    above at low values, below at high ones -- is the whole story, shown rather
    than reduced to a number.
    """
    list_length = _list_length(res, list_length)
    fig, ax = new_axes(figsize=(8.5, 7))
    models = condition_models(res, list_length, corrected=False)

    lo, hi = np.inf, -np.inf
    for label in ("rand", "temp"):
        table = build_list_table(res, list_length, condition=label)
        if table.empty:
            continue
        x = table["ValuePres"].to_numpy()
        y = table["ValueEst"].to_numpy()
        lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        ax.plot(x, y, _MARKERS[label], color=_COLORS[label], markersize=4,
                alpha=0.28, linestyle="none")

    span = np.array([lo, hi])
    ax.plot(span, span, "--", color="0.35", linewidth=2, zorder=2,
            label="Perfect accuracy")

    for label in ("rand", "temp"):
        block = models[models["condition"] == label].set_index("term")
        if block.empty:
            continue
        fitted = block.loc["Intercept", "estimate"] + block.loc["ValuePres", "estimate"] * span
        ax.plot(span, fitted, _LINESTYLES[label], color=_COLORS[label], linewidth=3,
                zorder=3, label=CONDITION_LABELS[label])

    ax.set(xlabel="Actual list value", ylabel="Estimated list value")
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_partial_effects(res: AnaResult, list_length: int | None = None):
    """Round 2: each predictor's contribution with the other held at its mean.

    A scatter against either predictor alone would confound the two, since
    recalled and non-recalled value both move with the list's overall value.
    """
    list_length = _list_length(res, list_length)
    models = condition_models(res, list_length, corrected=True)
    terms = ["ValueRecallW", "ValueNotRecallW"]
    titles = ["Recalled value", "Non-recalled value"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, focus, title in zip(axes, terms, titles):
        other = next(t for t in terms if t != focus)
        for label in ("rand", "temp"):
            table = build_list_table(
                res, list_length, condition=label, partial_only=True
            )
            block = models[models["condition"] == label].set_index("term")
            if table.empty or block.empty:
                continue
            xs = np.linspace(table[focus].min(), table[focus].max(), 50)
            predicted = (
                block.loc["Intercept", "estimate"]
                + block.loc[focus, "estimate"] * xs
                + block.loc[other, "estimate"] * table[other].mean()
            )
            ax.plot(table[focus], table["ValueEst"], _MARKERS[label],
                    color=_COLORS[label], markersize=4, alpha=0.25, linestyle="none")
            ax.plot(xs, predicted, _LINESTYLES[label], color=_COLORS[label],
                    linewidth=3, zorder=3, label=CONDITION_LABELS[label])
        ax.set(xlabel=f"{title} (weighted by list share)")
    axes[0].set_ylabel("Estimated list value")
    axes[0].legend(frameon=False, loc="upper left")
    return fig


def plot_condition_coefficients(res: AnaResult, list_length: int | None = None):
    """All four models at once: every coefficient with its 95% interval.

    Intervals that overlap between the two colours mean the conditions are not
    distinguishable on that term.
    """
    list_length = _list_length(res, list_length)
    rounds = [
        ("Uncorrected", condition_models(res, list_length, corrected=False)),
        ("Recall-corrected", condition_models(res, list_length, corrected=True)),
    ]

    entries = []
    for round_name, table in rounds:
        for term in table["term"].unique():
            if term == "Intercept":
                continue
            entries.append((round_name, term))

    fig, ax = new_axes(figsize=(9.5, 1.6 * len(entries) + 2.5))
    offsets = {"rand": +0.16, "temp": -0.16}

    for row, (round_name, term) in enumerate(entries):
        table = dict(rounds)[round_name]
        for label in ("rand", "temp"):
            block = table[(table["condition"] == label) & (table["term"] == term)]
            if block.empty:
                continue
            r = block.iloc[0]
            y = row + offsets[label]
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], "-",
                    color=_COLORS[label], linewidth=3)
            ax.plot(r["estimate"], y, _MARKERS[label], color=_COLORS[label],
                    markersize=11,
                    label=CONDITION_LABELS[label] if row == 0 else None)

    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=2)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([f"{t}\n({r})" for r, t in entries], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (95% CI)")
    ax.legend(frameon=False, loc="upper left")
    return fig
