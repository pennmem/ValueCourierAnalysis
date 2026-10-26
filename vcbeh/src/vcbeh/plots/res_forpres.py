"""Presentation figures.

Ports ``plot_res_forpres.m`` and the figure from ``stats_vaviv2.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..embam import lag_index
from ..stats.contrasts import error_recall_correlation, recall_probability
from ..stats.loftus_masson import loftus_masson
from ..types import AnaResult
from .style import RAND_COLOR, TEMP_COLOR, new_axes, publication_style, series_style


def plot_spc(res: AnaResult):
    """Serial position curve by list type, with within-subject error bars."""
    fig, ax = new_axes()
    positions = np.arange(1, res.both.spcs.shape[1] + 1)
    for name, cond in (("rand", res.rand_), ("temp", res.temp)):
        ax.errorbar(
            positions, np.nanmean(cond.spcs, axis=0), loftus_masson(cond.spcs),
            **series_style(name),
        )
    ax.set(xlabel="Serial Position", ylabel="Recall Probability",
           xlim=(0.5, positions[-1] + 0.5), ylim=(0, 1),
           xticks=np.arange(3, positions[-1] + 1, 3))
    ax.legend(loc="lower left", frameon=False)
    return fig


def plot_pfr(res: AnaResult):
    """Probability of first recall by list type."""
    fig, ax = new_axes()
    positions = np.arange(1, res.both.pfrs.shape[1] + 1)
    for name, cond in (("rand", res.rand_), ("temp", res.temp)):
        ax.plot(positions, np.nanmean(cond.pfrs, axis=0), **series_style(name))
    ax.set(xlabel="Serial Position", ylabel="Probability of First Recall",
           xlim=(0.5, positions[-1] + 0.5), ylim=(0, 1),
           xticks=np.arange(3, positions[-1] + 1, 3))
    ax.legend(loc="upper left", frameon=False)
    return fig


def plot_lag_crp(res: AnaResult, max_abs_lag: int | None = None):
    """Lag-CRP by list type.

    Negative and positive lags are drawn as separate lines so nothing is
    connected across the undefined lag of zero.
    """
    fig, ax = new_axes()
    ll = (res.both.crps.shape[1] + 1) // 2
    span = ll - 1 if max_abs_lag is None else max_abs_lag
    backward = np.arange(-span, 0)
    forward = np.arange(1, span + 1)

    for name, cond in (("rand", res.rand_), ("temp", res.temp)):
        style = series_style(name)
        selected = cond.crps[:, lag_index(np.concatenate([backward, forward]), ll)]
        errors = loftus_masson(selected)
        for i, lags in enumerate((backward, forward)):
            values = np.nanmean(cond.crps[:, lag_index(lags, ll)], axis=0)
            ax.errorbar(
                lags, values, errors[i * span : (i + 1) * span],
                **(style if i == 0 else {**style, "label": None}),
            )
    ax.set(xlabel="Lag", ylabel="Cond. Resp. Prob.",
           xlim=(-span - 0.5, span + 0.5), ylim=(0, 0.6))
    ax.legend(loc="upper center", frameon=False)
    return fig


def _bar_by_list_type(values_rand, values_temp, ylabel, ylim=(0, 1)):
    """Group means with the Wilcoxon p for the paired comparison in the title.

    The test is a signed-rank on the paired differences rather than a t-test:
    on the raw data those differences are strongly non-normal.
    """
    from scipy.stats import wilcoxon

    fig, ax = new_axes()
    values = np.column_stack([np.asarray(values_rand).ravel(), np.asarray(values_temp).ravel()])
    paired = values[np.isfinite(values).all(axis=1)]
    nsub = values.shape[0]
    xs = [1, 2]

    for x, col, color in zip(xs, values.T, (RAND_COLOR, TEMP_COLOR)):
        ax.bar(x, np.nanmean(col), color=color, edgecolor="k", linewidth=2, width=0.5)
    # Standard error as the original computed it, with n-1 in the denominator.
    ax.errorbar(xs, np.nanmean(values, axis=0),
                np.nanstd(values, axis=0, ddof=0) / np.sqrt(nsub - 1),
                fmt="k.", markersize=0.2, linewidth=2)

    ax.set(ylabel=ylabel, xlim=(0, 3), ylim=ylim, xticks=xs,
           xticklabels=["Rand", "Autocorr"])
    if paired.shape[0] > 1:
        ax.set_title(f"p = {wilcoxon(paired[:, 0], paired[:, 1]).pvalue:.3f}")
    return fig


def plot_recall_probability_bars(res: AnaResult):
    return _bar_by_list_type(
        recall_probability(res.rand_), recall_probability(res.temp), "Recall probability"
    )


def plot_temporal_clustering(res: AnaResult):
    return _bar_by_list_type(
        res.rand_.temp_clusts, res.temp.temp_clusts, "Temporal clustering score"
    )


def plot_error_vs_recall(res: AnaResult):
    """Value-estimation error against recall probability, across subjects."""
    fig, ax = new_axes()
    err = np.asarray(res.both.abserr_pres_sub).ravel()
    prec = recall_probability(res.both)
    stats = error_recall_correlation(res)

    xs = np.array([err.min(), err.max()])
    ax.plot(xs, xs * stats["slope"] + stats["intercept"], "--", color=(0.5, 0.5, 0.5), linewidth=3)
    ax.plot(err, prec, "o", color="k", markerfacecolor="none", linewidth=3, linestyle="none")
    ax.set(xlabel="|Tip estimation error|", ylabel="Recall probability",
           xlim=(0, 8), ylim=(0.5, 1), xticks=range(1, 8, 2))
    ax.text(0.95, 0.95, f"r={stats['r']:.2f}\np={stats['p']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=18)
    return fig


def _condition_figures():
    """Imported lazily: these fit mixed models, unlike the descriptive panels."""
    from . import condition_models as cm

    return {
        "estimate_vs_actual": cm.plot_estimate_vs_actual,
        "partial_effects": cm.plot_partial_effects,
        "condition_coefficients": cm.plot_condition_coefficients,
    }


FIGURES = {
    "spc": plot_spc,
    "pfr": plot_pfr,
    "lag_crp": lambda res: plot_lag_crp(res, max_abs_lag=4),
    "lag_crp_full": plot_lag_crp,
    "recall_probability": plot_recall_probability_bars,
    "temporal_clustering": plot_temporal_clustering,
    "error_vs_recall": plot_error_vs_recall,
    **_condition_figures(),
}


def save_all(res: AnaResult, outdir: str | Path, fmt: str = "png") -> list[Path]:
    """Render every presentation figure into ``outdir``."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    with publication_style():
        for name, builder in FIGURES.items():
            fig = builder(res)
            path = outdir / f"{name}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written
