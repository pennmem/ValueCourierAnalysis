"""Figures for the vcfrop analyses.

Palette and chart rules are shared with the notebook. The two series colours
are the project's own (the red ramp in ``beh_report_compare.m``), with the red
stepped one shade lighter than the MATLAB default so the pair clears a
colour-vision check: CVD separation dE 27.1 (protan), normal-vision dE 38.2,
both above 3:1 contrast on white.

Every chart carries a legend *and* direct labels where it can, so identity
never rests on colour alone.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from .stats import loftus_masson

RAND = "#ce1a18"
TEMP = "#1342c4"
NEUTRAL = "#6b6b6b"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

RAND_LABEL = "Random"
TEMP_LABEL = "Autocorrelated"
SERIES = {RAND_LABEL: RAND, TEMP_LABEL: TEMP}

_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
}


@contextmanager
def publication_style(**overrides):
    """Apply the shared chart style for the duration of the block."""
    import matplotlib as mpl

    with mpl.rc_context({**_STYLE, **overrides}):
        yield


def style_axes(ax, ylabel=None, xlabel=None, title=None):
    """Recessive grid and axes; text in ink, never in a series colour."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, color=INK)
    return ax


def _series_line(ax, x, mat, label, color, direct_label=True, capsize=0,
                 anchors=None):
    mean = np.nanmean(mat, axis=0)
    err = loftus_masson(mat)
    ax.errorbar(
        x, mean, yerr=err, color=color, lw=2, marker="o", ms=8,
        mfc=color, mec=SURFACE, mew=1.5, elinewidth=1.2, capsize=capsize,
        label=label, zorder=3,
    )
    if direct_label:
        finite = np.flatnonzero(np.isfinite(mean))
        if finite.size:
            last = finite[-1]
            anchor = (x[last], mean[last], label, color)
            if anchors is None:
                _place_direct_labels(ax, [anchor])
            else:
                anchors.append(anchor)
    return mean


def _place_direct_labels(ax, anchors, min_gap_frac=0.07):
    """Write end-of-line labels, nudged apart when two series end together.

    Without this the two condition labels overprint each other wherever the
    curves converge, which on these plots is most of the time.
    """
    if not anchors:
        return
    lo, hi = ax.get_ylim()
    span = hi - lo or 1.0
    gap = min_gap_frac * span

    ordered = sorted(anchors, key=lambda a: a[1])
    placed = []
    for x, y, label, color in ordered:
        if placed and y - placed[-1][1] < gap:
            y = placed[-1][1] + gap
        placed.append((x, y, label, color))

    for (x, y_text, label, color), (_, y_data, _, _) in zip(placed, ordered):
        ax.annotate(
            label, (x, y_text), color=color, fontsize=10, fontweight="semibold",
            va="center", xytext=(8, 0), textcoords="offset points",
            annotation_clip=False,
        )


def _legend_below(ax):
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))


# ------------------------------------------------------- recall measures


def plot_curve_by_list_type(res, field, ylabel, title, ylim=(0, 1), ax=None):
    """SPC- or PFR-style curve, random vs autocorrelated."""
    ll = res["both"][field].shape[1]
    x = np.arange(1, ll + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))

    for label, color in SERIES.items():
        cond = "rand" if label == RAND_LABEL else "temp"
        _series_line(ax, x, res[cond][field], label, color)

    style_axes(ax, ylabel=ylabel, xlabel="Serial position", title=title)
    ax.set_ylim(*ylim)
    # Headroom on the right so the direct labels are not clipped.
    ax.set_xlim(x[0] - 0.5, x[-1] + 4.5)
    ax.set_xticks(range(3, ll + 1, 3))
    _legend_below(ax)
    return ax


def plot_lag_crp(res, max_abs_lag=4, ax=None):
    ll = res["both"]["spcs"].shape[1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))

    lags_neg = np.arange(-max_abs_lag, 0)
    lags_pos = np.arange(1, max_abs_lag + 1)

    for label, color in SERIES.items():
        crps = res["rand" if label == RAND_LABEL else "temp"]["crps"]
        for i, lags in enumerate((lags_neg, lags_pos)):
            cols = lags + ll - 1  # lag L lives at column L + ll - 1
            _series_line(
                ax, lags, crps[:, cols], label if i == 0 else "_nolegend_",
                color, direct_label=False,
            )

    style_axes(ax, ylabel="Cond. response probability", xlabel="Lag",
               title=f"Lag-CRP (+/-{max_abs_lag})")
    ticks = np.concatenate([lags_neg, lags_pos])
    if max_abs_lag > 6:
        # Labelling every lag collides; keep +/-1 and every third.
        ticks = ticks[(np.abs(ticks) == 1) | (np.abs(ticks) % 3 == 0)]
    ax.set_xticks(ticks)
    _legend_below(ax)
    return ax


def plot_paired_bars(values, ylabel, title, chance=None, ax=None):
    """Condition means with every subject's own pair drawn behind them."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.2))

    means = [np.mean(v) for v in values]
    sem = [np.std(v, ddof=1) / np.sqrt(len(v)) for v in values]
    x = [0, 1]

    for xi, (m, e, color) in enumerate(zip(means, sem, SERIES.values())):
        ax.bar(xi, m, width=0.52, color=color, zorder=2,
               edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xi, m, yerr=e, color=INK, lw=1.4, capsize=0, zorder=4)

    # n is small enough that showing the subjects is more honest than a bar.
    for a, b in zip(*values):
        ax.plot(x, [a, b], color=MUTED, lw=0.8, alpha=0.55, zorder=3,
                marker="o", ms=4, mfc=SURFACE, mec=MUTED)

    if chance is not None:
        ax.axhline(chance, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.annotate("chance", (-0.55, chance), color=MUTED, fontsize=9,
                    va="bottom", ha="left", xytext=(0, 4),
                    textcoords="offset points")

    style_axes(ax, ylabel=ylabel, title=title)
    ax.set_xticks(x)
    ax.set_xticklabels(list(SERIES))
    ax.set_xlim(-0.6, 1.6)
    return ax


def plot_prec_byval(res, ax=None, smooth=5):
    """Recall probability as a function of an item's value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))

    pv = res["both"]["prec_byval"]
    values = np.arange(1, pv.shape[1] + 1)
    mean = np.nanmean(pv, axis=0)

    ax.plot(values, mean, "o", ms=5, color=NEUTRAL, alpha=0.5,
            label="per value", zorder=2)
    if smooth > 1:
        # A centred rolling mean that shrinks its window at the edges. A
        # convolution would pad with zeros and drag the ends of the curve down,
        # inventing structure at exactly the values with fewest observations.
        half = smooth // 2
        rolled = np.array([
            np.nanmean(mean[max(0, i - half): i + half + 1])
            for i in range(len(mean))
        ])
        ax.plot(values, rolled, lw=2.5, color=TEMP,
                label=f"{smooth}-point mean", zorder=3)

    overall = float(np.nanmean(res["both"]["spcs"]))
    ax.axhline(overall, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("overall P(recall)", (values[0], overall), color=MUTED,
                fontsize=9, ha="left", va="top", xytext=(2, -5),
                textcoords="offset points")

    style_axes(ax, ylabel="Recall probability", xlabel="Item value",
               title="Recall probability by item value")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    return ax


# ------------------------------------------- z-scored value estimation


def plot_value_curve(
    curves, x, xlabel, title, ylabel="Estimation error (list z units)",
    counts=None, min_count=20, ax=None, zero_line=True, truncate=True,
):
    """One of the new value-estimation curves, random vs autocorrelated.

    ``curves`` maps a series label to its ``(n_subjects, n_bins)`` matrix.
    ``counts`` is the per-bin observation count; bins below ``min_count`` are
    shaded, because a hierarchical mean makes a bin of three observations look
    exactly like a bin of three hundred.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.2))

    x = np.asarray(x)
    # A hierarchical mean makes a bin of three observations look exactly like a
    # bin of three hundred, and beyond the sparse threshold the error bars
    # vanish entirely because only one subject contributes. Drawing a line
    # through that implies a precision the data do not have, so the tail is cut
    # rather than merely shaded.
    if counts is not None and truncate:
        dense = np.flatnonzero(np.asarray(counts)[: len(x)] >= min_count)
        if dense.size:
            keep = slice(0, int(dense[-1]) + 1)
            x = x[keep]
            curves = {k: v[:, keep] for k, v in curves.items()}

    if zero_line:
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)

    anchors = []
    for label, mat in curves.items():
        _series_line(ax, x, mat, label, SERIES.get(label, NEUTRAL), anchors=anchors)

    style_axes(ax, ylabel=ylabel, xlabel=xlabel, title=title)
    ax.set_xlim(x[0] - 0.7, x[-1] + max(3.0, 0.10 * (x[-1] - x[0])))
    step = max(1, int(np.ceil(len(x) / 8)))
    ax.set_xticks([int(v) for v in x[::step]])
    _place_direct_labels(ax, anchors)
    _legend_below(ax)
    return ax


def plot_value_curve_lag(curves, ll, counts=None, min_count=20,
                         max_abs_lag=None, ylabel=None, title=None, ax=None):
    """The lag-binned value-estimation curve, drawn either side of lag 0."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.2))

    max_abs_lag = max_abs_lag or (ll - 1)
    lags_neg = np.arange(-max_abs_lag, 0)
    lags_pos = np.arange(1, max_abs_lag + 1)

    if counts is not None:
        counts = np.asarray(counts)
        for lag in np.concatenate([lags_neg, lags_pos]):
            if counts[lag + ll - 1] < min_count:
                ax.axvspan(lag - 0.5, lag + 0.5, color=GRID, alpha=0.55,
                           lw=0, zorder=0)

    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)

    for label, mat in curves.items():
        color = SERIES.get(label, NEUTRAL)
        for i, lags in enumerate((lags_neg, lags_pos)):
            _series_line(ax, lags, mat[:, lags + ll - 1],
                         label if i == 0 else "_nolegend_", color,
                         direct_label=False)

    style_axes(ax, ylabel=ylabel or "Estimation error (list z units)",
               xlabel="Lag", title=title or "Estimation error by lag")
    ticks = np.concatenate([lags_neg, lags_pos])
    if max_abs_lag > 6:
        ticks = ticks[(np.abs(ticks) == 1) | (np.abs(ticks) % 3 == 0)]
    ax.set_xticks(ticks)
    _legend_below(ax)
    return ax


# ------------------------------------------------------ condition models


def plot_estimate_vs_actual(res, table_builder, list_length, ax=None):
    """Estimate against the list's true value, with a fitted line per condition."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 5.0))

    table = table_builder(res, list_length)
    lo = min(table["ValuePres"].min(), table["ValueEst"].min()) - 2
    hi = max(table["ValuePres"].max(), table["ValueEst"].max()) + 2

    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.annotate("perfect accuracy", (0.985, 0.985), xycoords="axes fraction",
                color=MUTED, fontsize=9, ha="right", va="top")

    for label, color in SERIES.items():
        cond = "rand" if label == RAND_LABEL else "temp"
        block = table[table["ListType"] == cond]
        ax.scatter(block["ValuePres"], block["ValueEst"], s=22, color=color,
                   alpha=0.35, linewidths=0, zorder=3)
        slope, intercept = np.polyfit(block["ValuePres"], block["ValueEst"], 1)
        xs = np.array([lo, hi])
        ax.plot(xs, slope * xs + intercept, color=color, lw=2.5, zorder=4,
                label=f"{label} (slope {slope:.2f})")

    style_axes(ax, ylabel="Subject's estimate",
               xlabel="Mean value of all studied items",
               title="Estimate vs. truth")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _legend_below(ax)
    return ax


def plot_condition_coefficients(tables, ax=None):
    """Forest plot of coefficients with 95% intervals, by list type.

    ``tables`` maps a round name to a ``condition_models`` frame.
    """
    rows = []
    for round_name, table in tables.items():
        for _, row in table.iterrows():
            if row["term"] == "Intercept":
                continue
            rows.append({**row, "round": round_name})
    if not rows:
        return ax

    height = 0.55 * len(rows) + 1.6
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, height))

    labels = []
    for i, row in enumerate(reversed(rows)):
        color = RAND if row["condition"] == "rand" else TEMP
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=2.5,
                solid_capstyle="round", zorder=3)
        ax.plot(row["estimate"], i, "o", ms=9, color=color, mec=SURFACE,
                mew=1.5, zorder=4)
        labels.append(f"{row['round']} · {row['list_type']} · {row['label']}")

    ax.axvline(1.0, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.annotate("tracks value one-for-one", (1.0, len(rows) - 0.45),
                color=MUTED, fontsize=9, ha="right", va="bottom",
                xytext=(-5, 2), textcoords="offset points")
    ax.axvline(0.0, color=GRID, lw=1.5, zorder=1)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    style_axes(ax, xlabel="Coefficient (points of estimate per point of value)",
               title="Coefficients with 95% intervals")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    for label, color in SERIES.items():
        ax.plot([], [], "o-", color=color, label=label, lw=2.5, ms=9)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12 - 0.3 / height))
    return ax
