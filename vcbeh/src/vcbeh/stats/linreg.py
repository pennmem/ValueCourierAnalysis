"""Mixed-effects regressions of value estimates on recalled/non-recalled value.

Ports ``linreg_estval_pressrecallv2.m`` and
``linreg_estval_nonrecallrecallval.m``.

The question: a subject's estimate of a list's average value could be built
from the items they recalled, from the items they did not, or from both.  With
``nrec`` of ``ll`` items recalled, a full account predicts

    ValueEst ~ (nrec/ll) * ValueRecalled + ((ll-nrec)/ll) * ValueNotRecalled

but ``nrec`` differs per list, so no single weighting fits all lists at once.
Both scripts work around this by fitting one model per value of ``nrec``.
Lists with ``nrec == ll`` are excluded because the two accounts coincide there.

**No MATLAB ground truth exists for these outputs**: both scripts index
``act_nrec_list`` and ``act_nonrec_list``, which no saved ``res`` file
contains, so neither could run before this port.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t

from ..types import AnaResult

# The two sweeps as the original scripts defined them.
WEIGHTED_RECALL_SWEEP = range(7, 14)  # linreg_estval_nonrecallrecallval.m
UNWEIGHTED_RECALL_SWEEP = range(9, 15)  # linreg_estval_pressrecallv2.m

# statsmodels' L-BFGS can settle on a boundary solution here -- group variance
# exactly zero and an intercept pinned to zero -- while reporting convergence.
# Every other optimizer agrees on a sensible fit, so all are tried and the one
# with the highest likelihood wins.  These are ML fits, so that comparison is
# exactly the right tie-breaker.
_OPTIMIZERS = ("bfgs", "powell", "cg", "nm", "lbfgs")

ADDITIVE_MODEL = "ValueEst ~ ValueRecall + ValueNotRecall"
INTERACTION_MODEL = "ValueEst ~ ValueRecall * ValueNotRecall"


def build_table(
    res: AnaResult, nrec: int, list_length: int, weight_recall: bool
) -> pd.DataFrame:
    """Long-format table of lists on which exactly ``nrec`` items were recalled.

    ``ValueNotRecall`` is always scaled by the fraction of the list it covers.
    ``ValueRecall`` is scaled to match only when ``weight_recall`` is set --
    the two original scripts differ on exactly this point.
    """
    rec_coeff = nrec / list_length if weight_recall else 1.0
    nonrec_coeff = (list_length - nrec) / list_length

    frames = []
    for label, cond in (("rand", res.rand_), ("temp", res.temp)):
        selected = cond.act_nrec_list == nrec
        subject_idx, list_idx = np.nonzero(selected)
        if subject_idx.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "Subject": subject_idx,
                    "ValueEst": cond.est_list[subject_idx, list_idx],
                    "ValueRecall": rec_coeff * cond.act_rec_list[subject_idx, list_idx],
                    "ValueNotRecall": nonrec_coeff
                    * cond.act_nonrec_list[subject_idx, list_idx],
                    "ListType": label,
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["Subject", "ValueEst", "ValueRecall", "ValueNotRecall", "ListType"]
        )
    table = pd.concat(frames, ignore_index=True)
    return table.dropna(
        subset=["ValueEst", "ValueRecall", "ValueNotRecall"]
    ).reset_index(drop=True)


def _fit_best_likelihood(model):
    """Fit with each optimizer and keep the highest-likelihood result."""
    best = None
    for method in _OPTIMIZERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(reml=False, method=method)
        except Exception:  # optimizer-specific failures are expected
            continue
        if not np.isfinite(fit.llf):
            continue
        if best is None or fit.llf > best.llf + 1e-9:
            best = fit
    if best is None:
        raise RuntimeError(f"no optimizer converged for {model.formula!r}")
    return best


def fit_mixed(table: pd.DataFrame, formula: str) -> dict:
    """Fit ``formula`` with a random intercept per subject, matching ``fitlme``.

    Three settings have to be forced, because ``statsmodels`` and MATLAB
    disagree on all of them by default:

    * ``reml=False`` -- ``fitlme`` defaults to maximum likelihood.
    * p-values from a *t* distribution on residual degrees of freedom, not the
      z-test ``statsmodels`` reports; ``fitlme`` defaults to
      ``DFMethod='residual'``.
    * ``Rsquared.Ordinary`` is **conditional**, including the random effects,
      whereas ``statsmodels.fittedvalues`` excludes them.
    """
    if table.empty:
        raise ValueError("no lists matched; cannot fit")

    model = smf.mixedlm(formula, table, groups=table["Subject"])
    fit = _fit_best_likelihood(model)

    params = fit.fe_params
    names = list(params.index)
    se = fit.bse[names]
    nobs = int(table.shape[0])
    df_resid = nobs - len(names)
    tvals = params / se
    pvals = 2 * student_t.sf(np.abs(tvals), df_resid)

    intercepts = {g: float(np.asarray(v).ravel()[0]) for g, v in fit.random_effects.items()}
    conditional = fit.fittedvalues + table["Subject"].map(intercepts).to_numpy()
    observed = table["ValueEst"].to_numpy()
    sse = float(np.sum((observed - conditional) ** 2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    # Marginal R2 uses the fixed effects alone.  Reported alongside because
    # some earlier write-ups of this analysis quoted that version.
    sse_marginal = float(np.sum((observed - fit.fittedvalues.to_numpy()) ** 2))

    return {
        "n": nobs,
        "n_subjects": int(table["Subject"].nunique()),
        "df_resid": df_resid,
        "names": names,
        "estimates": params.to_numpy(),
        "se": se.to_numpy(),
        "t": tvals.to_numpy(),
        "p": np.asarray(pvals),
        "r2_conditional": 1 - sse / sst,
        "r2_marginal": 1 - sse_marginal / sst,
    }


def sweep(
    res: AnaResult,
    list_length: int,
    nrec_values,
    formula: str,
    weight_recall: bool,
) -> pd.DataFrame:
    """Fit one model per ``nrec`` and collect the coefficients into a table."""
    rows = []
    for nrec in nrec_values:
        table = build_table(res, nrec, list_length, weight_recall)
        if table.empty or table["Subject"].nunique() < 3:
            continue
        fit = fit_mixed(table, formula)
        row = {
            "nrec": nrec,
            "n_lists": fit["n"],
            "n_subjects": fit["n_subjects"],
            "R2": fit["r2_conditional"],
        }
        for name, est, p in zip(fit["names"], fit["estimates"], fit["p"]):
            key = name.replace("Intercept", "int").replace("[T.", "_").replace("]", "")
            row[f"b_{key}"] = est
            row[f"p_{key}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def recalled_vs_nonrecalled(res: AnaResult, list_length: int = 15) -> pd.DataFrame:
    """The additive model with both terms weighted by list share."""
    return sweep(
        res, list_length, WEIGHTED_RECALL_SWEEP, ADDITIVE_MODEL, weight_recall=True
    )


def recalled_unweighted(res: AnaResult, list_length: int = 15) -> pd.DataFrame:
    """The interaction model with an unweighted recalled-value term."""
    return sweep(
        res,
        list_length,
        UNWEIGHTED_RECALL_SWEEP,
        INTERACTION_MODEL,
        weight_recall=False,
    )


def format_sweep(table: pd.DataFrame, title: str) -> str:
    if table.empty:
        return f"== {title} ==\n  (no lists matched)"
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        body = table.to_string(index=False, float_format=lambda v: f"{v:8.4f}")
    return f"== {title} ==\n{body}"


# ---------------------------------------------------------------------------
# Per-condition models: is the estimate built differently for random vs
# autocorrelated lists?
# ---------------------------------------------------------------------------

UNCORRECTED_MODEL = "ValueEst ~ ValuePres"
CORRECTED_MODEL = "ValueEst ~ ValueRecallW + ValueNotRecallW"
CORRECTED_INTERACTION_MODEL = "ValueEst ~ ValueRecallW * ValueNotRecallW"

UNCORRECTED_CONTRAST = "ValueEst ~ ValuePres * ListType"
CORRECTED_CONTRAST = "ValueEst ~ (ValueRecallW + ValueNotRecallW) * ListType"

UNCORRECTED_PREDICTORS = ("ValuePres",)
CORRECTED_PREDICTORS = ("ValueRecallW", "ValueNotRecallW")

CONDITION_LABELS = {"rand": "Random", "temp": "Autocorr"}

# The model terms are named after dataframe columns, which does not make clear
# which value each one refers to.  These are what gets printed instead.
TERM_LABELS = {
    "Intercept": "Intercept",
    "ValuePres": "Value of ALL studied items",
    "ValueRecallW": "Value of RECALLED items",
    "ValueNotRecallW": "Value of NON-RECALLED items",
    "ValueAll": "Value from ALL studied items",
    "ValueRecalled": "Value from RECALLED items",
    "ValueAll:ValueRecalled": "Interaction (All x Recalled)",
    "ValueRecallW:ValueNotRecallW": "Interaction (Recalled x Non-recalled)",
}

# Short forms, so interaction labels stay inside a readable column.
SHORT_TERM_LABELS = {
    "ValuePres": "ALL studied",
    "ValueRecallW": "RECALLED",
    "ValueNotRecallW": "NON-RECALLED",
    "ValueAll": "ALL studied",
    "ValueRecalled": "RECALLED",
    "ValueRecallW:ValueNotRecallW": "REC x NON-REC",
}

_LIST_TYPE_TERM = "ListType[T.temp]"


def term_label(term: str) -> str:
    """Human-readable name for a model term, interactions included."""
    if term in TERM_LABELS:
        return TERM_LABELS[term]
    if term == _LIST_TYPE_TERM:
        return "List type (Autocorr vs Random)"
    if term.endswith(":" + _LIST_TYPE_TERM):
        base = term[: -len(":" + _LIST_TYPE_TERM)]
        return f"{SHORT_TERM_LABELS.get(base, base)}  x  list type"
    return term


def build_list_table(
    res: AnaResult,
    list_length: int,
    *,
    condition: str | None = None,
    partial_only: bool = False,
) -> pd.DataFrame:
    """One row per list, with the recall decomposition weighted per list.

    Unlike `build_table`, this does not fix ``nrec``: each list is weighted by
    **its own** recall count, so a single model spans every list rather than
    one model per recall count.  That is what makes a per-condition fit
    possible without splitting the data seven ways.

    ``partial_only`` drops lists where nothing or everything was recalled --
    ``act_nonrec_list`` is undefined for a fully recalled list, and a list with
    no recalls has no recalled value.
    """
    ll = int(list_length)
    sources = {"rand": res.rand_, "temp": res.temp}
    if condition is not None:
        sources = {condition: sources[condition]}

    frames = []
    for label, cond in sources.items():
        nrec = cond.act_nrec_list
        # A list with no recalls has nrec == NaN.  That only disqualifies it
        # from the decomposition -- the uncorrected model does not use nrec, so
        # requiring it there would silently drop the list for no reason.
        usable = np.isfinite(cond.est_list) & np.isfinite(cond.act_pres_list)
        if partial_only:
            usable &= (
                np.isfinite(nrec)
                & (nrec > 0)
                & (nrec < ll)
                & np.isfinite(cond.act_rec_list)
                & np.isfinite(cond.act_nonrec_list)
            )
        si, li = np.nonzero(usable)
        if si.size == 0:
            continue
        n = nrec[si, li]
        frames.append(
            pd.DataFrame(
                {
                    "Subject": si,
                    "ListType": label,
                    "ValueEst": cond.est_list[si, li],
                    "ValuePres": cond.act_pres_list[si, li],
                    "nrec": n,
                    "ValueRecallW": (n / ll) * cond.act_rec_list[si, li],
                    "ValueNotRecallW": ((ll - n) / ll) * cond.act_nonrec_list[si, li],
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "Subject", "ListType", "ValueEst", "ValuePres", "nrec",
                "ValueRecallW", "ValueNotRecallW",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def center_predictors(
    table: pd.DataFrame, columns, centers: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Subtract each predictor's mean, returning the table and the constants used.

    Centering moves the intercept from the predictor's zero -- which is far
    outside the observed range here, list values run 8 to 42 -- to its mean, so
    the intercept and any categorical main effect become statements about a
    typical list rather than extrapolations.

    It leaves slopes, interaction terms, fitted values and R-squared untouched;
    only terms whose meaning depends on where zero sits actually move.

    Pass ``centers`` to reuse constants computed elsewhere. Per-condition fits
    must share one set, or their intercepts are measured at different points
    and cannot be compared.
    """
    if centers is None:
        centers = {name: float(table[name].mean()) for name in columns}
    centered = table.copy()
    for name, mean in centers.items():
        centered[name] = centered[name] - mean
    return centered, centers


def _coefficient_rows(fit: dict, centers: dict[str, float] | None = None, **extra) -> list[dict]:
    """Flatten a `fit_mixed` result into one tidy row per coefficient."""
    crit = student_t.ppf(0.975, fit["df_resid"])
    rows = []
    for name, est, se, t, p in zip(
        fit["names"], fit["estimates"], fit["se"], fit["t"], fit["p"]
    ):
        rows.append(
            {
                **extra,
                "term": name,
                "label": term_label(name),
                "centered_at": (centers or {}).get(name, np.nan),
                "estimate": est,
                "se": se,
                "ci_low": est - crit * se,
                "ci_high": est + crit * se,
                "t": t,
                "p": p,
                "n_lists": fit["n"],
                "n_subjects": fit["n_subjects"],
                "R2": fit["r2_conditional"],
                "R2_marginal": fit["r2_marginal"],
            }
        )
    return rows


def condition_models(
    res: AnaResult,
    list_length: int = 15,
    *,
    corrected: bool,
    partial_only: bool | None = None,
    center: bool = False,
    interaction: bool = False,
) -> pd.DataFrame:
    """Fit one mixed model per list type; tidy one row per coefficient.

    ``corrected=False`` regresses the estimate on the list's true average value.
    ``corrected=True`` replaces that with the recalled / non-recalled
    decomposition, each weighted by the share of the list it covers.

    ``interaction=True`` (corrected models only) adds a
    ``ValueRecallW x ValueNotRecallW`` term.  Note that this makes each main
    effect the slope of that predictor where the *other* is zero, so the two
    are no longer read as independent contributions.

    Predictors are left uncentered by default, which is how the original
    analyses were run.  Pass ``center=True`` to mean-center them, using
    constants computed over both conditions pooled so the two intercepts stay
    comparable.  Centering moves the intercept and, where an interaction is
    present, the other terms in it; slopes and fit are unaffected.
    """
    if not corrected:
        if interaction:
            raise ValueError(
                "the uncorrected model has a single predictor, so there is "
                "nothing for it to interact with"
            )
        formula = UNCORRECTED_MODEL
        predictors = UNCORRECTED_PREDICTORS
    else:
        formula = CORRECTED_INTERACTION_MODEL if interaction else CORRECTED_MODEL
        predictors = CORRECTED_PREDICTORS
    if partial_only is None:
        partial_only = corrected

    centers = None
    if center:
        pooled = build_list_table(res, list_length, partial_only=partial_only)
        if not pooled.empty:
            _, centers = center_predictors(pooled, predictors)

    rows = []
    for label in ("rand", "temp"):
        table = build_list_table(
            res, list_length, condition=label, partial_only=partial_only
        )
        if table.empty:
            continue
        if centers:
            table, _ = center_predictors(table, predictors, centers)
        fit = fit_mixed(table, formula)
        rows += _coefficient_rows(
            fit, centers, condition=label, list_type=CONDITION_LABELS[label]
        )
    return pd.DataFrame(rows)


def condition_contrast(
    res: AnaResult, list_length: int = 15, *, corrected: bool, center: bool = False
) -> pd.DataFrame:
    """Pooled model with a ``ListType`` interaction.

    Fitting the two conditions separately shows what each looks like but cannot
    say whether they *differ*; the interaction terms here are that test.  A
    term like ``ValueNotRecallW:ListType[T.temp]`` asks whether non-recalled
    value contributes differently in autocorrelated lists.
    """
    formula = CORRECTED_CONTRAST if corrected else UNCORRECTED_CONTRAST
    predictors = CORRECTED_PREDICTORS if corrected else UNCORRECTED_PREDICTORS
    table = build_list_table(res, list_length, partial_only=corrected)
    if table.empty:
        return pd.DataFrame()

    centers = None
    if center:
        table, centers = center_predictors(table, predictors)
    return pd.DataFrame(_coefficient_rows(fit_mixed(table, formula), centers))


def _centering_note(table: pd.DataFrame) -> str:
    """One line describing where the intercept sits."""
    if "centered_at" not in table.columns:
        return "  predictors uncentred: intercept is at a predictor value of 0"
    seen = table.dropna(subset=["centered_at"]).drop_duplicates(subset="term")
    if seen.empty:
        return "  predictors uncentred: intercept is at a predictor value of 0"
    parts = ", ".join(f"{row.term}={row.centered_at:.2f}" for row in seen.itertuples())
    return f"  predictors mean-centred ({parts}); intercept is at a typical list"


def format_condition_models(table: pd.DataFrame, title: str) -> str:
    if table.empty:
        return f"== {title} ==\n  (no lists matched)"
    lines = [f"== {title} ==", _centering_note(table)]
    for condition, block in table.groupby("condition", sort=False):
        head = block.iloc[0]
        lines.append(
            f"  {CONDITION_LABELS.get(condition, condition):9s} "
            f"n={int(head['n_lists'])} lists, {int(head['n_subjects'])} subjects, "
            f"R2={head['R2']:.3f}"
        )
        for _, row in block.iterrows():
            lines.append(
                f"      {row['label']:38s} {row['estimate']:+8.4f} "
                f"[{row['ci_low']:+7.4f}, {row['ci_high']:+7.4f}]  "
                f"t={row['t']:+7.3f}  p={row['p']:.4g}"
            )
    return "\n".join(lines)


def format_contrast(table: pd.DataFrame, title: str) -> str:
    if table.empty:
        return f"== {title} ==\n  (no lists matched)"
    head = table.iloc[0]
    lines = [
        f"== {title} ==",
        _centering_note(table),
        f"  n={int(head['n_lists'])} lists, {int(head['n_subjects'])} subjects, "
        f"R2={head['R2']:.3f}   (the 'x list type' rows test the difference)",
    ]
    for _, row in table.iterrows():
        marker = "  *" if row["p"] < 0.05 else ""
        lines.append(
            f"      {row['label']:32s} {row['estimate']:+8.4f}  "
            f"t={row['t']:+7.3f}  p={row['p']:.4g}{marker}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The original ValueAll x ValueRecalled model
# ---------------------------------------------------------------------------

ALL_RECALLED_MODEL = "ValueEst ~ ValueAll * ValueRecalled"
ALL_RECALLED_PREDICTORS = ("ValueAll", "ValueRecalled")

# Slide/table ordering: intercept, the studied term, the recalled term, then
# the interaction -- matching `inds = [1;3;2;4]` in the MATLAB .tex block.
ALL_RECALLED_TERM_ORDER = (
    "Intercept",
    "ValueAll",
    "ValueRecalled",
    "ValueAll:ValueRecalled",
)


def build_all_recalled_table(
    res: AnaResult, *, condition: str | None = None
) -> pd.DataFrame:
    """One row per list, with the whole-list value and the recalled value.

    ``ValueRecalled`` here is the **unweighted** mean value of the recalled
    items (``act_rec_list``) -- *not* the list-share-weighted ``ValueRecallW``
    that `build_list_table` produces.  The two are easy to confuse and give
    quite different coefficients; this is the form the original analysis used.

    Unlike the recalled/non-recalled decomposition, the two predictors here do
    not sum to anything: ``ValueAll`` already contains the recalled items, so
    the pair is collinear by construction and the recalled term measures what
    it adds *beyond* the list as a whole.

    Every list with an estimate, a list value and at least one recall is kept;
    there is no ``nrec`` restriction.
    """
    sources = {"rand": res.rand_, "temp": res.temp}
    if condition is not None:
        sources = {condition: sources[condition]}

    frames = []
    for label, cond in sources.items():
        usable = (
            np.isfinite(cond.est_list)
            & np.isfinite(cond.act_pres_list)
            & np.isfinite(cond.act_rec_list)
        )
        si, li = np.nonzero(usable)
        if si.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "Subject": si,
                    "ListType": label,
                    "ValueEst": cond.est_list[si, li],
                    "ValueAll": cond.act_pres_list[si, li],
                    "ValueRecalled": cond.act_rec_list[si, li],
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["Subject", "ListType", "ValueEst", "ValueAll", "ValueRecalled"]
        )
    return pd.concat(frames, ignore_index=True)


def all_vs_recalled(
    res: AnaResult, *, by_condition: bool = False, center: bool = False
) -> pd.DataFrame:
    """Fit ``ValueEst ~ 1 + ValueAll * ValueRecalled + (1|Subject)``.

    With ``by_condition=False`` this is a single model over every list, which
    is the form the original analysis reported.  With ``by_condition=True`` it
    is fit separately for random and autocorrelated lists.

    Predictors are left uncentered by default, matching the previously
    published coefficients (0.902 / 0.155).

    ``center=True`` is worth knowing about here in particular: because the two
    predictors interact, the uncentered coefficient on ``ValueAll`` is its
    effect where ``ValueRecalled`` is **zero**, which no list comes close to.
    Centered, the recalled term is 0.091 (p=0.161) rather than 0.155 (p=0.036)
    -- the same fit, with identical R-squared and interaction, read from a
    reference point inside the data.
    """
    groups = (("rand", "rand"), ("temp", "temp")) if by_condition else ((None, "both"),)

    centers = None
    if center:
        pooled = build_all_recalled_table(res)
        if not pooled.empty:
            _, centers = center_predictors(pooled, ALL_RECALLED_PREDICTORS)

    rows = []
    for condition, label in groups:
        table = build_all_recalled_table(res, condition=condition)
        if table.empty:
            continue
        if centers:
            table, _ = center_predictors(table, ALL_RECALLED_PREDICTORS, centers)
        fit = fit_mixed(table, ALL_RECALLED_MODEL)
        rows += _coefficient_rows(
            fit,
            centers,
            condition=label,
            list_type=CONDITION_LABELS.get(label, "All lists"),
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {term: i for i, term in enumerate(ALL_RECALLED_TERM_ORDER)}
    frame["_order"] = frame["term"].map(order).fillna(len(order))
    return (
        frame.sort_values(["condition", "_order"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def format_all_vs_recalled(table: pd.DataFrame, title: str) -> str:
    if table.empty:
        return f"== {title} ==\n  (no lists matched)"
    lines = [f"== {title} ==", _centering_note(table)]
    for condition, block in table.groupby("condition", sort=False):
        head = block.iloc[0]
        lines.append(
            f"  {head['list_type']:9s} n={int(head['n_lists'])} lists, "
            f"{int(head['n_subjects'])} subjects, "
            f"R2={head['R2']:.3f} conditional / {head['R2_marginal']:.3f} marginal"
        )
        for _, row in block.iterrows():
            lines.append(
                f"      {row['label']:30s} {row['estimate']:+9.4f}  "
                f"t={row['t']:+7.3f}  p={row['p']:.4g}"
            )
    return "\n".join(lines)
