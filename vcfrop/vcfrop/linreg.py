"""Mixed-effects regressions of the value estimate on recalled / non-recalled value.

Ported from ``../vcbeh/src/vcbeh/stats/linreg.py`` and adapted to vcfrop's
dict-shaped results. The analyses are the same; the sample is not -- vcfrop has
10 subjects and 470 lists where vcbeh had 22-26 and 1080-1120, and there is no
raw event log to fall back on.

The question: a subject's estimate of a list's value could be built from the
items they recalled, from the items they did not, or from both. With ``nrec`` of
``ll`` items recalled, a full account predicts::

    ValueEst ~ (nrec/ll) * ValueRecalled + ((ll-nrec)/ll) * ValueNotRecalled

The two weighted predictors sum exactly to the whole list's value, so the
uncorrected model is this one with the two sources forced to count equally.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t

# statsmodels' L-BFGS can settle on a boundary solution here -- group variance
# exactly zero, intercept pinned -- while still reporting convergence. Every
# other optimizer finds a sensible fit, so all are tried and the highest
# likelihood wins. These are ML fits, so that is exactly the right tie-break.
_OPTIMIZERS = ("bfgs", "powell", "cg", "nm", "lbfgs")

ADDITIVE_MODEL = "ValueEst ~ ValueRecall + ValueNotRecall"
INTERACTION_MODEL = "ValueEst ~ ValueRecall * ValueNotRecall"

UNCORRECTED_MODEL = "ValueEst ~ ValuePres"
CORRECTED_MODEL = "ValueEst ~ ValueRecallW + ValueNotRecallW"
CORRECTED_INTERACTION_MODEL = "ValueEst ~ ValueRecallW * ValueNotRecallW"

UNCORRECTED_CONTRAST = "ValueEst ~ ValuePres * ListType"
CORRECTED_CONTRAST = "ValueEst ~ (ValueRecallW + ValueNotRecallW) * ListType"

UNCORRECTED_PREDICTORS = ("ValuePres",)
CORRECTED_PREDICTORS = ("ValueRecallW", "ValueNotRecallW")

ALL_RECALLED_MODEL = "ValueEst ~ ValueAll * ValueRecalled"
ALL_RECALLED_PREDICTORS = ("ValueAll", "ValueRecalled")
ALL_RECALLED_TERM_ORDER = (
    "Intercept",
    "ValueAll",
    "ValueRecalled",
    "ValueAll:ValueRecalled",
)

#: A sweep cell below this many lists cannot support a random-intercept fit.
#: vcbeh hard-coded nrec 7-13; on vcfrop's 470 lists those cells hold as few as
#: 7 lists, so the range is derived from the data instead.
MIN_LISTS_PER_SWEEP_CELL = 30

CONDITION_LABELS = {"rand": "Random", "temp": "Autocorr"}

TERM_LABELS = {
    "Intercept": "Intercept",
    "ValuePres": "Value of ALL studied items",
    "ValueRecall": "Value of RECALLED items",
    "ValueNotRecall": "Value of NON-RECALLED items",
    "ValueRecallW": "Value of RECALLED items",
    "ValueNotRecallW": "Value of NON-RECALLED items",
    "ValueAll": "Value from ALL studied items",
    "ValueRecalled": "Value from RECALLED items",
    "ValueAll:ValueRecalled": "Interaction (All x Recalled)",
    "ValueRecallW:ValueNotRecallW": "Interaction (Recalled x Non-recalled)",
}

SHORT_TERM_LABELS = {
    "ValuePres": "ALL studied",
    "ValueRecallW": "RECALLED",
    "ValueNotRecallW": "NON-RECALLED",
    "ValueAll": "ALL studied",
    "ValueRecalled": "RECALLED",
    "ValueRecallW:ValueNotRecallW": "REC x NON-REC",
}

_LIST_TYPE_TERM = "ListType[T.temp]"


def term_label(term):
    """Human-readable name for a model term, interactions included."""
    if term in TERM_LABELS:
        return TERM_LABELS[term]
    if term == _LIST_TYPE_TERM:
        return "List type (Autocorr vs Random)"
    if term.endswith(":" + _LIST_TYPE_TERM):
        base = term[: -len(":" + _LIST_TYPE_TERM)]
        return f"{SHORT_TERM_LABELS.get(base, base)}  x  list type"
    return term


# ---------------------------------------------------------------- fitting


def _fit_best_likelihood(model, formula):
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
        raise RuntimeError(f"no optimizer converged for {formula!r}")
    return best


def fit_mixed(table, formula):
    """Fit ``formula`` with a random intercept per subject, matching ``fitlme``.

    Three defaults have to be forced, because statsmodels and MATLAB disagree
    on all of them:

    * ``reml=False`` -- ``fitlme`` defaults to maximum likelihood.
    * p-values from a *t* on residual degrees of freedom rather than the z-test
      statsmodels reports; ``fitlme`` defaults to ``DFMethod='residual'``.
    * ``Rsquared.Ordinary`` is **conditional**, including the random effects,
      where ``statsmodels.fittedvalues`` excludes them.
    """
    if table.empty:
        raise ValueError("no lists matched; cannot fit")

    model = smf.mixedlm(formula, table, groups=table["Subject"])
    fit = _fit_best_likelihood(model, formula)

    params = fit.fe_params
    names = list(params.index)
    se = fit.bse[names]
    nobs = int(table.shape[0])
    df_resid = nobs - len(names)
    tvals = params / se
    pvals = 2 * student_t.sf(np.abs(tvals), df_resid)

    # Both R-squareds are built from the design matrix rather than from
    # `fit.fittedvalues`, because what that property contains is not stable:
    # in statsmodels 0.14.5 it already includes the random effects. vcbeh's
    # version adds the BLUPs to it, which double-counts them -- its
    # "conditional" R2 is therefore wrong and its "marginal" is in fact the
    # conditional one. On vcbeh's data the two sat within 0.005 of each other
    # so the error was invisible; here they are 0.68 vs 0.49. Deriving both
    # from `model.exog` sidesteps the question entirely.
    intercepts = {
        g: float(np.asarray(v).ravel()[0]) for g, v in fit.random_effects.items()
    }
    fixed_pred = np.asarray(model.exog) @ params.to_numpy()
    conditional_pred = fixed_pred + table["Subject"].map(intercepts).to_numpy()

    observed = table["ValueEst"].to_numpy()
    sst = float(np.sum((observed - observed.mean()) ** 2))
    sse = float(np.sum((observed - conditional_pred) ** 2))
    sse_marginal = float(np.sum((observed - fixed_pred) ** 2))

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


def _coefficient_rows(fit, centers=None, **extra):
    """Flatten a :func:`fit_mixed` result into one tidy row per coefficient."""
    crit = student_t.ppf(0.975, fit["df_resid"])
    return [
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
        for name, est, se, t, p in zip(
            fit["names"], fit["estimates"], fit["se"], fit["t"], fit["p"]
        )
    ]


def center_predictors(table, columns, centers=None):
    """Subtract each predictor's mean; return the table and the constants used.

    Centring moves the intercept from a predictor value of zero -- far outside
    the observed range, list values here run about 8 to 45 -- to the mean, so
    the intercept and any categorical main effect describe a typical list
    instead of an extrapolation. Slopes, interactions, fitted values and
    R-squared are untouched.

    Pass ``centers`` to reuse constants: per-condition fits must share one set
    or their intercepts sit at different points and cannot be compared.
    """
    if centers is None:
        centers = {name: float(table[name].mean()) for name in columns}
    centered = table.copy()
    for name, mean in centers.items():
        centered[name] = centered[name] - mean
    return centered, centers


class InconsistentRecallCountError(ValueError):
    """``act_nrec_list`` does not agree with the value decomposition.

    Raised when ``ValueRecallW + ValueNotRecallW`` does not reconstruct
    ``ValuePres``. Every model in this module rests on that identity, so a
    silent violation would quietly invalidate all of them.

    On vcfrop data the cause is always the same: ``act_nrec_list`` is built
    through the output-position weighting, and under the default MATLAB-
    faithful path that loop reads ``max_op`` from the *unfiltered* recalls
    array (ana_vavi_e2frop.m:114). For the ``rand`` and ``temp`` conditions the
    rows then misalign and the recall counts belong to unrelated trials.

    Fix it at the source::

        res = ana_vavi_e2frop(path, fix_op_weights=True)

    The ``both`` condition is unaffected, because nothing is filtered out of it.
    """


def _require_consistent_nrec(table, list_length):
    """Guard the identity every model here depends on."""
    if table.empty:
        return table
    residual = np.abs(
        table["ValueRecallW"] + table["ValueNotRecallW"] - table["ValuePres"]
    )
    bad = int((residual > 1e-9).sum())
    if bad:
        raise InconsistentRecallCountError(
            f"{bad} of {len(table)} lists fail "
            "ValueRecallW + ValueNotRecallW == ValuePres "
            f"(largest residual {residual.max():.3g}). "
            "Re-run the analysis with fix_op_weights=True."
        )
    return table


# ------------------------------------------------- one model per nrec


def build_table(res, nrec, list_length, weight_recall):
    """Lists on which exactly ``nrec`` items were recalled.

    ``ValueNotRecall`` is always scaled by the fraction of the list it covers.
    ``ValueRecall`` is scaled to match only when ``weight_recall`` is set --
    the two original MATLAB scripts differ on exactly this point.
    """
    rec_coeff = nrec / list_length if weight_recall else 1.0
    nonrec_coeff = (list_length - nrec) / list_length

    frames = []
    for label in ("rand", "temp"):
        cond = res[label]
        subject_idx, list_idx = np.nonzero(cond["act_nrec_list"] == nrec)
        if subject_idx.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "Subject": subject_idx,
                    "ValueEst": cond["est_list"][subject_idx, list_idx],
                    "ValueRecall": rec_coeff
                    * cond["act_rec_list"][subject_idx, list_idx],
                    "ValueNotRecall": nonrec_coeff
                    * cond["act_nonrec_list"][subject_idx, list_idx],
                    "ListType": label,
                }
            )
        )

    columns = ["Subject", "ValueEst", "ValueRecall", "ValueNotRecall", "ListType"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["ValueEst", "ValueRecall", "ValueNotRecall"])
        .reset_index(drop=True)
    )


def recall_count_sizes(res, list_length):
    """Lists available at each recall count, pooled over both list types.

    Printed alongside any sweep: a coefficient from a cell of eight lists is
    not comparable to one from a cell of sixty, and the difference is invisible
    in the fitted table.
    """
    rows = []
    for nrec in range(0, int(list_length) + 1):
        table = build_table(res, nrec, list_length, weight_recall=True)
        rows.append(
            {
                "nrec": nrec,
                "n_lists": int(table.shape[0]),
                "n_subjects": int(table["Subject"].nunique()) if len(table) else 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["viable"] = frame["n_lists"] >= MIN_LISTS_PER_SWEEP_CELL
    return frame


def viable_recall_counts(res, list_length, min_lists=MIN_LISTS_PER_SWEEP_CELL):
    """Recall counts with enough lists to fit, as a sorted list."""
    sizes = recall_count_sizes(res, list_length)
    return sizes.loc[sizes["n_lists"] >= min_lists, "nrec"].tolist()


def sweep(res, list_length, nrec_values, formula, weight_recall):
    """Fit one model per recall count and collect the coefficients."""
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


def recalled_vs_nonrecalled(res, list_length=15, nrec_values=None):
    """Additive model, both terms weighted by the list share they cover."""
    if nrec_values is None:
        nrec_values = viable_recall_counts(res, list_length)
    return sweep(res, list_length, nrec_values, ADDITIVE_MODEL, weight_recall=True)


def recalled_unweighted(res, list_length=15, nrec_values=None):
    """Interaction model with an unweighted recalled-value term."""
    if nrec_values is None:
        nrec_values = viable_recall_counts(res, list_length)
    return sweep(res, list_length, nrec_values, INTERACTION_MODEL, weight_recall=False)


# ------------------------------------------------ per-condition models


def build_list_table(res, list_length, condition=None, partial_only=False):
    """One row per list, with the recall decomposition weighted per list.

    Unlike :func:`build_table` this does not fix ``nrec``: each list is
    weighted by its own recall count, so one model spans every list instead of
    one model per recall count.

    ``partial_only`` drops lists where nothing or everything was recalled --
    ``act_nonrec_list`` is undefined for a fully recalled list, and a list with
    no recalls has no recalled value.
    """
    ll = int(list_length)
    labels = ("rand", "temp") if condition is None else (condition,)

    frames = []
    for label in labels:
        cond = res[label]
        nrec = cond["act_nrec_list"]
        # A list with no recalls has nrec NaN. That disqualifies it from the
        # decomposition only -- the uncorrected model never uses nrec, so
        # requiring it there would drop the list for no reason.
        usable = np.isfinite(cond["est_list"]) & np.isfinite(cond["act_pres_list"])
        if partial_only:
            usable &= (
                np.isfinite(nrec)
                & (nrec > 0)
                & (nrec < ll)
                & np.isfinite(cond["act_rec_list"])
                & np.isfinite(cond["act_nonrec_list"])
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
                    "ValueEst": cond["est_list"][si, li],
                    "ValuePres": cond["act_pres_list"][si, li],
                    "nrec": n,
                    "ValueRecallW": (n / ll) * cond["act_rec_list"][si, li],
                    "ValueNotRecallW": ((ll - n) / ll)
                    * cond["act_nonrec_list"][si, li],
                }
            )
        )

    columns = [
        "Subject", "ListType", "ValueEst", "ValuePres", "nrec",
        "ValueRecallW", "ValueNotRecallW",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)

    table = pd.concat(frames, ignore_index=True)
    if partial_only:
        # Only meaningful where the decomposition is defined; the uncorrected
        # model does not use nrec at all, so leave that path alone.
        _require_consistent_nrec(table, ll)
    return table


def condition_models(
    res, list_length=15, corrected=False, partial_only=None,
    center=False, interaction=False,
):
    """One mixed model per list type; one tidy row per coefficient.

    ``corrected=False`` regresses the estimate on the list's true value.
    ``corrected=True`` replaces that with the recalled / non-recalled
    decomposition, each term weighted by the share of the list it covers.
    """
    if not corrected:
        if interaction:
            raise ValueError(
                "the uncorrected model has a single predictor, so there is "
                "nothing for it to interact with"
            )
        formula, predictors = UNCORRECTED_MODEL, UNCORRECTED_PREDICTORS
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
        rows += _coefficient_rows(
            fit_mixed(table, formula),
            centers,
            condition=label,
            list_type=CONDITION_LABELS[label],
        )
    return pd.DataFrame(rows)


def condition_contrast(res, list_length=15, corrected=False, center=False):
    """Pooled model with a ``ListType`` interaction.

    Fitting the conditions separately shows what each looks like but cannot say
    whether they *differ*; the interaction terms here are that test.
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


# --------------------------------- the ValueAll x ValueRecalled model


def build_all_recalled_table(res, condition=None):
    """One row per list, with the whole-list value and the recalled value.

    ``ValueRecalled`` is the **unweighted** mean value of the recalled items,
    not the list-share-weighted ``ValueRecallW``. The two are easy to confuse
    and give quite different coefficients.

    The two predictors here do not sum to anything: ``ValueAll`` already
    contains the recalled items, so the pair is collinear by construction and
    the recalled term measures what it adds *beyond* the list as a whole.
    """
    labels = ("rand", "temp") if condition is None else (condition,)

    frames = []
    for label in labels:
        cond = res[label]
        usable = (
            np.isfinite(cond["est_list"])
            & np.isfinite(cond["act_pres_list"])
            & np.isfinite(cond["act_rec_list"])
        )
        si, li = np.nonzero(usable)
        if si.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "Subject": si,
                    "ListType": label,
                    "ValueEst": cond["est_list"][si, li],
                    "ValueAll": cond["act_pres_list"][si, li],
                    "ValueRecalled": cond["act_rec_list"][si, li],
                }
            )
        )

    columns = ["Subject", "ListType", "ValueEst", "ValueAll", "ValueRecalled"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def all_vs_recalled(res, by_condition=False, center=False):
    """Fit ``ValueEst ~ 1 + ValueAll * ValueRecalled + (1|Subject)``.

    Because the two predictors interact, the uncentred coefficient on
    ``ValueAll`` is its effect where ``ValueRecalled`` is **zero**, which no
    list comes close to. ``center=True`` reads the same fit from a reference
    point inside the data.
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
        rows += _coefficient_rows(
            fit_mixed(table, ALL_RECALLED_MODEL),
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


# --------------------------------------------------------------- output


def _centering_note(table):
    if "centered_at" not in table.columns:
        return "  predictors uncentred: intercept is at a predictor value of 0"
    seen = table.dropna(subset=["centered_at"]).drop_duplicates(subset="term")
    if seen.empty:
        return "  predictors uncentred: intercept is at a predictor value of 0"
    parts = ", ".join(f"{row.term}={row.centered_at:.2f}" for row in seen.itertuples())
    return f"  predictors mean-centred ({parts}); intercept is at a typical list"


def format_sweep(table, title):
    if table.empty:
        return f"== {title} ==\n  (no lists matched)"
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        body = table.to_string(index=False, float_format=lambda v: f"{v:8.4f}")
    return f"== {title} ==\n{body}"


def format_condition_models(table, title):
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


def format_contrast(table, title):
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


def format_all_vs_recalled(table, title):
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
