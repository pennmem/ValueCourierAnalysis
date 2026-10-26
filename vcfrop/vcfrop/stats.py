"""Port of the statistics in stats_vaviv2.m (figure code excluded).

MATLAB's ``std`` normalizes by N-1, so every standard deviation here uses
``ddof=1``. ``corrcoef`` and ``ttest`` are paired/Pearson as in the source.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def loftus_masson(dat):
    """Loftus-Masson within-subject 95% confidence intervals.

    Direct translation of LoftusMasson.m (itself after Kahana's Pascal
    implementation of Loftus & Masson, 1994, PB&R 1(4):476-490). Rows are
    subjects, columns are the within-subject conditions.

    These intervals support comparison *between* the plotted conditions; they
    are not intervals on the individual means. Assumes sphericity.

    Returns
    -------
    (n_conditions,) array
        Half-width of the interval for each column. NaN where the variance
        estimator goes negative, which it can for near-zero interaction.
    """
    dat = np.asarray(dat, dtype=np.float64)
    n_rows, n_cols = dat.shape

    # Normalize out each subject's overall level.
    grand_mean = np.nanmean(dat)
    subj_mean = np.nanmean(dat, axis=1, keepdims=True)
    dat = dat - (subj_mean - grand_mean)

    tc = np.nansum(dat, axis=0)
    n_subj = np.sum(~np.isnan(dat), axis=0)
    ts = np.nansum(dat, axis=1)
    n_cond = np.sum(~np.isnan(dat), axis=1)

    total = np.nansum(dat)
    n_valid = int(np.sum(~np.isnan(dat)))

    ss_t = np.nansum(dat**2) - total**2 / n_valid
    ss_c = np.sum(tc**2 / n_subj) - total**2 / n_valid
    ss_s = np.sum(ts**2 / n_cond) - total**2 / n_valid
    ss_sxc = ss_t - ss_s - ss_c

    n_sub_valid = n_subj.sum() / n_cols
    n_cond_valid = n_cond.sum() / n_rows

    ms_sxc = ss_sxc / (n_valid - n_sub_valid - n_cond_valid - 1)
    ms_w = (np.nansum(dat**2, axis=0) - tc**2 / n_subj) / (n_subj - 1)

    estimator = (n_cond_valid / (n_cond_valid - 1)) * (ms_w - ms_sxc / n_cond_valid)
    with np.errstate(invalid="ignore"):
        return np.sqrt(estimator / n_subj) * sps.t.ppf(0.975, n_rows - 1)


def _mean_sd(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.mean(x)), float(np.std(x, ddof=1))


def run_stats(res, out=print):
    """Compute and report the stats_vaviv2 results.

    Parameters
    ----------
    res : dict
        Output of :func:`vcfrop.analysis.ana_vavi_e2frop`, or a reference
        loaded by :func:`vcfrop.matio.load_res`.
    out : callable
        Line sink; defaults to ``print``.

    Returns
    -------
    dict
        Every reported quantity, for programmatic comparison.
    """
    results = {}

    # P(recall) per subject, added as a field by the MATLAB script itself.
    prec_sub = np.nanmean(res["both"]["spcs"], axis=1)
    results["prec_sub"] = prec_sub

    # -- sanity checks on performance ---------------------------------------
    m, sd = _mean_sd(prec_sub)
    results["prec"] = (m, sd)
    out(f"P(recall): M={m:1.2f},SD={sd:1.2f} ")

    m, sd = _mean_sd(res["both"]["abserr_pres_sub"])
    results["abserr_pres"] = (m, sd)
    out(f"Pres value error: M={m:1.2f},SD={sd:1.2f} ")

    m, sd = _mean_sd(res["both"]["abserr_rec_item_sub"])
    results["abserr_rec_item"] = (m, sd)
    out(f"Item rec value error: M={m:1.2f},SD={sd:1.2f} ")

    # -- across tasks --------------------------------------------------------
    r, p = sps.pearsonr(res["both"]["abserr_pres_sub"].ravel(), prec_sub)
    results["corr_listerr_precall"] = (float(r), float(p))
    out(f"both: corr b/t abs_err LIST and p(recall), r={r:1.4f}, p={p:1.4f} ")

    r, p = sps.pearsonr(
        res["both"]["abserr_pres_sub"].ravel(),
        res["both"]["abserr_rec_item_sub"].ravel(),
    )
    results["corr_listerr_itemerr"] = (float(r), float(p))
    out(
        "both: corr b/t abs_err LIST and avg abs error for recalled items, "
        f"r={r:1.4f}, p={p:1.4f} "
    )

    # -- differences by list type (paired) -----------------------------------
    def paired(label, a, b, key):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        res_t = sps.ttest_rel(a, b)
        lo, hi = res_t.confidence_interval()
        results[key] = (float(res_t.pvalue), float(lo), float(hi))
        out(f"{label}, p={res_t.pvalue:1.4f}, CI=[{lo:1.3f},{hi:1.3f}] ")

    paired(
        "abs err: rand>temp",
        res["rand"]["abserr_pres_sub"],
        res["temp"]["abserr_pres_sub"],
        "ttest_abserr",
    )
    paired(
        "p(recall): rand>temp",
        np.nanmean(res["rand"]["spcs"], axis=1),
        np.nanmean(res["temp"]["spcs"], axis=1),
        "ttest_precall",
    )
    paired(
        "temp clusts: rand>temp",
        res["rand"]["temp_clusts"],
        res["temp"]["temp_clusts"],
        "ttest_tempclust",
    )

    return results


# ===================================================================
# Group contrasts, ported from ../vcbeh/src/vcbeh/stats/contrasts.py
#
# These sit alongside run_stats rather than replacing it: run_stats is the
# faithful stats_vaviv2.m port and stays as it is.
# ===================================================================


def recall_probability(cond):
    """Mean recall probability per subject, averaged over serial positions."""
    return np.nanmean(cond["spcs"], axis=1)


@dataclass
class Contrast:
    """A paired random-vs-autocorrelated comparison, reported two ways.

    The MATLAB original used a paired t-test. Whether that is appropriate
    depends on the sample, so the Wilcoxon signed-rank result is carried
    alongside it together with a Shapiro-Wilk check on the paired differences.
    Reporting all three makes the choice of test auditable instead of silent;
    which one to quote is a judgement about this data, not a default.
    """

    name: str
    n: int
    median_diff: float
    w: float
    p: float          # Wilcoxon signed-rank
    mean_diff: float
    t: float
    df: int
    t_p: float
    shapiro_p: float
    ci_low: float
    ci_high: float

    @property
    def differences_are_normal(self):
        return self.shapiro_p >= 0.05


def _paired(name, x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    diff = x - y

    signed_rank = sps.wilcoxon(x, y)
    parametric = sps.ttest_rel(x, y)
    df = diff.size - 1
    se = diff.std(ddof=1) / np.sqrt(diff.size)
    crit = sps.t.ppf(0.975, df)

    return Contrast(
        name=name,
        n=int(diff.size),
        median_diff=float(np.median(diff)),
        w=float(signed_rank.statistic),
        p=float(signed_rank.pvalue),
        mean_diff=float(diff.mean()),
        t=float(parametric.statistic),
        df=df,
        t_p=float(parametric.pvalue),
        shapiro_p=float(sps.shapiro(diff).pvalue),
        # Interval around the mean difference, so it pairs with the t-test.
        ci_low=float(diff.mean() - crit * se),
        ci_high=float(diff.mean() + crit * se),
    )


def list_type_contrasts(res):
    """Random vs autocorrelated lists, paired within subject."""
    rows = [
        _paired(
            "|value estimation error|",
            res["rand"]["abserr_pres_sub"],
            res["temp"]["abserr_pres_sub"],
        ),
        _paired(
            "recall probability",
            recall_probability(res["rand"]),
            recall_probability(res["temp"]),
        ),
        _paired(
            "temporal clustering",
            res["rand"]["temp_clusts"],
            res["temp"]["temp_clusts"],
        ),
    ]
    return pd.DataFrame([vars(r) for r in rows])


def descriptives(res):
    """Across-subject means and SDs for the headline measures (all lists)."""
    both = res["both"]
    measures = {
        "|value estimation error|": np.asarray(both["abserr_pres_sub"]).ravel(),
        "recall probability": recall_probability(both),
        "temporal clustering": np.asarray(both["temp_clusts"]).ravel(),
        "signed value error": np.nanmean(both["signerr_pres_list"], axis=1),
        "|per-item value error|": np.asarray(both["abserr_rec_item_sub"]).ravel(),
    }
    return pd.DataFrame(
        [
            {
                "measure": name,
                "n": int(np.isfinite(v).sum()),
                "mean": float(np.nanmean(v)),
                "sd": float(np.nanstd(v, ddof=1)),
            }
            for name, v in measures.items()
        ]
    )


def error_recall_correlation(res):
    """Does estimating list value badly go with recalling less?"""
    both = res["both"]
    err = np.asarray(both["abserr_pres_sub"], dtype=np.float64).ravel()
    prec = recall_probability(both)
    ok = np.isfinite(err) & np.isfinite(prec)

    r, p = sps.pearsonr(err[ok], prec[ok])
    slope, intercept = np.polyfit(err[ok], prec[ok], 1)
    return {
        "r": float(r),
        "p": float(p),
        "n": int(ok.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def format_report(res):
    """Text summary of the contrasts, descriptives and the correlation."""
    lines = ["== Random vs autocorrelated lists (Wilcoxon signed-rank) =="]
    for _, row in list_type_contrasts(res).iterrows():
        flag = "" if row["shapiro_p"] >= 0.05 else "  [differences non-normal]"
        lines.append(
            f"  {row['name']:26s} median diff={row['median_diff']:+7.4f}  "
            f"W={row['w']:7.1f}  p={row['p']:.4f}  n={int(row['n'])}{flag}"
        )
        lines.append(
            f"  {'':26s} (paired t: mean diff={row['mean_diff']:+7.4f}, "
            f"t({int(row['df'])})={row['t']:+6.3f}, p={row['t_p']:.4f}; "
            f"Shapiro p={row['shapiro_p']:.4f})"
        )

    lines += ["", "== Descriptives, all lists =="]
    for _, row in descriptives(res).iterrows():
        lines.append(
            f"  {row['measure']:26s} M={row['mean']:7.3f}  "
            f"SD={row['sd']:6.3f}  n={int(row['n'])}"
        )

    corr = error_recall_correlation(res)
    lines += ["", "== |value error| vs recall probability =="]
    lines.append(
        f"  r={corr['r']:+.4f}  p={corr['p']:.4g}  n={corr['n']}  "
        f"(fit: y = {corr['slope']:+.4f}x {corr['intercept']:+.4f})"
    )
    return "\n".join(lines)
