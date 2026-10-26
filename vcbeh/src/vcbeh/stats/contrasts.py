"""Group contrasts and correlations.

Port of ``stats_vaviv2.m``.  That script cannot run as written: it prints a
correlation against ``prec_sub``, which is only defined inside a block that had
been commented out, so the variable is undefined at line 31.  Here the quantity
is computed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, shapiro, t as student_t, ttest_rel, wilcoxon

from ..types import AnaResult


def recall_probability(cond) -> np.ndarray:
    """Mean recall probability per subject, averaged over serial positions."""
    return np.nanmean(cond.spcs, axis=1)


@dataclass
class Contrast:
    """A paired random-vs-autocorrelated comparison, by two tests.

    The Wilcoxon signed-rank result is the headline: on the raw 26-subject
    data the paired differences are strongly non-normal (Shapiro-Wilk p in the
    1e-3 range, skew up to +2.0), so the t-test's assumption does not hold
    there.  It held on the 22-subject .mat, where the differences are close to
    normal, which is why the MATLAB original used a t-test.

    Both tests and the normality check are reported, so the choice of test is
    auditable rather than silent.
    """

    name: str
    n: int
    median_diff: float
    w: float
    p: float  # Wilcoxon signed-rank
    mean_diff: float
    t: float
    df: int
    t_p: float
    shapiro_p: float
    ci_low: float
    ci_high: float

    @property
    def differences_are_normal(self) -> bool:
        return self.shapiro_p >= 0.05


def _paired(name: str, x: np.ndarray, y: np.ndarray) -> Contrast:
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    diff = x - y

    signed_rank = wilcoxon(x, y)
    parametric = ttest_rel(x, y)
    df = diff.size - 1
    se = diff.std(ddof=1) / np.sqrt(diff.size)
    crit = student_t.ppf(0.975, df)

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
        shapiro_p=float(shapiro(diff).pvalue),
        # Interval around the mean difference, so it pairs with the t-test.
        ci_low=float(diff.mean() - crit * se),
        ci_high=float(diff.mean() + crit * se),
    )


def list_type_contrasts(res: AnaResult) -> pd.DataFrame:
    """Random vs autocorrelated lists, paired within subject."""
    rows = [
        _paired(
            "|value estimation error|",
            res.rand_.abserr_pres_sub,
            res.temp.abserr_pres_sub,
        ),
        _paired(
            "recall probability",
            recall_probability(res.rand_),
            recall_probability(res.temp),
        ),
        _paired(
            "temporal clustering",
            res.rand_.temp_clusts,
            res.temp.temp_clusts,
        ),
    ]
    return pd.DataFrame([vars(r) for r in rows])


def descriptives(res: AnaResult) -> pd.DataFrame:
    """Across-subject means and SDs for the headline measures (all lists)."""
    prec = recall_probability(res.both)
    measures = {
        "|value estimation error|": res.both.abserr_pres_sub.ravel(),
        "recall probability": prec,
        "temporal clustering": res.both.temp_clusts.ravel(),
        "signed value error": np.nanmean(res.both.signerr_pres_list, axis=1),
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


def error_recall_correlation(res: AnaResult) -> dict[str, float]:
    """Does estimating value badly go with recalling less?

    The relationship this reports is the reason the pipeline exists, so it is
    computed from `abserr_pres_sub` and the subject-mean SPC directly rather
    than from a variable left over from an earlier session.
    """
    err = np.asarray(res.both.abserr_pres_sub).ravel()
    prec = recall_probability(res.both)
    ok = np.isfinite(err) & np.isfinite(prec)
    r, p = pearsonr(err[ok], prec[ok])
    slope, intercept = np.polyfit(err[ok], prec[ok], 1)
    return {
        "r": float(r),
        "p": float(p),
        "n": int(ok.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def format_report(res: AnaResult) -> str:
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
    lines.append("")
    lines.append("== Descriptives, all lists ==")
    for _, row in descriptives(res).iterrows():
        lines.append(
            f"  {row['measure']:26s} M={row['mean']:7.3f}  SD={row['sd']:6.3f}  n={row['n']}"
        )
    lines.append("")
    corr = error_recall_correlation(res)
    lines.append("== |value error| vs recall probability ==")
    lines.append(
        f"  r={corr['r']:+.4f}  p={corr['p']:.4g}  n={corr['n']}  "
        f"(fit: y = {corr['slope']:+.4f}x {corr['intercept']:+.4f})"
    )
    return "\n".join(lines)
