"""Port of linreg_estval_presrecallval.m.

Flattens the per-subject/per-list matrices into a long table and fits

    ValueEst ~ 1 + ValueRecall * ValuePres + (1 | Subject)

MATLAB/statsmodels equivalence, handled explicitly below:

* ``fitlme`` defaults to ``FitMethod='ML'``; statsmodels defaults to REML. The
  fit here passes ``reml=False``.
* ``fitlme`` reports t-statistics against residual degrees of freedom
  (``DFMethod='residual'``); ``MixedLM`` reports z-statistics. p-values are
  recomputed as two-sided t with ``n_obs - n_fixed`` df. That reproduces
  MATLAB's default, which is itself an approximation for a mixed model.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

FORMULA = "ValueEst ~ ValueRecall * ValuePres"

# Row order and labels for the LaTeX table (MATLAB's inds = [1;3;2;4],
# converted to 0-based). Studied is reported before Recalled.
TABLE_ROWS = [
    (0, "Intercept"),
    (2, "Tip, Studied Items"),
    (1, "Tip, Recalled Items"),
    (3, "Interaction: Tip Studied and Tip Recalled"),
]

_SOURCES = {
    "ValueEst": "est_list",
    "ValueRecall": "act_rec_list",
    "ValuePres": "act_pres_list",
    "WeValueRecall": "act_weightrec_list",
    "AsValueRecall": "act_wtascrec_list",
}


def build_table(res):
    """Long-format table, one row per (subject, list).

    Lists are selected by ``signerr_pres_list`` being non-NaN, i.e. lists the
    subject actually has, exactly as the MATLAB loop does. Rows are emitted
    subject-major, rand before temp.
    """
    nsub = res["rand"]["signerr_pres_list"].shape[0]
    chunks = []

    for si in range(nsub):
        for list_type in ("rand", "temp"):
            cond = res[list_type]
            keep = ~np.isnan(cond["signerr_pres_list"][si])
            if not keep.any():
                continue
            chunk = {"Subject": np.full(keep.sum(), si + 1, dtype=float)}
            for col, field in _SOURCES.items():
                chunk[col] = cond[field][si][keep]
            # Built by the MATLAB script but left out of its model: presentation
            # and recall value estimates did not differ by list type.
            chunk["ListType"] = np.repeat(list_type, keep.sum())
            chunks.append(pd.DataFrame(chunk))

    return pd.concat(chunks, ignore_index=True)


def fit(table, formula=FORMULA):
    """Fit the mixed model and return ``(result, coefficients_frame)``."""
    model = smf.mixedlm(
        formula, table, groups=table["Subject"], missing="drop"
    )
    fitted = model.fit(reml=False)

    names = list(fitted.fe_params.index)
    estimate = fitted.fe_params.to_numpy()
    se = fitted.bse_fe.to_numpy()

    # MATLAB's DFMethod='residual'.
    n_obs = int(fitted.nobs)
    df_resid = n_obs - len(names)
    tstat = estimate / se
    pvalue = 2 * sps.t.sf(np.abs(tstat), df_resid)

    coefs = pd.DataFrame(
        {
            "Name": names,
            "Estimate": estimate,
            "SE": se,
            "tStat": tstat,
            "DF": df_resid,
            "pValue": pvalue,
        }
    )
    return fitted, coefs


def write_latex_table(coefs, path):
    """Write the coefficient table.

    The MATLAB version never emits the closing ``\\end{tabular}`` /
    ``\\end{table}``, leaving an uncompilable file; they are written here.
    """
    lines = [
        "\\begin{table} \n",
        "\\begin{tabular}{lll} \n",
        "Effect & Mean (SE) & $p$  \\\\ \\hline \n",
    ]
    for idx, title in TABLE_ROWS:
        row = coefs.iloc[idx]
        lines.append(
            f"{title} & {row.Estimate:1.3f} ({row.SE:1.3f}) & {row.pValue:1.3f}"
            " \\\\ \\hline \n"
        )
    lines += ["\\end{tabular} \n", "\\end{table} \n"]

    with open(path, "w") as fh:
        fh.writelines(lines)
    return path


def run_regression(res, latex_path=None, formula=FORMULA, out=print):
    """Build the table, fit the model, optionally write the LaTeX table."""
    table = build_table(res)
    fitted, coefs = fit(table, formula)

    out(f"n observations: {int(fitted.nobs)} (of {len(table)} rows)")
    out(f"formula: {formula} + (1 | Subject)   [ML]")
    out("")
    out(coefs.to_string(index=False))
    out("")
    out(f"random intercept variance: {float(fitted.cov_re.iloc[0, 0]):.4f}")
    out(f"residual variance:         {float(fitted.scale):.4f}")

    if latex_path:
        write_latex_table(coefs, latex_path)
        out(f"\nwrote {latex_path}")

    return table, fitted, coefs
