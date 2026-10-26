"""End-to-end checks against the saved MATLAB result.

res_083026_xs0.mat was produced by an earlier revision of ana_vavi_e2frop.m and
does not carry act_nrec_list or act_nonrec_list; those are skipped here rather
than treated as failures.
"""

from pathlib import Path

import numpy as np
import pytest

from vcfrop.analysis import ana_vavi_e2frop
from vcfrop.matio import load_data, load_res
from vcfrop.regression import build_table, fit
from vcfrop.stats import run_stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_083026_xs0.mat"
REFERENCE = ROOT / "res_083026_xs0.mat"
DATA_WS0 = ROOT / "data_083026_ws0.mat"

CONDITIONS = ("both", "rand", "temp")
# Fields the saved res_083026_xs0.mat predates. act_nonrec_stray is the
# MATLAB line-132 artifact; the rest were added when the vcbeh analyses were
# brought over.
NOT_IN_REFERENCE = {
    "act_nrec_list",
    "act_nonrec_stray",
    "act_nonrec_list",
    "prec_byval",
    "abserr_wtrec_list",
    "abserr_wtrec_sub",
    "baseerr_rec_list",
}


@pytest.fixture(scope="module")
def res():
    return ana_vavi_e2frop(DATA)


@pytest.fixture(scope="module")
def ref():
    return load_res(REFERENCE)


def _fields(res, cond):
    return [f for f in res[cond] if f not in NOT_IN_REFERENCE]


@pytest.mark.parametrize("cond", CONDITIONS)
def test_every_reference_field_is_reproduced(res, ref, cond):
    for field in _fields(res, cond):
        got = np.asarray(res[cond][field], dtype=np.float64)
        want = np.asarray(ref[cond][field], dtype=np.float64)

        assert got.shape == want.shape, f"{cond}.{field} shape"
        np.testing.assert_array_equal(
            np.isnan(got), np.isnan(want), err_msg=f"{cond}.{field} NaN placement"
        )
        finite = ~np.isnan(got)
        np.testing.assert_allclose(
            got[finite],
            want[finite],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{cond}.{field} values",
        )


@pytest.mark.parametrize("cond", CONDITIONS)
def test_reference_has_no_extra_fields(res, ref, cond):
    assert set(ref[cond]) <= set(res[cond])


def test_unilist_matches(res, ref):
    np.testing.assert_array_equal(res["unilist"], ref["unilist"])


def test_inf_sentinels_survive_only_in_est_rec_list(res, ref):
    """est_rec_list is left out of the Inf->NaN conversion, in MATLAB too.

    ana_vavi_e2frop.m:165-169 converts act_rec_list, act_weightrec_list,
    act_wtascrec_list, abserr_rec_list_item and act_nrec_list, but not
    est_rec_list, so the no-recall sentinel survives there. The saved reference
    carries the same Inf in the same cells. Nothing downstream reads the field,
    but a nanmean over it would silently give Inf.
    """
    for cond in CONDITIONS:
        for field, arr in res[cond].items():
            leaked = np.isinf(np.asarray(arr, dtype=np.float64))
            if field == "est_rec_list":
                np.testing.assert_array_equal(
                    leaked, np.isinf(ref[cond][field]), err_msg=f"{cond}.{field}"
                )
            else:
                assert not leaked.any(), f"{cond}.{field} leaked an Inf sentinel"


# ------------------------------------------------------- output-position bugs


def test_fix_op_weights_is_inert_for_both_condition():
    """The row-index bug only bites once rows are filtered.

    For 'both' every row is kept, so the descending weights are unaffected.
    """
    buggy = ana_vavi_e2frop(DATA, fix_op_weights=False)
    fixed = ana_vavi_e2frop(DATA, fix_op_weights=True)
    np.testing.assert_allclose(
        buggy["both"]["act_weightrec_list"],
        fixed["both"]["act_weightrec_list"],
        equal_nan=True,
    )


def test_fix_op_weights_changes_ascending_weights():
    """The asc-from-dsc bug bites everywhere, including 'both'."""
    buggy = ana_vavi_e2frop(DATA, fix_op_weights=False)
    fixed = ana_vavi_e2frop(DATA, fix_op_weights=True)
    a = buggy["both"]["act_wtascrec_list"]
    b = fixed["both"]["act_wtascrec_list"]
    finite = ~np.isnan(a) & ~np.isnan(b)
    assert np.abs(a[finite] - b[finite]).max() > 1.0


def test_fixed_ascending_weights_are_a_ramp():
    """Corrected ascending weights should mirror the descending ones.

    A list recalled in full forward order has its descending and ascending
    weighted means straddle the plain mean symmetrically.
    """
    fixed = ana_vavi_e2frop(DATA, fix_op_weights=True)
    dsc = fixed["both"]["act_weightrec_list"]
    asc = fixed["both"]["act_wtascrec_list"]
    plain = fixed["both"]["act_rec_list"]
    finite = ~np.isnan(dsc) & ~np.isnan(asc) & ~np.isnan(plain)
    # Under the bug this relation does not hold; under the fix it does.
    assert np.allclose(
        (dsc[finite] + asc[finite]) / 2.0, plain[finite], atol=2.0
    )


# ------------------------------------------------------------- other dataset


def test_ws0_uses_the_incs0_branch():
    """data_083026_ws0 includes session 0, so maxnlist is 55, not 50."""
    data = load_data(DATA_WS0)
    assert (data.session == 0).any()

    res = ana_vavi_e2frop(DATA_WS0)
    for cond in CONDITIONS:
        assert res[cond]["est_list"].shape[1] == 55
        assert res[cond]["spcs"].shape[1] == data.list_length


def test_bysess_folds_session_into_subject():
    plain = ana_vavi_e2frop(DATA_WS0)
    bysess = ana_vavi_e2frop(DATA_WS0, bysess=True)
    assert len(bysess["unilist"]) > len(plain["unilist"])
    # subject + 1000 * session recovers the original subject with mod.
    np.testing.assert_array_equal(
        np.unique(bysess["unilist"] % 1000), plain["unilist"]
    )


# --------------------------------------------------------------- stats layer


def test_stats_agree_between_ported_and_reference_res(res, ref):
    got = run_stats(res, out=lambda _: None)
    want = run_stats(ref, out=lambda _: None)

    assert set(got) == set(want)
    for key in got:
        np.testing.assert_allclose(
            np.asarray(got[key], dtype=float),
            np.asarray(want[key], dtype=float),
            rtol=1e-9,
            atol=1e-12,
            err_msg=key,
        )


def test_regression_agrees_between_ported_and_reference_res(res, ref):
    _, coefs_got = fit(build_table(res))
    _, coefs_want = fit(build_table(ref))
    np.testing.assert_allclose(
        coefs_got[["Estimate", "SE", "pValue"]].to_numpy(),
        coefs_want[["Estimate", "SE", "pValue"]].to_numpy(),
        rtol=1e-6,
    )


def test_regression_drops_only_the_no_recall_list(res):
    """469 of 470 lists enter the model; the all-intrusion list has no ValueRecall."""
    table = build_table(res)
    assert len(table) == 470
    fitted, _ = fit(table)
    assert int(fitted.nobs) == 469
    assert fitted.converged


def test_regression_coefficient_order_matches_matlab_table_rows():
    from vcfrop.regression import TABLE_ROWS

    _, coefs = fit(build_table(ana_vavi_e2frop(DATA)))
    labelled = [(coefs.iloc[i].Name, title) for i, title in TABLE_ROWS]
    assert labelled == [
        ("Intercept", "Intercept"),
        ("ValuePres", "Tip, Studied Items"),
        ("ValueRecall", "Tip, Recalled Items"),
        ("ValueRecall:ValuePres", "Interaction: Tip Studied and Tip Recalled"),
    ]


# ------------------------------------------------- fields added from vcbeh


def test_act_nonrec_list_completes_the_value_decomposition(res):
    """nrec*recalled + (ll-nrec)*non-recalled == ll*all-studied, exactly.

    This is the identity the recalled/non-recalled regressions rest on: the two
    predictors must sum to the whole list's value. Checked on the 'both'
    condition, whose act_nrec_list is unaffected by the row-misalignment bug.
    """
    data = load_data(DATA)
    ll = data.list_length
    c = res["both"]

    nrec = c["act_nrec_list"]
    partial = ~np.isnan(nrec) & (nrec > 0) & (nrec < ll)
    assert partial.sum() == 462

    lhs = (
        nrec[partial] * c["act_rec_list"][partial]
        + (ll - nrec[partial]) * c["act_nonrec_list"][partial]
    ) / ll
    np.testing.assert_allclose(lhs, c["act_pres_list"][partial], atol=1e-12)


def test_act_nonrec_list_is_nan_when_everything_was_recalled(res):
    """No non-recalled items to average; must not leak the Inf sentinel."""
    data = load_data(DATA)
    ll = data.list_length
    for cond in CONDITIONS:
        c = res[cond]
        full = c["act_nrec_list"] == ll
        if full.any():
            assert np.isnan(c["act_nonrec_list"][full]).all()
        assert not np.isinf(c["act_nonrec_list"]).any()


def test_prec_byval_is_indexed_by_value(res):
    """Column v-1 holds value v, for values 1..50."""
    for cond in CONDITIONS:
        pv = res[cond]["prec_byval"]
        assert pv.shape == (10, 50)
        finite = pv[~np.isnan(pv)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_prec_byval_is_in_the_same_range_as_overall_recall(res):
    c = res["both"]
    overall = float(np.nanmean(c["spcs"]))
    by_value = float(np.nanmean(c["prec_byval"]))
    # An unweighted mean over values, so not identical to the overall rate,
    # but a denominator bug would put it orders of magnitude off.
    assert abs(by_value - overall) < 0.15


def test_baseerr_beats_nothing_but_the_real_estimate_beats_it(res):
    """|actual - 25| is the guess-the-midpoint baseline a real estimate must beat."""
    c = res["both"]
    baseline = float(np.nanmean(c["baseerr_rec_list"]))
    actual = float(np.nanmean(c["abserr_pres_sub"]))
    assert actual < baseline


def test_act_nonrec_stray_is_the_matlab_artifact(res):
    """The line-132 artifact is still reproduced, under its own name."""
    assert res["both"]["act_nonrec_stray"].shape == (215,)
    assert res["rand"]["act_nonrec_stray"].shape == (0,)
    assert res["temp"]["act_nonrec_stray"].shape == (0,)
