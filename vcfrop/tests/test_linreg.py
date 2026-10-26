"""Checks on the mixed-effects models ported from vcbeh.

The definitions were validated by running both implementations on vcbeh's own
``data_050526_45sess.mat``: estimates, standard errors, t and p agree to
0.00e+00 across every model and option. The R-squared columns deliberately do
NOT agree -- see :func:`test_conditional_r2_is_never_worse_than_marginal`.
"""

import numpy as np
import pytest

from vcfrop import linreg as L
from vcfrop.analysis import ana_vavi_e2frop
from vcfrop.matio import load_data

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_083026_xs0.mat"


@pytest.fixture(scope="module")
def res():
    """The decomposition models require corrected output-position weights.

    act_nrec_list is built through that weighting, and under the default
    MATLAB-faithful path its rows misalign for rand/temp -- see
    InconsistentRecallCountError.
    """
    return ana_vavi_e2frop(DATA, fix_op_weights=True)


@pytest.fixture(scope="module")
def res_matlab_faithful():
    return ana_vavi_e2frop(DATA)


@pytest.fixture(scope="module")
def ll():
    return load_data(DATA).list_length


# ------------------------------------------------------------- table shape


def test_weighted_predictors_sum_to_the_whole_list_value(res, ll):
    """ValueRecallW + ValueNotRecallW == ValuePres, exactly.

    This identity is what makes round 1 the special case of round 2 in which
    both sources are forced to count equally. If it does not hold, the two
    rounds are not comparable and nothing built on them means anything.
    """
    table = L.build_list_table(res, ll, partial_only=True)
    assert len(table) == 462
    np.testing.assert_allclose(
        table["ValueRecallW"] + table["ValueNotRecallW"],
        table["ValuePres"],
        atol=1e-12,
    )


def test_partial_only_drops_empty_and_complete_lists(res, ll):
    partial = L.build_list_table(res, ll, partial_only=True)
    assert partial["nrec"].between(1, ll - 1).all()

    everything = L.build_list_table(res, ll, partial_only=False)
    assert len(everything) > len(partial)


def test_build_all_recalled_uses_unweighted_recalled_value(res, ll):
    """ValueRecalled is act_rec_list itself, not the list-share-weighted form."""
    table = L.build_all_recalled_table(res)
    weighted = L.build_list_table(res, ll, partial_only=True)
    # The weighted term is scaled down by nrec/ll < 1, so its mean is smaller.
    assert table["ValueRecalled"].mean() > weighted["ValueRecallW"].mean()


def test_condition_filter_splits_the_lists(res, ll):
    rand = L.build_list_table(res, ll, condition="rand")
    temp = L.build_list_table(res, ll, condition="temp")
    both = L.build_list_table(res, ll)
    assert len(rand) + len(temp) == len(both)
    assert set(rand["ListType"]) == {"rand"}


# ------------------------------------------------------------------ fitting


def test_conditional_r2_is_never_worse_than_marginal(res, ll):
    """Adding the random intercepts cannot make the fit worse.

    vcbeh computes these from `fit.fittedvalues`, which in statsmodels 0.14.5
    already contains the random effects; adding the BLUPs to it double-counts
    them. On vcbeh's own data that produces conditional 0.8311 < marginal
    0.8366, which is impossible. Both are derived from `model.exog` here so the
    ordering holds by construction.
    """
    for corrected in (False, True):
        table = L.build_list_table(res, ll, partial_only=corrected)
        formula = L.CORRECTED_MODEL if corrected else L.UNCORRECTED_MODEL
        fit = L.fit_mixed(table, formula)
        assert fit["r2_conditional"] >= fit["r2_marginal"]
        assert 0.0 <= fit["r2_conditional"] <= 1.0


def test_fit_refuses_an_empty_table():
    import pandas as pd

    with pytest.raises(ValueError, match="no lists matched"):
        L.fit_mixed(pd.DataFrame(columns=["Subject", "ValueEst"]), L.UNCORRECTED_MODEL)


def test_misaligned_recall_counts_are_refused(res_matlab_faithful, ll):
    """The default (bug-faithful) result must not silently fit these models.

    Its act_nrec_list is wrong for rand/temp, which would corrupt every
    weighted predictor without changing a single shape or dtype.
    """
    with pytest.raises(L.InconsistentRecallCountError, match="fix_op_weights=True"):
        L.build_list_table(res_matlab_faithful, ll, partial_only=True)


def test_uncorrected_path_is_unaffected_by_the_weighting_bug(
    res_matlab_faithful, ll
):
    """The uncorrected model never touches nrec, so it must still build."""
    table = L.build_list_table(res_matlab_faithful, ll, partial_only=False)
    assert len(table) > 0
    assert table["ValuePres"].notna().all()


def test_uncorrected_model_cannot_take_an_interaction(res, ll):
    with pytest.raises(ValueError, match="nothing for it to interact with"):
        L.condition_models(res, ll, corrected=False, interaction=True)


# ---------------------------------------------------------------- centring


def test_centring_moves_the_intercept_but_not_the_slope(res, ll):
    """Same fit, different reference point."""
    plain = L.condition_models(res, ll, corrected=False).set_index(
        ["condition", "term"]
    )
    centred = L.condition_models(res, ll, corrected=False, center=True).set_index(
        ["condition", "term"]
    )

    for cond in ("rand", "temp"):
        np.testing.assert_allclose(
            plain.loc[(cond, "ValuePres"), "estimate"],
            centred.loc[(cond, "ValuePres"), "estimate"],
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            plain.loc[(cond, "ValuePres"), "R2"],
            centred.loc[(cond, "ValuePres"), "R2"],
            rtol=1e-8,
        )
        assert not np.isclose(
            plain.loc[(cond, "Intercept"), "estimate"],
            centred.loc[(cond, "Intercept"), "estimate"],
        )


def test_centring_constants_are_shared_across_conditions(res, ll):
    """Per-condition intercepts must be measured at the same point."""
    centred = L.condition_models(res, ll, corrected=True, center=True)
    for term, block in centred.groupby("term"):
        centres = block["centered_at"].dropna().unique()
        assert len(centres) <= 1, f"{term} centred at {centres}"


# ------------------------------------------------------------ sweep sizing


def test_recall_count_sizes_covers_every_count(res, ll):
    sizes = L.recall_count_sizes(res, ll)
    assert list(sizes["nrec"]) == list(range(ll + 1))
    assert sizes["n_lists"].sum() > 0


def test_viable_recall_counts_respects_the_threshold(res, ll):
    sizes = L.recall_count_sizes(res, ll).set_index("nrec")
    viable = L.viable_recall_counts(res, ll)
    assert viable, "no recall count has enough lists to fit"
    for nrec in viable:
        assert sizes.loc[nrec, "n_lists"] >= L.MIN_LISTS_PER_SWEEP_CELL
    for nrec in set(sizes.index) - set(viable):
        assert sizes.loc[nrec, "n_lists"] < L.MIN_LISTS_PER_SWEEP_CELL


def test_sweep_reports_the_cell_size_it_used(res, ll):
    table = L.recalled_vs_nonrecalled(res, ll)
    assert not table.empty
    assert (table["n_lists"] >= L.MIN_LISTS_PER_SWEEP_CELL).all()


# ------------------------------------------------------------------ labels


def test_term_label_handles_interactions_with_list_type():
    assert L.term_label("ValuePres") == "Value of ALL studied items"
    assert L.term_label("ListType[T.temp]") == "List type (Autocorr vs Random)"
    assert (
        L.term_label("ValueNotRecallW:ListType[T.temp]")
        == "NON-RECALLED  x  list type"
    )
    assert L.term_label("Unknown") == "Unknown"


def test_contrast_carries_the_list_type_interaction_terms(res, ll):
    contrast = L.condition_contrast(res, ll, corrected=True)
    terms = set(contrast["term"])
    assert "ValueRecallW:ListType[T.temp]" in terms
    assert "ValueNotRecallW:ListType[T.temp]" in terms


def test_formatters_produce_text(res, ll):
    assert "ROUND" in L.format_condition_models(
        L.condition_models(res, ll, corrected=True), "ROUND 2"
    )
    assert "list type" in L.format_contrast(
        L.condition_contrast(res, ll, corrected=True), "contrast"
    )
    assert "All lists" in L.format_all_vs_recalled(L.all_vs_recalled(res), "pooled")
