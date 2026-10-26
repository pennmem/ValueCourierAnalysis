"""Statistics and figures.

No MATLAB ground truth exists for the regressions -- both linreg scripts index
fields no saved res file contains, so they could never have run.  These tests
pin behaviour and check the settings that make statsmodels agree with fitlme.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vcbeh import ana_vavi
from vcbeh.stats import (
    error_recall_correlation,
    list_type_contrasts,
    loftus_masson,
    recalled_unweighted,
    recalled_vs_nonrecalled,
)
from vcbeh.stats.linreg import ADDITIVE_MODEL, build_table, fit_mixed


@pytest.fixture(scope="module")
def res45(trials_45sess):
    return ana_vavi(trials_45sess)


def test_loftus_masson_shape_and_positivity(res45):
    ci = loftus_masson(res45.both.spcs)
    assert ci.shape == (res45.both.spcs.shape[1],)
    assert (ci > 0).all()


def test_loftus_masson_is_narrower_than_between_subject_se(res45):
    """The whole point: removing between-subject variance tightens the interval."""
    data = res45.both.spcs
    between = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0]) * 1.96
    assert (loftus_masson(data) < between).mean() > 0.8


def test_list_type_contrasts_are_paired_across_all_subjects(res45):
    table = list_type_contrasts(res45)
    assert len(table) == 3
    assert (table["df"] == res45.both.spcs.shape[0] - 1).all()
    assert table["p"].between(0, 1).all()


def test_error_recall_correlation_is_negative(res45):
    """Subjects who estimate list value poorly also recall less."""
    stats = error_recall_correlation(res45)
    assert stats["n"] == res45.both.spcs.shape[0]
    assert stats["r"] < 0 and stats["p"] < 0.05


def test_regression_sweeps_cover_the_expected_recall_counts(res45):
    additive = recalled_vs_nonrecalled(res45, 15)
    interaction = recalled_unweighted(res45, 15)
    assert list(additive["nrec"]) == list(range(7, 14))
    assert list(interaction["nrec"]) == list(range(9, 15))
    for table in (additive, interaction):
        assert table["R2"].between(0, 1).all()
        assert (table["n_lists"] > 50).all()


def test_both_predictors_are_significant_in_the_additive_model(res45):
    table = recalled_vs_nonrecalled(res45, 15)
    assert (table["p_ValueRecall"] < 0.05).all()
    assert (table["p_ValueNotRecall"] < 0.05).all()
    assert (table["b_ValueRecall"] > 0).all()
    assert (table["b_ValueNotRecall"] > 0).all()


def test_mixed_model_avoids_the_degenerate_boundary_fit(res45):
    """statsmodels' L-BFGS can return group variance 0 and intercept 0.

    It reports convergence while doing so, so the only defence is comparing
    likelihoods across optimizers.  A zero intercept here means that guard
    stopped working.
    """
    for nrec in range(7, 14):
        table = build_table(res45, nrec, 15, weight_recall=True)
        fit = fit_mixed(table, ADDITIVE_MODEL)
        intercept = fit["estimates"][fit["names"].index("Intercept")]
        assert abs(intercept) > 1e-6, f"nrec={nrec}: degenerate intercept"


def test_mixed_model_uses_residual_degrees_of_freedom(res45):
    """fitlme defaults to DFMethod='residual'; statsmodels reports z-tests."""
    table = build_table(res45, 10, 15, weight_recall=True)
    fit = fit_mixed(table, ADDITIVE_MODEL)
    assert fit["df_resid"] == fit["n"] - len(fit["names"])


@pytest.mark.parametrize(
    "name", ["spc", "pfr", "lag_crp", "recall_probability", "temporal_clustering", "error_vs_recall"]
)
def test_figures_render(res45, name, tmp_path):
    import matplotlib.pyplot as plt

    from vcbeh.plots.res_forpres import FIGURES
    from vcbeh.plots.style import publication_style

    with publication_style():
        fig = FIGURES[name](res45)
        out = tmp_path / f"{name}.png"
        fig.savefig(out)
        plt.close(fig)
    assert out.stat().st_size > 5000


def test_lag_crp_does_not_connect_across_lag_zero(res45):
    """The MATLAB scripts index crps with negative lags, which wraps in Python.

    Drawing backward and forward lags as separate lines is what keeps the
    undefined lag of zero out of the figure.
    """
    import matplotlib.pyplot as plt

    from vcbeh.plots import plot_lag_crp

    fig = plot_lag_crp(res45, max_abs_lag=4)
    for line in fig.axes[0].get_lines():
        xs = line.get_xdata()
        if len(xs):
            assert 0 not in np.asarray(xs), "a line spans lag 0"
    plt.close(fig)


# --- per-condition models --------------------------------------------------

CONDITION_FIGURES = ["estimate_vs_actual", "partial_effects", "condition_coefficients"]


@pytest.fixture(scope="module")
def raw_res(raw_trials):
    return ana_vavi(raw_trials)


def test_uncorrected_models_use_every_list(raw_res, raw_trials):
    from vcbeh.stats import condition_models

    table = condition_models(raw_res, raw_trials.list_length, corrected=False)
    assert set(table["condition"]) == {"rand", "temp"}
    assert set(table["term"]) == {"Intercept", "ValuePres"}
    # A list with no recalls has no recall decomposition, but it still has an
    # estimate and a true value, so it belongs in the uncorrected model.
    assert table.groupby("condition")["n_lists"].first().sum() == raw_trials.ntrials


def test_corrected_models_drop_only_the_undefined_lists(raw_res, raw_trials):
    from vcbeh.stats import build_list_table, condition_models

    ll = raw_trials.list_length
    table = condition_models(raw_res, ll, corrected=True)
    assert set(table["term"]) == {"Intercept", "ValueRecallW", "ValueNotRecallW"}

    for condition in ("rand", "temp"):
        lists = build_list_table(raw_res, ll, condition=condition, partial_only=True)
        assert lists["nrec"].between(1, ll - 1).all()
        assert np.isfinite(lists["ValueRecallW"]).all()
        assert np.isfinite(lists["ValueNotRecallW"]).all()


def test_per_list_weighting_matches_the_definition(raw_res, raw_trials):
    from vcbeh.stats import build_list_table

    ll = raw_trials.list_length
    lists = build_list_table(raw_res, ll, condition="rand", partial_only=True)
    nrec = lists["nrec"].to_numpy()
    # Undo the weighting and the two terms must sum back to the list average.
    recalled = lists["ValueRecallW"].to_numpy() * ll / nrec
    not_recalled = lists["ValueNotRecallW"].to_numpy() * ll / (ll - nrec)
    combined = (nrec * recalled + (ll - nrec) * not_recalled) / ll
    np.testing.assert_allclose(combined, lists["ValuePres"].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("corrected", [False, True])
def test_models_are_well_conditioned(raw_res, raw_trials, corrected):
    from vcbeh.stats import condition_models

    table = condition_models(raw_res, raw_trials.list_length, corrected=corrected)
    assert (table["se"] > 0).all()
    assert (table["ci_low"] < table["ci_high"]).all()
    assert table["R2"].between(0, 1).all()
    assert (table["n_subjects"] == 26).all()
    # A zero intercept is the signature of the degenerate boundary fit.
    intercepts = table[table["term"] == "Intercept"]["estimate"]
    assert (intercepts.abs() > 1e-6).all()


@pytest.mark.parametrize("corrected", [False, True])
def test_contrast_includes_the_interaction_terms(raw_res, raw_trials, corrected):
    """Two separate fits cannot test a difference; these terms are the test."""
    from vcbeh.stats import condition_contrast

    table = condition_contrast(raw_res, raw_trials.list_length, corrected=corrected)
    interactions = [t for t in table["term"] if ":ListType" in t]
    assert len(interactions) == (2 if corrected else 1)
    assert table["p"].between(0, 1).all()


def test_value_slopes_are_below_one_in_both_conditions(raw_res, raw_trials):
    """Estimates compress toward the middle: the slope on true value is < 1."""
    from vcbeh.stats import condition_models

    table = condition_models(raw_res, raw_trials.list_length, corrected=False)
    slopes = table[table["term"] == "ValuePres"]
    assert (slopes["estimate"] < 1).all()
    assert (slopes["ci_high"] < 1).all()


@pytest.mark.parametrize("name", CONDITION_FIGURES)
def test_condition_figures_render(raw_res, raw_trials, name, tmp_path):
    import matplotlib.pyplot as plt

    from vcbeh.plots.res_forpres import FIGURES
    from vcbeh.plots.style import publication_style

    with publication_style():
        fig = FIGURES[name](raw_res, raw_trials.list_length)
        out = tmp_path / f"{name}.png"
        fig.savefig(out)
        plt.close(fig)
    assert out.stat().st_size > 5000


# --- the original ValueAll x ValueRecalled model ---------------------------

# Published coefficients for `ValueEst ~ 1 + ValueAll*ValueRecalled + (1|Subject)`
# fit over all 1080 lists of data_050526_45sess.mat.  This is the only external
# check that exists for this model -- the MATLAB script that produced it has
# since been edited to a different formula -- so it is pinned tightly.
SLIDE_COEFFICIENTS = {
    "Intercept": 0.559,
    "ValueAll": 0.902,
    "ValueRecalled": 0.155,
    "ValueAll:ValueRecalled": -0.003,
}
SLIDE_PVALUES = {"ValueRecalled": 0.036, "ValueAll:ValueRecalled": 0.082}


def test_reproduces_the_published_all_x_recalled_model(trials_45sess):
    """Golden test against the previously reported result."""
    from vcbeh.stats import all_vs_recalled

    # Published coefficients predate centring, so reproduce them uncentred.
    table = all_vs_recalled(ana_vavi(trials_45sess), center=False).set_index("term")
    assert int(table["n_lists"].iloc[0]) == 1080
    assert int(table["n_subjects"].iloc[0]) == 22

    for term, expected in SLIDE_COEFFICIENTS.items():
        assert table.loc[term, "estimate"] == pytest.approx(expected, abs=5e-4), term
    for term, expected in SLIDE_PVALUES.items():
        assert table.loc[term, "p"] == pytest.approx(expected, abs=5e-4), term

    # The published R-squared of 0.84 is the marginal one; the conditional
    # version this codebase reports elsewhere is 0.831.
    assert table["R2_marginal"].iloc[0] == pytest.approx(0.837, abs=5e-4)
    assert table["R2"].iloc[0] == pytest.approx(0.831, abs=5e-4)


def test_all_x_recalled_term_order_matches_the_published_table(trials_45sess):
    from vcbeh.stats import all_vs_recalled
    from vcbeh.stats.linreg import ALL_RECALLED_TERM_ORDER

    table = all_vs_recalled(ana_vavi(trials_45sess), center=False)
    assert list(table["term"]) == list(ALL_RECALLED_TERM_ORDER)


def test_all_x_recalled_splits_by_condition(raw_res):
    from vcbeh.stats import all_vs_recalled

    pooled = all_vs_recalled(raw_res)
    split = all_vs_recalled(raw_res, by_condition=True)

    assert list(pooled["condition"].unique()) == ["both"]
    assert sorted(split["condition"].unique()) == ["rand", "temp"]
    # Every list appears in exactly one condition.
    assert split.groupby("condition")["n_lists"].first().sum() == pooled["n_lists"].iloc[0]


def test_recalled_value_here_is_unweighted(raw_res, raw_trials):
    """This model uses the raw recalled mean, not the list-share-weighted one."""
    from vcbeh.stats import build_all_recalled_table, build_list_table

    unweighted = build_all_recalled_table(raw_res, condition="rand")
    weighted = build_list_table(raw_res, raw_trials.list_length,
                                condition="rand", partial_only=True)
    assert unweighted["ValueRecalled"].mean() > weighted["ValueRecallW"].mean() * 1.5
    # Unweighted recalled value is on the same 1-50 scale as the list value.
    assert 1 <= unweighted["ValueRecalled"].min()
    assert unweighted["ValueRecalled"].max() <= 50


# --- Wilcoxon contrasts ----------------------------------------------------


def test_contrasts_report_wilcoxon_as_the_headline_test(raw_res):
    from scipy.stats import wilcoxon

    from vcbeh.stats import list_type_contrasts, recall_probability

    table = list_type_contrasts(raw_res).set_index("name")
    expected = wilcoxon(recall_probability(raw_res.rand_), recall_probability(raw_res.temp))
    row = table.loc["recall probability"]
    assert row["p"] == pytest.approx(expected.pvalue)
    assert row["w"] == pytest.approx(expected.statistic)
    assert row["n"] == raw_res.both.spcs.shape[0]


def test_contrasts_keep_the_t_test_and_a_normality_check(raw_res):
    from vcbeh.stats import list_type_contrasts

    table = list_type_contrasts(raw_res)
    for column in ("t", "t_p", "df", "shapiro_p", "median_diff", "mean_diff"):
        assert column in table.columns
        assert np.isfinite(table[column]).all()


def test_raw_data_differences_are_non_normal(raw_res):
    """Why the test changed: on the raw data the t-test assumption fails."""
    from vcbeh.stats import list_type_contrasts

    table = list_type_contrasts(raw_res)
    assert (table["shapiro_p"] < 0.05).all()


# --- mean-centring ---------------------------------------------------------


@pytest.mark.parametrize("corrected", [False, True])
def test_centring_leaves_slopes_and_fit_untouched(raw_res, raw_trials, corrected):
    """Centring moves only the terms whose meaning depends on where zero is."""
    from vcbeh.stats import condition_models

    ll = raw_trials.list_length
    raw_fit = condition_models(raw_res, ll, corrected=corrected, center=False)
    centred = condition_models(raw_res, ll, corrected=corrected, center=True)

    a = raw_fit.set_index(["condition", "term"])
    b = centred.set_index(["condition", "term"])
    for key in a.index:
        if key[1] == "Intercept":
            continue
        assert a.loc[key, "estimate"] == pytest.approx(b.loc[key, "estimate"], abs=1e-6), key
        assert a.loc[key, "R2"] == pytest.approx(b.loc[key, "R2"], abs=1e-6), key


def test_centring_uses_one_constant_across_conditions(raw_res, raw_trials):
    """Both conditions must share a reference, or their intercepts differ in kind."""
    from vcbeh.stats import build_list_table, condition_models

    ll = raw_trials.list_length
    table = condition_models(raw_res, ll, corrected=False, center=True)
    constants = table.dropna(subset=["centered_at"])["centered_at"].unique()
    assert len(constants) == 1
    pooled = build_list_table(raw_res, ll)
    assert constants[0] == pytest.approx(pooled["ValuePres"].mean())


def test_centring_makes_the_list_type_effect_interpretable(raw_res, raw_trials):
    """Uncentred, the ListType term is the gap at a list value of zero.

    Observed list values run from ~8 to ~42, so that is an extrapolation and
    the coefficient even changes sign once the reference moves to a real list.
    """
    from vcbeh.stats import condition_contrast

    ll = raw_trials.list_length
    term = "ListType[T.temp]"
    uncentred = condition_contrast(raw_res, ll, corrected=False, center=False).set_index("term")
    centred = condition_contrast(raw_res, ll, corrected=False, center=True).set_index("term")

    assert uncentred.loc[term, "estimate"] < 0 < centred.loc[term, "estimate"]
    # The interaction is a slope difference, so it is invariant.
    interaction = "ValuePres:ListType[T.temp]"
    assert uncentred.loc[interaction, "estimate"] == pytest.approx(
        centred.loc[interaction, "estimate"], abs=1e-6
    )


def test_centring_changes_the_published_recalled_coefficient(trials_45sess):
    """The reported recalled-value effect does not survive centring.

    In a model with an interaction, the uncentred ValueRecalled coefficient is
    its effect where ValueAll is zero -- no list is near that.  Moving the
    reference to a typical list leaves the same fit (identical R-squared and
    interaction) but the term is no longer significant.
    """
    from vcbeh.stats import all_vs_recalled

    res = ana_vavi(trials_45sess)
    uncentred = all_vs_recalled(res, center=False).set_index("term")
    centred = all_vs_recalled(res, center=True).set_index("term")

    assert uncentred.loc["ValueRecalled", "p"] < 0.05
    assert centred.loc["ValueRecalled", "p"] > 0.05

    assert uncentred["R2"].iloc[0] == pytest.approx(centred["R2"].iloc[0], abs=1e-6)
    assert uncentred.loc["ValueAll:ValueRecalled", "estimate"] == pytest.approx(
        centred.loc["ValueAll:ValueRecalled", "estimate"], abs=1e-6
    )


def test_corrected_model_can_include_an_interaction(raw_res, raw_trials):
    from vcbeh.stats import condition_models

    ll = raw_trials.list_length
    additive = condition_models(raw_res, ll, corrected=True)
    with_interaction = condition_models(raw_res, ll, corrected=True, interaction=True)

    assert "ValueRecallW:ValueNotRecallW" not in set(additive["term"])
    assert "ValueRecallW:ValueNotRecallW" in set(with_interaction["term"])
    # Same lists, so the fit can only improve.
    for cond in ("rand", "temp"):
        a = additive[additive["condition"] == cond]
        b = with_interaction[with_interaction["condition"] == cond]
        assert a["n_lists"].iloc[0] == b["n_lists"].iloc[0]
        assert b["R2"].iloc[0] >= a["R2"].iloc[0] - 1e-9


def test_interaction_is_rejected_for_the_single_predictor_model(raw_res, raw_trials):
    from vcbeh.stats import condition_models

    with pytest.raises(ValueError, match="nothing for it to interact with"):
        condition_models(raw_res, raw_trials.list_length, corrected=False, interaction=True)
