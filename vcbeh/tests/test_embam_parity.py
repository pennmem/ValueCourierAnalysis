"""The reimplemented EMBAM functions must reproduce the saved MATLAB output.

The toolbox itself is not available, so these tests are the only evidence that
the reimplementation is faithful.  spc/pfr/crp are asserted **exactly**: they
are ratios of small integers, binary-exact in float64, and they do in fact
agree to the last bit.  A refactor that loses that is worth failing on.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcbeh.embam import crp, lag_axis, lag_index, make_clean_recalls_mask2d, pfr, spc, temp_fact

from conftest import CONDITIONS, condition_masks


@pytest.fixture(scope="module")
def _cache():
    return {}


def _measures(trials, mask):
    t = trials.subset(mask)
    rec_mask = make_clean_recalls_mask2d(t.recalls)
    pres_mask = np.ones(t.pres_itemnos.shape, dtype=bool)
    ll = trials.list_length
    return {
        "spcs": spc(t.recalls, t.subject, ll, rec_mask, pres_mask),
        "pfrs": pfr(t.recalls, t.subject, ll, rec_mask, pres_mask),
        "crps": crp(t.recalls, t.subject, ll, rec_mask, rec_mask, pres_mask, pres_mask),
        "temp_clusts": temp_fact(t.recalls, t.subject, ll, rec_mask, pres_mask),
    }


@pytest.mark.parametrize("condition", CONDITIONS)
def test_spc_pfr_crp_exact(pair, condition):
    trials, golden, _ = pair
    got = _measures(trials, condition_masks(trials)[condition])
    for field in ("spcs", "pfrs", "crps"):
        a, b = got[field], golden[condition][field]
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{field}: NaN pattern differs"
        finite = np.isfinite(a)
        assert np.array_equal(a[finite], b[finite]), f"{field}: not bit-exact"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_temp_fact(pair, condition):
    """Midrank tie handling; strict > or >= would be off by ~0.024."""
    trials, golden, _ = pair
    got = _measures(trials, condition_masks(trials)[condition])["temp_clusts"]
    np.testing.assert_allclose(got, golden[condition]["temp_clusts"], rtol=1e-14, atol=1e-14)


def test_crp_lag_zero_is_nan(pair):
    trials, golden, _ = pair
    ll = trials.list_length
    crps = golden["both"]["crps"]
    assert crps.shape[1] == 2 * ll - 1
    assert np.isnan(crps[:, lag_index(0, ll)]).all()


def test_lag_axis_round_trips(pair):
    ll = pair[0].list_length
    lags = lag_axis(ll)
    assert lags[0] == -(ll - 1) and lags[-1] == ll - 1
    assert np.array_equal(lag_index(lags, ll), np.arange(2 * ll - 1))


def test_pfr_rows_sum_to_at_most_one(pair):
    """PFR is evaluated at output position 1 literally.

    A trial whose first response is an intrusion lands in the denominator but
    in no numerator, so rows fall short of 1 rather than summing to it.
    """
    _, golden, _ = pair
    sums = np.nansum(golden["both"]["pfrs"], axis=1)
    assert (sums <= 1.0 + 1e-12).all()
    assert (sums < 1.0 - 1e-9).any(), "expected at least one trial led by an intrusion"


def test_clean_recalls_mask_drops_repeats_and_intrusions():
    recalls = np.array([[3, 5, 3, -1, 0, 0], [1, 2, 3, 4, 0, 0]])
    expected = np.array(
        [[True, True, False, False, False, False], [True, True, True, True, False, False]]
    )
    assert np.array_equal(make_clean_recalls_mask2d(recalls), expected)


def test_group_apply_orders_rows_by_sorted_subject():
    from vcbeh.embam import group_apply

    subjects = np.array([30, 10, 20, 10])
    values = np.array([[3.0], [1.0], [2.0], [1.5]])
    out = group_apply(lambda v: np.array([v.sum()]), subjects, (values,))
    np.testing.assert_allclose(out.ravel(), [2.5, 2.0, 3.0])  # subjects 10, 20, 30
