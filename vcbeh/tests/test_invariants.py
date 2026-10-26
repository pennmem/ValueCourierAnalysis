"""Algebraic pinning for the four fields no saved result file contains.

`act_nonrec_list`, `act_nrec_list`, `act_wtdscrec_list` and `prec_byval` were
added to ana_vavi.m after the last `res` file was written, so there is no
golden value to compare against.  Each is instead tied to fields that *are*
golden, by identities that leave no room for a wrong implementation to pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcbeh.embam import make_clean_recalls_mask2d

from conftest import CONDITIONS, condition_masks


@pytest.mark.parametrize("condition", CONDITIONS)
def test_wtdscrec_is_identical_to_weightrec(pair, condition):
    """ana_vavi.m:149 repeats line 143's call verbatim, so the two must agree."""
    _, _, computed = pair
    c = computed.conditions()[condition]
    np.testing.assert_array_equal(c.act_wtdscrec_list, c.act_weightrec_list)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_nrec_counts_correct_recalls(pair, condition):
    """act_nrec_list must equal the per-list count of clean recalls."""
    trials, _, computed = pair
    t = trials.subset(condition_masks(trials)[condition])
    rec_mask = make_clean_recalls_mask2d(t.recalls)
    got = computed.conditions()[condition].act_nrec_list

    for row, subj in enumerate(np.unique(t.subject)):
        counts = rec_mask[t.subject == subj].sum(axis=1).astype(float)
        # Lists with no recalls carry the Inf sentinel, folded to NaN earlier.
        expected = np.where(counts == 0, np.nan, counts)
        actual = got[row, : counts.size]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(actual[finite], expected[finite])
        assert np.isnan(got[row, counts.size :]).all(), "trailing slots must stay NaN"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_nrec_agrees_with_golden_spc(pair, condition):
    """Tie act_nrec_list to a golden field: total recalls implied by the SPC."""
    trials, golden, computed = pair
    t = trials.subset(condition_masks(trials)[condition])
    c = computed.conditions()[condition]

    for row, subj in enumerate(np.unique(t.subject)):
        ntrials = int((t.subject == subj).sum())
        from_spc = np.nansum(golden[condition]["spcs"][row]) * ntrials
        from_nrec = np.nansum(c.act_nrec_list[row])
        assert from_nrec == pytest.approx(from_spc, abs=1e-9)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_recalled_and_nonrecalled_values_reconstruct_the_list(pair, condition):
    """The decomposition identity, against two golden fields.

    Averaging the recalled and non-recalled items back together, weighted by
    how many of each there were, must give the whole-list average.  This pins
    act_nonrec_list completely.
    """
    trials, _, computed = pair
    ll = trials.list_length
    c = computed.conditions()[condition]

    nrec = c.act_nrec_list
    usable = np.isfinite(nrec) & (nrec > 0) & (nrec < ll)
    assert usable.sum() > 100, "too few partially-recalled lists to be meaningful"

    combined = (nrec * c.act_rec_list + (ll - nrec) * c.act_nonrec_list) / ll
    np.testing.assert_allclose(
        combined[usable], c.act_pres_list[usable], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_nonrec_is_nan_exactly_when_everything_was_recalled(pair, condition):
    trials, _, computed = pair
    c = computed.conditions()[condition]
    full = np.isfinite(c.act_nrec_list) & (c.act_nrec_list == trials.list_length)
    assert np.isnan(c.act_nonrec_list[full]).all()
    assert not np.isinf(c.act_nonrec_list).any()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_prec_byval_is_a_probability_indexed_by_value(pair, condition):
    trials, _, computed = pair
    t = trials.subset(condition_masks(trials)[condition])
    prec = computed.conditions()[condition].prec_byval

    assert prec.shape[1] == int(trials.pres_values.max())
    finite = np.isfinite(prec)
    assert (prec[finite] >= 0).all() and (prec[finite] <= 1).all()

    # NaN exactly where the subject was never shown that value.
    for row, subj in enumerate(np.unique(t.subject)):
        seen = np.unique(t.pres_values[t.subject == subj])
        for v in range(1, prec.shape[1] + 1):
            if v not in seen:
                assert np.isnan(prec[row, v - 1]), f"subject {subj}, value {v}"


def test_act_pres_list_matches_the_stored_list_average(pair):
    """Independent anchor: data.act_values is unused by the pipeline.

    It is the experiment's own record of each list's mean value, stored
    rounded, so agreement confirms the list-slot layout is right.
    """
    trials, _, computed = pair
    c = computed.conditions()["both"]
    for row, subj in enumerate(np.unique(trials.subject)):
        stored = trials.act_values[trials.subject == subj]
        np.testing.assert_allclose(
            c.act_pres_list[row, : stored.size], stored, atol=1e-5
        )


# Recall-count histogram over all 1080 lists, computed in MATLAB and left as a
# comment block at linreg_estval_nonrecallrecallval.m:124-140.  It is the only
# MATLAB-produced ground truth that exists for act_nrec_list, which no saved
# res file contains: (n recalled) -> (n lists, n subjects).
MATLAB_RECALL_HISTOGRAM = {
    1: (0, 0), 2: (2, 2), 3: (6, 4), 4: (11, 7), 5: (35, 11),
    6: (50, 13), 7: (76, 17), 8: (87, 16), 9: (108, 19), 10: (114, 18),
    11: (93, 20), 12: (103, 20), 13: (97, 19), 14: (116, 15), 15: (182, 13),
}


def test_nrec_matches_matlab_histogram(trials_45sess):
    from vcbeh import ana_vavi

    nrec = ana_vavi(trials_45sess).both.act_nrec_list
    for count, (n_lists, n_subjects) in MATLAB_RECALL_HISTOGRAM.items():
        selected = nrec == count
        assert int(selected.sum()) == n_lists, f"{count} recalled: list count"
        assert int(selected.any(axis=1).sum()) == n_subjects, (
            f"{count} recalled: subject count"
        )
