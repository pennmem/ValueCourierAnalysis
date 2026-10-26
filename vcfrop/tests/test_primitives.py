"""Hand-computed checks of the reimplemented MATLAB primitives.

Fixture: list_length 4, 5 output slots, 2 subjects x 2 trials.

    subject 1, trial 1: [ 2,  3,  2,  1,  0]   op3 repeats SP2
    subject 1, trial 2: [-1,  4,  1,  0,  0]   op1 is an intrusion
    subject 2, trial 1: [ 1,  2,  3,  4,  0]   perfect forward run
    subject 2, trial 2: [-1, -1,  0,  0,  0]   no correct recalls at all

Every expected value below is worked out by hand in the test's docstring.
"""

import numpy as np
import pytest

from vcfrop.masks import apply_by_index, make_clean_recalls_mask2d
from vcfrop.recall import crp, pfr, spc, temp_fact
from vcfrop.stats import loftus_masson
from vcfrop.value import mean_abserr_sess, value_from_mask

LL = 4

RECALLS = np.array(
    [
        [2, 3, 2, 1, 0],
        [-1, 4, 1, 0, 0],
        [1, 2, 3, 4, 0],
        [-1, -1, 0, 0, 0],
    ],
    dtype=float,
)
SUBJECTS = np.array([[1], [1], [2], [2]], dtype=float)
PRES_MASK = np.ones((4, LL), dtype=bool)


@pytest.fixture
def rec_mask():
    return make_clean_recalls_mask2d(RECALLS)


# ------------------------------------------------------------------- masks


def test_clean_mask_drops_intrusions_and_repeats(rec_mask):
    """Positive and not already seen in the row.

    t1: SP2, SP3 kept; the second SP2 at op3 is a repeat; SP1 kept; 0 is empty.
    t2: op1 is -1; SP4 and SP1 kept.
    t4: nothing is positive.
    """
    expected = np.array(
        [
            [True, True, False, True, False],
            [False, True, True, False, False],
            [True, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(rec_mask, expected)


def test_apply_by_index_splits_rows_and_stacks():
    rows = apply_by_index(
        lambda m: np.array([m.shape[0], m.sum()]), SUBJECTS, 1, (RECALLS,)
    )
    # Subject 1 rows sum to 2+3+2+1 + (-1)+4+1 = 12; subject 2 to 10 + (-2) = 8.
    np.testing.assert_array_equal(rows, np.array([[2, 12], [2, 8]]))


def test_apply_by_index_flattens_scalar_results():
    out = apply_by_index(lambda m: m.shape[0], SUBJECTS, 1, (RECALLS,))
    assert out.shape == (2,)


# --------------------------------------------------------------------- spc


def test_spc(rec_mask):
    """Subject 1: SP1 in both trials -> 1.0; SP2, SP3 in t1 only; SP4 in t2 only.

    Subject 2: every SP appears in t1 only, of 2 trials -> 0.5 each.
    """
    got = spc(RECALLS, SUBJECTS, LL, rec_mask, PRES_MASK)
    np.testing.assert_allclose(
        got, [[1.0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    )


# --------------------------------------------------------------------- pfr


def test_pfr_uses_output_position_one_only(rec_mask):
    """A list whose first response is an intrusion credits no serial position.

    Subject 1: t1 starts at SP2; t2 starts with -1, so it contributes to the
    denominator only -> [0, 0.5, 0, 0].
    Subject 2: t1 starts at SP1; t4 starts with -1 -> [0.5, 0, 0, 0].
    """
    got = pfr(RECALLS, SUBJECTS, LL, rec_mask, PRES_MASK)
    np.testing.assert_allclose(got, [[0.0, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])


# --------------------------------------------------------------------- crp


def test_crp(rec_mask):
    """Columns are lag -3..+3; centre (lag 0) is NaN.

    Subject 1
      t1 op1->op2: from SP2, recalled={2}, available {1,3,4} -> lags -1,+1,+2;
                   transition to SP3 is lag +1.
         op2->op3 is masked out (repeat), op4->op5 has no target.
      t2 op2->op3: from SP4, recalled={4}, available {1,2,3} -> lags -3,-2,-1;
                   transition to SP1 is lag -3.
      possible: -3:1 -2:1 -1:2 +1:1 +2:1     actual: -3:1 +1:1
      -> [1.0, 0.0, 0.0, nan, 1.0, 0.0, nan]  (+3 never possible)

    Subject 2
      t1 gives three forward transitions, each lag +1, with available sets
      {2,3,4}, {3,4}, {4} -> possible +1:3 +2:2 +3:1, actual +1:3.
      t4 contributes nothing.
      -> [nan, nan, nan, nan, 1.0, 0.0, 0.0]
    """
    got = crp(RECALLS, SUBJECTS, LL, rec_mask, rec_mask, PRES_MASK, PRES_MASK)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, np.nan, 1.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(got, expected)


def test_crp_lag_zero_is_nan(rec_mask):
    got = crp(RECALLS, SUBJECTS, LL, rec_mask, rec_mask, PRES_MASK, PRES_MASK)
    assert np.isnan(got[:, LL - 1]).all()


# --------------------------------------------------------------- temp_fact


def test_temp_fact(rec_mask):
    """Percentile rank of the actual lag among lags to available items.

    Subject 1
      t1 SP2->SP3: available {1,3,4}, actual |lag|=1; others SP1 (1), SP4 (2).
                   greater=1, tied=1 -> (1 + 0.5)/2 = 0.75
      t2 SP4->SP1: available {1,2,3}, actual |lag|=3; others SP2 (2), SP3 (1).
                   greater=0, tied=0 -> 0.0
      pooled mean = 0.375

    Subject 2
      SP1->SP2: available {2,3,4}, actual 1; others 2, 3 -> 2/2 = 1.0
      SP2->SP3: available {3,4},   actual 1; others 2    -> 1/1 = 1.0
      SP3->SP4: only one item left, no comparison -> skipped
      pooled mean = 1.0
    """
    got = temp_fact(RECALLS, SUBJECTS, LL, rec_mask, PRES_MASK)
    np.testing.assert_allclose(got, [0.375, 1.0])


def test_temp_fact_uses_adjacent_output_positions():
    """A masked-out response breaks the pair instead of being skipped over.

    [1, 2, 1, 4] with the op3 repeat masked out yields only the 1->2
    transition; if the sequence were compressed it would also count 2->4.
    """
    recalls = np.array([[1.0, 2.0, 1.0, 4.0]])
    subjects = np.array([[1.0]])
    mask = make_clean_recalls_mask2d(recalls)
    pres = np.ones((1, LL), dtype=bool)

    # 1->2: available {2,3,4}, actual 1; others 2, 3 -> greater 2 of 2 = 1.0
    np.testing.assert_allclose(temp_fact(recalls, subjects, LL, mask, pres), [1.0])


# ------------------------------------------------------------------- value


def test_value_from_mask_plain_mean():
    values = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=float)
    got = value_from_mask(values, np.array([[1.0], [1.0]]), mask, 4)
    np.testing.assert_allclose(got[0, :2], [15.0, 2.0])
    assert np.isnan(got[0, 2:]).all()  # padded out to maxnlist


def test_value_from_mask_weighted():
    """Non-binary mask makes this a weighted mean: (10*3 + 20*1)/4 = 12.5."""
    values = np.array([[10.0, 20.0]])
    mask = np.array([[3.0, 1.0]])
    got = value_from_mask(values, np.array([[1.0]]), mask, 1)
    np.testing.assert_allclose(got, [[12.5]])


def test_value_from_mask_empty_mask_is_inf():
    """The no-recall sentinel, distinct from the NaN used for a missing list."""
    got = value_from_mask(
        np.array([[5.0, 6.0]]), np.array([[1.0]]), np.zeros((1, 2)), 2
    )
    assert np.isposinf(got[0, 0])
    assert np.isnan(got[0, 1])


def test_value_from_mask_nan_inside_mask_counts_as_zero():
    """NaN contributes 0 to the numerator but still counts in the denominator.

    (27 + 0 + 35 + 30) / 4 = 23.0 -- the case that pins this behavior in the
    real data (trial 8 of data_083026_xs0).
    """
    values = np.array([[27.0, np.nan, 35.0, 30.0]])
    mask = np.ones((1, 4))
    got = value_from_mask(values, np.array([[1.0]]), mask, 1)
    np.testing.assert_allclose(got, [[23.0]])


def test_value_from_mask_rejects_more_lists_than_maxnlist():
    with pytest.raises(ValueError, match="exceeding maxnlist"):
        value_from_mask(
            np.ones((3, 2)), np.array([[1.0], [1.0], [1.0]]), np.ones((3, 2)), 2
        )


def test_mean_abserr_sess_averages_within_session():
    abserr = np.array([[1.0, 3.0, 10.0, 20.0]])
    sessions = np.array([[1.0, 1.0, 2.0, 2.0]])
    got = mean_abserr_sess(abserr, sessions, [1], 2)
    np.testing.assert_allclose(got, [[2.0, 15.0]])


def test_mean_abserr_sess_propagates_nan():
    """A single undefined list makes the whole session NaN, as in MATLAB."""
    abserr = np.array([[1.0, np.nan, 10.0, 20.0]])
    sessions = np.array([[1.0, 1.0, 2.0, 2.0]])
    got = mean_abserr_sess(abserr, sessions, [1], 2)
    assert np.isnan(got[0, 0])
    np.testing.assert_allclose(got[0, 1], 15.0)


def test_mean_abserr_sess_columns_are_positional_per_subject():
    """Column j is a subject's j-th session, not "session j".

    Subject 2 never saw session 1, so its session-2 data lands in column 0,
    alongside subject 1's session-1 data. This is what the MATLAB helper does.
    """
    abserr = np.array([[1.0, 5.0], [7.0, 9.0]])
    sessions = np.array([[1.0, 2.0], [2.0, 3.0]])
    got = mean_abserr_sess(abserr, sessions, [1, 2], 3)
    np.testing.assert_allclose(got[0], [1.0, 5.0, np.nan])
    np.testing.assert_allclose(got[1], [7.0, 9.0, np.nan])


def test_mean_abserr_sess_rejects_more_sessions_than_maxnsess():
    with pytest.raises(ValueError, match="exceeding maxnsess"):
        mean_abserr_sess(
            np.ones((1, 3)), np.array([[1.0, 2.0, 3.0]]), [1], 2
        )


# ----------------------------------------------------------- loftus_masson


def test_loftus_masson_shape_and_positivity():
    rng = np.random.default_rng(0)
    dat = rng.normal(size=(12, 5)) + np.arange(12)[:, None]
    ci = loftus_masson(dat)
    assert ci.shape == (5,)
    assert np.all(ci > 0)


def test_loftus_masson_ignores_per_subject_level():
    """The defining property: within-subject CIs drop each subject's offset.

    Adding a constant to one subject's whole row shifts the between-subject
    variance but not the subject-by-condition interaction, so the intervals
    must not move.
    """
    rng = np.random.default_rng(1)
    dat = rng.normal(size=(9, 4))
    shifted = dat.copy()
    shifted[3] += 25.0

    np.testing.assert_allclose(loftus_masson(dat), loftus_masson(shifted), rtol=1e-9)


def test_loftus_masson_grows_with_interaction_noise():
    rng = np.random.default_rng(2)
    base = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (10, 1))
    quiet = base + rng.normal(scale=0.05, size=base.shape)
    noisy = base + rng.normal(scale=1.0, size=base.shape)
    assert np.nanmean(loftus_masson(noisy)) > np.nanmean(loftus_masson(quiet))


# ------------------------------------------- pres_mask as the spc denominator


def test_spc_denominator_is_presented_lists_not_trial_count(rec_mask):
    """A value-filtered pres_mask changes the denominator, not just the numerator.

    Restrict the mask to serial position 1 only. Subject 1 presented SP1 in
    both trials and recalled it in both (t1 at op4, t2 at op3), so P = 2/2 = 1.
    Every other position has a zero denominator and must come back NaN, which
    is what lets prec_byval average over positions a subject never saw.
    """
    pres = np.zeros((4, LL), dtype=bool)
    pres[:, 0] = True

    got = spc(RECALLS, SUBJECTS, LL, rec_mask, pres)
    np.testing.assert_allclose(got[:, 0], [1.0, 0.5])
    assert np.isnan(got[:, 1:]).all()


def test_spc_denominator_counts_only_that_subjects_lists(rec_mask):
    """pres_mask must be sliced per subject before the denominator is taken.

    Summing it over all four trials instead of the subject's two would halve
    every value.
    """
    pres = np.ones((4, LL), dtype=bool)
    got = spc(RECALLS, SUBJECTS, LL, rec_mask, pres)
    # Subject 1 recalled SP1 in both of ITS two lists -> 1.0, not 2/4.
    np.testing.assert_allclose(got[0, 0], 1.0)
