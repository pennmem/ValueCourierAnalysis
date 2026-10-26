"""Hand-computed checks of the z-scored value-estimation curves.

Base fixture: list length 4, one subject, values ``[10, 20, 30, 40]`` in every
list, so every list has the same centre and scale::

    mu = 25
    sd = std([10, 20, 30, 40], ddof=1) = sqrt(500/3) ~ 12.90994

Each list is recalled perfectly in forward order, and the subject's estimate is
the true value plus a constant offset for that list. The signed z-error of every
item in a list is therefore exactly ``offset / sd``, which makes the
aggregation arithmetic checkable by hand.

The sessions are deliberately **unbalanced** -- two lists in session 1, one in
session 2 -- because that is the only way to tell the lists->sessions->subjects
hierarchy apart from a pooled mean.
"""

import numpy as np
import pytest

from vcfrop.matio import Data
from vcfrop.valuecurves import (
    MEASURES,
    error_by_lag,
    error_by_output_position,
    error_by_serial_position,
    item_z_scores,
    observation_counts,
    value_curves,
)

LL = 4
VALUES = np.array([10.0, 20.0, 30.0, 40.0])
MU = 25.0
SD = float(np.std(VALUES, ddof=1))  # sqrt(500/3)

#: offset added to the true value to make each list's estimate
OFFSETS = (2.0, 4.0, 10.0)
SESSIONS = (1.0, 1.0, 2.0)  # unbalanced on purpose


def _fixture(offsets=OFFSETS, sessions=SESSIONS, recalls=None):
    n = len(offsets)
    if recalls is None:
        recalls = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1))
    recalls = np.asarray(recalls, dtype=float)

    act = np.full(recalls.shape, np.nan)
    est = np.full(recalls.shape, np.nan)
    for t in range(n):
        for c in range(recalls.shape[1]):
            sp = int(recalls[t, c])
            if 1 <= sp <= LL:
                act[t, c] = VALUES[sp - 1]
                est[t, c] = VALUES[sp - 1] + offsets[t]

    return Data(
        {
            "list_length": LL,
            "recalls": recalls,
            "pres_values": np.tile(VALUES, (n, 1)),
            "rec_values_act": act,
            "rec_values_est": est,
            "subject": np.ones((n, 1)),
            "session": np.array(sessions, dtype=float).reshape(-1, 1),
        }
    )


# ------------------------------------------------------------- item z scores


def test_z_act_is_the_list_z_score_of_the_true_value():
    """(10-25)/sd, (20-25)/sd, (30-25)/sd, (40-25)/sd at output positions 1-4."""
    z = item_z_scores(_fixture())
    np.testing.assert_allclose(z["z_act"][0], (VALUES - MU) / SD)


def test_signed_error_is_the_offset_in_z_units():
    z = item_z_scores(_fixture())
    for trial, offset in enumerate(OFFSETS):
        np.testing.assert_allclose(z["signed"][trial], offset / SD)
        np.testing.assert_allclose(z["absolute"][trial], abs(offset) / SD)


def test_items_without_an_estimate_are_excluded():
    data = _fixture()
    data["rec_values_est"][0, 2] = np.nan  # subject skipped this one
    z = item_z_scores(data)
    assert not z["valid"][0, 2]
    assert np.isnan(z["signed"][0, 2])


def test_zero_variance_list_is_refused():
    data = _fixture()
    data["pres_values"][1] = 7.0
    with pytest.raises(ValueError, match="zero variance"):
        item_z_scores(data)


# -------------------------------------------------------------- aggregation


def test_hierarchy_is_lists_then_sessions_then_subjects():
    """The whole point of the ordering, on an unbalanced fixture.

    Session 1 holds offsets 2 and 4 -> mean 3.
    Session 2 holds offset 10       -> mean 10.
    Subject  = (3 + 10) / 2         = 6.5     -> 6.5 / sd

    Pooling every list equally would instead give (2+4+10)/3 = 5.333, so the
    two differ and this test distinguishes them.
    """
    got = error_by_serial_position(_fixture(), measure="signed")

    expected = 6.5 / SD
    pooled = (2.0 + 4.0 + 10.0) / 3.0 / SD

    assert got.shape == (1, LL)
    np.testing.assert_allclose(got, np.full((1, LL), expected))
    assert not np.isclose(expected, pooled)


def test_balanced_sessions_agree_with_pooling():
    """The hierarchy only diverges when the sessions are unbalanced."""
    data = _fixture(offsets=(2.0, 4.0, 10.0, 20.0), sessions=(1.0, 1.0, 2.0, 2.0))
    got = error_by_serial_position(data, measure="signed")
    np.testing.assert_allclose(got, np.full((1, LL), 9.0 / SD))


# ------------------------------------------------------------------ curves


def test_serial_position_bins_by_serial_position():
    """Recall the list backwards; the error must follow the item, not the slot."""
    data = _fixture(offsets=(2.0,), sessions=(1.0,), recalls=[[4.0, 3.0, 2.0, 1.0]])
    z_act = error_by_serial_position(data, measure="z_act")
    np.testing.assert_allclose(z_act[0], (VALUES - MU) / SD)


def test_output_position_bins_by_output_position():
    data = _fixture(offsets=(2.0,), sessions=(1.0,), recalls=[[4.0, 3.0, 2.0, 1.0]])
    z_act = error_by_output_position(data, measure="z_act", n_positions=LL)
    np.testing.assert_allclose(z_act[0], (VALUES[::-1] - MU) / SD)


def test_output_position_column_count_is_shared_across_conditions():
    data = _fixture()
    everything = error_by_output_position(data, None, "signed")
    subset = error_by_output_position(data, np.array([True, False, True]), "signed")
    assert everything.shape[1] == subset.shape[1]


def test_lag_zero_is_nan():
    got = error_by_lag(_fixture(), measure="signed")
    assert got.shape == (1, 2 * LL - 1)
    assert np.isnan(got[:, LL - 1]).all()


def test_lag_uses_adjacent_output_positions():
    """A masked-out repeat breaks the pair instead of being skipped over.

    Recalling [1, 2, 1, 4]: the op3 repeat of SP1 is dropped by the clean mask,
    so only the 1->2 transition (lag +1) is scored. If the sequence were
    compressed first, 2->4 (lag +2) would be scored too.
    """
    data = _fixture(offsets=(2.0,), sessions=(1.0,), recalls=[[1.0, 2.0, 1.0, 4.0]])
    got = error_by_lag(data, measure="signed")

    assert not np.isnan(got[0, 1 + LL - 1])      # lag +1 scored
    assert np.isnan(got[0, 2 + LL - 1])          # lag +2 not scored


def test_lag_scores_the_item_transitioned_to():
    """The value scored is the destination's, and same-lag pairs average.

    Recalling [1, 3, 2, 4] gives three transitions:

        1->3   lag +2, lands on SP3 (value 30)
        3->2   lag -1, lands on SP2 (value 20)
        2->4   lag +2, lands on SP4 (value 40)

    Two of them share lag +2, so that bin holds the mean of the two
    destinations before the trial enters the hierarchy.
    """
    data = _fixture(offsets=(0.0,), sessions=(1.0,), recalls=[[1.0, 3.0, 2.0, 4.0]])
    got = error_by_lag(data, measure="z_act")

    expected_plus_2 = ((30.0 - MU) / SD + (40.0 - MU) / SD) / 2
    np.testing.assert_allclose(got[0, 2 + LL - 1], expected_plus_2)
    np.testing.assert_allclose(got[0, -1 + LL - 1], (20.0 - MU) / SD)


# -------------------------------------------------------------- bookkeeping


def test_observation_counts_match_the_scored_items():
    data = _fixture()
    np.testing.assert_array_equal(
        observation_counts(data, None, "serial_position"), [3, 3, 3, 3]
    )
    # Three lists x three forward transitions, all at lag +1.
    lags = observation_counts(data, None, "lag")
    assert lags[1 + LL - 1] == 9
    assert lags.sum() == 9


def test_bad_measure_is_refused():
    with pytest.raises(ValueError, match="measure must be one of"):
        error_by_serial_position(_fixture(), measure="nonsense")


def test_bad_binning_is_refused():
    with pytest.raises(ValueError, match="binning must be"):
        observation_counts(_fixture(), None, "nonsense")


def test_value_curves_returns_every_binning_and_measure():
    curves = value_curves(_fixture())
    assert set(curves) == {"serial_position", "output_position", "lag"}
    for binning in curves.values():
        assert set(binning) == set(MEASURES)
