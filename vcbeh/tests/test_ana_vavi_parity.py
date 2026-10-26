"""Full-pipeline parity against the saved MATLAB results."""

from __future__ import annotations

import numpy as np
import pytest

from vcbeh.ana_vavi import _output_position_weights
from vcbeh.embam import make_clean_recalls_mask2d
from vcbeh.primitives import value_from_mask

from conftest import CONDITIONS, STALE_WEIGHTED_FIELDS, assert_matches, is_stale

# Tolerance by field family.  The first group is asserted exactly because it
# measured exactly; the rest accumulate a few float operations.
EXACT = frozenset(
    {
        "spcs",
        "pfrs",
        "crps",
        "est_list",
        "act_rec_list",
        "act_pres_list",
        "act_weightrec_list",
        "act_wtascrec_list",
        "signerr_pres_list",
        "abserr_rec_list",
        "abserr_pres_list",
        "abserr_wtrec_list",
    }
)


def _tolerance(field: str) -> tuple[float, float]:
    if field in EXACT:
        return 0.0, 0.0
    if field == "temp_clusts":
        return 1e-14, 1e-14
    return 1e-12, 1e-12


@pytest.mark.parametrize("condition", CONDITIONS)
def test_saved_fields_match(pair, condition):
    _, golden, computed = pair
    got = computed.conditions()[condition].to_dict()

    compared = 0
    for field, expected in golden[condition].items():
        if field not in got or is_stale(condition, field):
            continue
        rtol, atol = _tolerance(field)
        assert_matches(
            got[field], expected, rtol=rtol, atol=atol, label=f"{condition}.{field}"
        )
        compared += 1

    # "both" compares every saved field; rand/temp drop the four stale ones.
    floor = 20 if condition == "both" else 16
    assert compared >= floor, f"only {compared} fields compared for {condition}"


def test_unilist_matches(pair):
    _, golden, computed = pair
    np.testing.assert_array_equal(computed.unilist, golden["unilist"])


@pytest.mark.parametrize("condition", ["rand", "temp"])
@pytest.mark.parametrize("field", sorted(STALE_WEIGHTED_FIELDS))
def test_stale_weighted_fields_really_do_differ(pair, condition, field):
    """Guard the exclusion in conftest: these must still be *wrong*.

    If a future change made them agree, the exclusion would be silently
    masking a regression rather than documenting a known-stale golden value.
    """
    _, golden, computed = pair
    if field not in golden[condition]:
        pytest.skip(f"{field} not saved in this result file")
    a, b = computed.conditions()[condition].to_dict()[field], golden[condition][field]
    finite = np.isfinite(a) & np.isfinite(b)
    assert np.max(np.abs(a[finite] - b[finite])) > 1e-6


@pytest.mark.parametrize("condition", ["rand", "temp"])
def test_legacy_weighting_bug_reproduces_saved_values_exactly(pair, condition):
    """Explain the excluded fields, rather than just skipping them.

    The saved `rand`/`temp` weights were built with `max_op` read from row *i*
    of the **full** recalls matrix instead of row *i* of the condition subset.
    Reproducing that misalignment recovers the saved numbers bit-for-bit, which
    is what identifies the discrepancy as a stale golden file rather than a
    defect in this port.
    """
    trials, golden, _ = pair
    if "act_weightrec_list" not in golden[condition]:
        pytest.skip("weighted fields not saved in this result file")

    mask = trials.pres_types == (1 if condition == "rand" else 2)
    t = trials.subset(mask)
    rec_mask = make_clean_recalls_mask2d(t.recalls)

    # The defect: max_op comes from the unsubset matrix, so row i of the
    # weights belongs to a different trial than row i of the mask.
    misaligned = trials.recalls[: t.ntrials]
    wtd_dsc, wtd_asc, _ = _output_position_weights(
        misaligned, rec_mask, ascending_weight_bug=True
    )

    for field, weights in (
        ("act_weightrec_list", wtd_dsc),
        ("act_wtascrec_list", wtd_asc),
    ):
        got = value_from_mask(t.rec_values, t.subject, weights, 50)
        got[np.isinf(got)] = np.nan
        expected = golden[condition][field]
        finite = np.isfinite(got)
        assert np.array_equal(np.isnan(got), np.isnan(expected)), field
        assert np.array_equal(got[finite], expected[finite]), f"{field} not bit-exact"


def test_ascending_weight_bug_flag_changes_only_that_field(trials_45sess):
    """B1: ana_vavi.m:138 builds the "ascending" weight from the descending one."""
    from vcbeh import ana_vavi

    buggy = ana_vavi(trials_45sess, ascending_weight_bug=True).both
    fixed = ana_vavi(trials_45sess, ascending_weight_bug=False).both

    a, b = buggy.act_wtascrec_list, fixed.act_wtascrec_list
    finite = np.isfinite(a) & np.isfinite(b)
    assert np.max(np.abs(a[finite] - b[finite])) > 1.0, "flag had no effect"

    for field in ("act_rec_list", "act_weightrec_list", "act_pres_list", "spcs"):
        np.testing.assert_array_equal(
            getattr(buggy, field), getattr(fixed, field), err_msg=field
        )
