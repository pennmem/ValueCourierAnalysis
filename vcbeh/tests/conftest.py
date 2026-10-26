"""Shared fixtures: the two data/result pairs that serve as golden files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vcbeh import ana_vavi, load_data, load_res

DATA_DIR = Path(__file__).resolve().parent.parent

# The raw event data lives outside the project; tests needing it skip when it
# is not present so the suite still runs from a bare checkout.
RAW_DIR = DATA_DIR.parent.parent / "data"

# Each pair is (data file, saved result file).  `ws0` is deliberately absent:
# its recall matrices are 65 columns wide while rec_values is 60, so the
# pipeline cannot run on it at all (see test_io.py).
PAIRS = [
    ("data_050526_45sess.mat", "res_050526_45sess.mat"),
    ("data_050526_xs0.mat", "res_050526_xs0.mat"),
]

CONDITIONS = ("both", "rand", "temp")

# Fields whose saved values came from an earlier, buggier ana_vavi: in the
# `rand` and `temp` conditions the output-position weights were built with
# max_op read from the *unsubset* recalls matrix, misaligning every row.
# test_ana_vavi_parity.py reproduces that legacy behaviour exactly, which is
# what licenses excluding them here.
STALE_WEIGHTED_FIELDS = frozenset(
    {
        "act_weightrec_list",
        "act_wtascrec_list",
        "abserr_wtrec_list",
        "abserr_wtrec_sub",
    }
)


def is_stale(condition: str, field: str) -> bool:
    return condition != "both" and field in STALE_WEIGHTED_FIELDS


@pytest.fixture(scope="session", params=PAIRS, ids=[p[0][:16] for p in PAIRS])
def pair(request):
    """(Trials, golden res dict, computed AnaResult) for one dataset."""
    data_file, res_file = request.param
    trials = load_data(DATA_DIR / data_file)
    golden = load_res(DATA_DIR / res_file)
    computed = ana_vavi(trials)
    return trials, golden, computed


@pytest.fixture(scope="session")
def trials_45sess():
    return load_data(DATA_DIR / "data_050526_45sess.mat")


@pytest.fixture(scope="session")
def raw_trials():
    """Raw VCBehOnly events, session 0 excluded (26 subjects, 1120 lists)."""
    if not RAW_DIR.is_dir():
        pytest.skip(f"raw data not available at {RAW_DIR}")
    from vcbeh.rawio import load_raw

    return load_raw(RAW_DIR, include_session0=False)


def condition_masks(trials):
    return {
        "both": np.ones(trials.ntrials, dtype=bool),
        "rand": trials.pres_types == 1,
        "temp": trials.pres_types == 2,
    }


def assert_matches(actual, expected, *, rtol, atol, label):
    """Compare NaN pattern, Inf pattern, and finite values as three steps.

    Checking the sentinel patterns separately matters: `value_from_mask` uses
    Inf to mean "no valid items", distinct from NaN, and an `assert_allclose`
    with `equal_nan=True` would hide a swap between the two.
    """
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), f"{label}: NaN pattern differs"
    assert np.array_equal(np.isinf(actual), np.isinf(expected)), f"{label}: Inf pattern differs"
    finite = np.isfinite(actual)
    if finite.any():
        np.testing.assert_allclose(
            actual[finite], expected[finite], rtol=rtol, atol=atol, err_msg=label
        )
