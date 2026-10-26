"""Rebuilding the trial matrices from the raw event log.

These tests are the justification for trusting `load_raw` at all: the raw
events must reproduce the MATLAB-built `.mat` files exactly.  Two independent
golden files are used, covering different subject and session sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcbeh import ana_vavi, load_data, load_res
from vcbeh.rawio import load_raw

from conftest import DATA_DIR, RAW_DIR, STALE_WEIGHTED_FIELDS, is_stale

MATRIX_FIELDS = [
    "pres_itemnos", "pres_values", "pres_types", "recalls",
    "rec_itemnos", "intrusions", "rec_values", "est_values", "act_values",
]

pytestmark = pytest.mark.skipif(
    not RAW_DIR.is_dir(), reason=f"raw data not available at {RAW_DIR}"
)


def _aligned_subset(raw, mat):
    """Raw rows restricted to the subject/session set the .mat covers."""
    keep = np.zeros(raw.ntrials, dtype=bool)
    for subject in np.unique(mat.subject):
        for session in np.unique(mat.session[mat.subject == subject]):
            keep |= (raw.subject == subject) & (raw.session == session)
    return raw.subset(keep)


@pytest.mark.parametrize(
    "matfile", ["data_050526_45sess.mat", "data_050526_xs0.mat"]
)
def test_raw_reproduces_the_mat_exactly(raw_trials, matfile):
    mat = load_data(DATA_DIR / matfile)
    sub = _aligned_subset(raw_trials, mat)

    assert sub.ntrials == mat.ntrials
    np.testing.assert_array_equal(sub.subject, mat.subject)
    np.testing.assert_array_equal(sub.session, mat.session)
    assert sub.list_length == mat.list_length

    ncols = min(sub.n_recall_cols, mat.n_recall_cols)
    for field in MATRIX_FIELDS:
        got, expected = getattr(sub, field), getattr(mat, field)
        if got.ndim == 2 and got.shape[1] != expected.shape[1]:
            # The .mat was built at a fixed width; the columns beyond the
            # longest trial must be empty on both sides before we compare.
            _assert_padding_is_empty(got[:, ncols:], f"{field} (raw)")
            _assert_padding_is_empty(expected[:, ncols:], f"{field} (mat)")
            got, expected = got[:, :ncols], expected[:, :ncols]
        if field in ("rec_values", "act_values"):
            np.testing.assert_array_equal(np.isnan(got), np.isnan(expected), err_msg=field)
            finite = ~np.isnan(got)
            np.testing.assert_allclose(got[finite], expected[finite], atol=1e-5, err_msg=field)
        else:
            np.testing.assert_array_equal(got, expected, err_msg=field)


def _assert_padding_is_empty(block, label):
    if block.size == 0:
        return
    empty = np.all(np.isnan(block)) if block.dtype.kind == "f" else not block.any()
    assert empty, f"{label}: padding columns are not empty"


def test_raw_through_ana_vavi_matches_saved_results(raw_trials):
    """End to end: raw events -> Trials -> ana_vavi -> the saved res file."""
    result = ana_vavi(raw_trials)
    golden = load_res(DATA_DIR / "res_050526_xs0.mat")

    compared = 0
    for condition, computed in result.conditions().items():
        got = computed.to_dict()
        for field, expected in golden[condition].items():
            if field not in got or is_stale(condition, field):
                continue
            actual = got[field]
            assert actual.shape == expected.shape, f"{condition}.{field}"
            np.testing.assert_array_equal(
                np.isnan(actual), np.isnan(expected), err_msg=f"{condition}.{field}"
            )
            np.testing.assert_array_equal(
                np.isinf(actual), np.isinf(expected), err_msg=f"{condition}.{field}"
            )
            finite = np.isfinite(actual)
            np.testing.assert_allclose(
                actual[finite], expected[finite], rtol=1e-12, atol=1e-12,
                err_msg=f"{condition}.{field}",
            )
            compared += 1
    assert compared >= 50


def test_excluding_session0_also_drops_session0_only_subjects(raw_trials):
    """Eight subjects did nothing but session 0, so they go with it."""
    everything = load_raw(RAW_DIR, include_session0=True)

    assert np.unique(everything.subject).size == 34
    assert everything.ntrials == 1285
    assert np.unique(raw_trials.subject).size == 26
    assert raw_trials.ntrials == 1120
    assert (raw_trials.session > 0).all()

    dropped = set(np.unique(everything.subject)) - set(np.unique(raw_trials.subject))
    assert dropped == {618, 622, 624, 626, 629, 633, 647, 654}


def test_recall_matrices_share_one_width(raw_trials):
    """The defect that makes data_050526_ws0.mat unusable.

    There, `recalls` is 65 columns wide while `rec_values` is 60, so a mask
    built from one cannot index the other.
    """
    for trials in (raw_trials, load_raw(RAW_DIR, include_session0=True)):
        widths = {
            getattr(trials, name).shape[1]
            for name in ("recalls", "rec_itemnos", "intrusions", "rec_values")
        }
        assert len(widths) == 1, widths
        from vcbeh.io import structural_problems

        assert structural_problems(trials) == []


def test_experiment_filter_is_applied(raw_trials):
    """Every row here is VCBehOnly, so the filter should change nothing."""
    unfiltered = load_raw(RAW_DIR, include_session0=False, experiment=None)
    assert unfiltered.ntrials == raw_trials.ntrials
    with pytest.raises(ValueError, match="no rows left for experiment"):
        load_raw(RAW_DIR, include_session0=False, experiment="NoSuchExperiment")


def test_csv_and_directory_inputs_agree(raw_trials):
    exports = sorted(RAW_DIR.glob("*.csv"))
    assert exports, "expected a concatenated csv export"
    from_csv = load_raw(exports[0], include_session0=False)
    assert from_csv.ntrials == raw_trials.ntrials
    np.testing.assert_array_equal(from_csv.recalls, raw_trials.recalls)
    np.testing.assert_array_equal(from_csv.pres_itemnos, raw_trials.pres_itemnos)


def test_list_types_are_recovered_from_storepointtype(raw_trials):
    assert set(np.unique(raw_trials.pres_types)) == {1, 2}
    counts = np.bincount(raw_trials.pres_types)
    assert abs(int(counts[1]) - int(counts[2])) < 0.1 * raw_trials.ntrials
