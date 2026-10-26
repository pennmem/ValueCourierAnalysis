"""Checks on the raw vcfrop event-log loader.

The strong check is :func:`test_reproduces_the_saved_mat`: the loader must
rebuild `data_083026_xs0.mat` cell for cell from the TSVs. That file was
produced by the original MATLAB pipeline from this same log, so agreement on
every analysis field is independent evidence the parsing is right.
"""

from pathlib import Path

import numpy as np
import pytest

from vcfrop.masks import make_clean_recalls_mask2d
from vcfrop.matio import load_data
from vcfrop.rawio import LIST_TYPE_CODES, _session_files, load_raw

ROOT = Path(__file__).resolve().parent.parent
RAW = Path("/Users/zrentala/Documents/Value Courier/vcfrop_modified_data")
MAT = ROOT / "data_083026_xs0.mat"

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw event log not present")

ANALYSIS_FIELDS = [
    "subject", "session", "pres_values", "pres_types", "est_values",
    "act_values", "recalls", "intrusions", "rec_values_act", "rec_values_est",
]


@pytest.fixture(scope="module")
def raw():
    return load_raw(RAW)


@pytest.fixture(scope="module")
def mat():
    return load_data(MAT)


def test_shape_matches_the_xs0_convention(raw):
    assert raw.subject.shape[0] == 470
    assert len(np.unique(raw.subject)) == 10
    assert raw.list_length == 15
    assert 0 not in np.unique(raw.session)


def test_reproduces_the_saved_mat(raw, mat):
    """Cell for cell, on every field any analysis reads."""
    width = min(raw.recalls.shape[1], mat.recalls.shape[1])
    for field in ANALYSIS_FIELDS:
        a = np.asarray(getattr(mat, field), dtype=float)
        b = np.asarray(getattr(raw, field), dtype=float)
        if a.shape[1] != b.shape[1]:
            a, b = a[:, :width], b[:, :width]
        mismatched = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        assert not mismatched.any(), (
            f"{field}: {int(mismatched.sum())} cells differ from the saved .mat"
        )


def test_trial_column_is_complete_unlike_the_mat(raw, mat):
    """The loader numbers every trial; the .mat only filled session 1.

    Nothing in the analysis reads `trial`, so this is a latent defect in the
    saved file rather than a live one -- but the loader should not copy it.
    """
    assert sorted(np.unique(raw.trial).astype(int)) == list(range(1, 11))
    later = mat.session[:, 0] > 1
    assert set(mat.trial[later, 0].astype(int)) == {0}


def test_item_numbers_are_internally_consistent(raw):
    """rec_itemnos[i] must be pres_itemnos[recalls[i] - 1] for every clean recall.

    The TSVs leave `itemno` as n/a on presentation rows, so these are derived
    from the word strings. They do not match the .mat's numbering, which came
    from a wordpool file that is not in this tree; what matters is that the
    mapping is consistent within a trial, which is all _build_nonrec_mask uses.
    """
    mask = make_clean_recalls_mask2d(raw.recalls)
    checked = 0
    for trial in range(raw.recalls.shape[0]):
        for col in np.flatnonzero(mask[trial]):
            sp = int(raw.recalls[trial, col])
            assert raw.rec_itemnos[trial, col] == raw.pres_itemnos[trial, sp - 1]
            checked += 1
    assert checked == 3092


def test_session0_is_excluded_by_default_and_includable():
    with_zero = load_raw(RAW, include_session0=True)
    assert 0 in np.unique(with_zero.session)
    # LTP663 has only session 0, so it appears only when session 0 is kept.
    assert len(np.unique(with_zero.subject)) == 11
    assert with_zero.subject.shape[0] > 470


def test_unmod_backups_are_skipped_by_default():
    modified = {(s, ses) for s, ses, _ in _session_files(RAW, modified=True)}
    backups = {(s, ses) for s, ses, _ in _session_files(RAW, modified=False)}
    assert backups, "expected some unmod- backups to exist"
    assert backups < modified, "every backed-up session should also exist modified"
    assert len(modified) == 58


def test_list_types_are_both_present(raw):
    present = set(np.unique(raw.pres_types).astype(int))
    assert present == set(LIST_TYPE_CODES.values())


def test_intrusions_are_marked_in_recalls(raw):
    """An intrusion is -1 in `recalls`; `intrusions` keeps the original code."""
    intruded = raw.intrusions != 0
    padding = raw.recalls == 0
    assert (raw.recalls[intruded & ~padding] == -1).all()
    # Prior-list intrusions carry a positive code, extra-list ones -1.
    codes = set(np.unique(raw.intrusions).astype(int))
    assert -1 in codes and any(c > 0 for c in codes)


def test_curation_only_touches_the_logged_sessions():
    """Modified vs unmod, on the four sessions that kept a backup."""
    unmod = load_raw(RAW, modified=False)
    mod = load_raw(RAW, modified=True)

    index = {
        (int(mod.subject[i, 0]), int(mod.session[i, 0]), int(mod.trial[i, 0])): i
        for i in range(mod.subject.shape[0])
    }
    width = min(unmod.recalls.shape[1], mod.recalls.shape[1])

    changed = 0
    for i in range(unmod.subject.shape[0]):
        key = (int(unmod.subject[i, 0]), int(unmod.session[i, 0]),
               int(unmod.trial[i, 0]))
        j = index[key]
        for field in ("recalls", "intrusions", "rec_values_act", "rec_values_est"):
            a = np.nan_to_num(getattr(unmod, field)[i, :width], nan=-999)
            b = np.nan_to_num(getattr(mod, field)[j, :width], nan=-999)
            if not np.array_equal(a, b):
                changed += 1
                break

    assert unmod.subject.shape[0] == 40
    assert changed == 5
