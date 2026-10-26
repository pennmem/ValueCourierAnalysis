"""Loading, saving, and the datasets the pipeline cannot process."""

from __future__ import annotations

import numpy as np
import pytest

from vcbeh import ana_vavi, load_data, load_res, save_res
from vcbeh.io import structural_problems

from conftest import DATA_DIR


def test_integer_fields_are_widened_at_load(trials_45sess):
    """The uint8 session field is what makes MATLAB's `bysess` path fail.

    There, 1000*uint8(session) saturates at 255 for every session, so every
    session gets the same offset and the split silently never happens.
    """
    assert trials_45sess.session.dtype == np.int64
    assert trials_45sess.subject.dtype == np.int64
    offsets = np.unique(1000 * trials_45sess.session)
    assert offsets.size > 1, "sessions must produce distinct offsets"


def test_bysess_actually_separates_sessions(trials_45sess):
    plain = ana_vavi(trials_45sess)
    split = ana_vavi(trials_45sess, bysess=True)
    assert split.unilist.size > plain.unilist.size


def test_ws0_is_reported_as_structurally_broken():
    """recalls/rec_itemnos are 65 columns wide but rec_values is 60."""
    trials = load_data(DATA_DIR / "data_050526_ws0.mat")
    problems = structural_problems(trials)
    assert any("rec_values" in p for p in problems), problems
    with pytest.raises((IndexError, ValueError)):
        ana_vavi(trials)


def test_save_res_round_trips(trials_45sess, tmp_path):
    result = ana_vavi(trials_45sess)
    out = tmp_path / "res_roundtrip.mat"
    save_res(result, out)

    reloaded = load_res(out)
    np.testing.assert_array_equal(reloaded["unilist"], result.unilist)
    for name in ("both", "rand", "temp"):
        original = result.conditions()[name].to_dict()
        assert set(reloaded[name]) == set(original)
        for field, value in original.items():
            got = reloaded[field] if field in reloaded else reloaded[name][field]
            assert got.shape == value.shape, field
            np.testing.assert_array_equal(np.isnan(got), np.isnan(value), err_msg=field)
            finite = np.isfinite(value)
            np.testing.assert_array_equal(got[finite], value[finite], err_msg=field)


def test_saved_res_fills_the_previously_missing_fields(trials_45sess, tmp_path):
    """The point of the export: unblock the regressions.

    Both linreg scripts index act_nrec_list and act_nonrec_list, which no
    saved res file contains -- so they could not run at all before this.
    """
    out = tmp_path / "res_new.mat"
    save_res(ana_vavi(trials_45sess), out)
    fresh = load_res(out)
    stale = load_res(DATA_DIR / "res_050526_45sess.mat")
    for field in ("act_nrec_list", "act_nonrec_list", "act_wtdscrec_list", "prec_byval"):
        assert field not in stale["both"]
        assert field in fresh["both"]
        assert np.isfinite(fresh["both"][field]).any()
