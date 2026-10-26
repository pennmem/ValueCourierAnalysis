"""Port of ana_vavi_e2frop.m.

Runs the full analysis three times -- all lists, random-value lists, and
temporally autocorrelated lists -- and returns the same field names the MATLAB
struct carried, so results can be compared field-for-field against a saved
res_*.mat.

Two defects in the MATLAB source are reproduced by default and corrected under
``fix_op_weights=True``; see :func:`_output_position_weights`.
"""

import warnings

import numpy as np

from .masks import make_clean_recalls_mask2d, unique_subjects
from .matio import Data, load_data
from .recall import crp, pfr, spc, temp_fact
from .value import mean_abserr_sess, value_from_mask


#: Midpoint of the 1-50 value range: the error a subject would make by
#: ignoring the list entirely and guessing the middle every time. It is the
#: reference any real estimate has to beat.
VALUE_BASELINE = 25


class AnalysisConsistencyError(RuntimeError):
    """A cross-measure alignment check failed.

    These stand in for the bare ``keyboard`` calls at ana_vavi_e2frop.m:142,
    171 and 175, which dropped MATLAB into the debugger.
    """


def _nan_reduce(func, arr, axis):
    """NaN-omitting reduction that stays quiet on all-NaN slices.

    MATLAB's ``mean(...,'omitnan')`` and ``min/max`` return NaN for an all-NaN
    slice without complaint; numpy warns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(arr, axis=axis)


def _output_position_weights(recalls_all, recalls_sel, rec_mask, fix):
    """Build the output-position weighted masks (ana_vavi_e2frop.m:106-120).

    Returns ``(wtd_dsc, wtd_asc, nrec)``.

    ``wtd_dsc`` weights each correct recall by how many output positions remain
    (first recall gets the largest weight); ``wtd_asc`` is meant to be the
    reverse. ``nrec`` carries the correct-recall count for the list.

    Two bugs in the MATLAB source are reproduced when ``fix`` is False:

    1. The loop runs over the *filtered* row count but indexes
       ``data.recalls(tri,:)`` on the *unfiltered* array, so for the rand/temp
       conditions ``max_op`` is read from an unrelated trial. Inert for the
       'both' condition, where the two line up.
    2. ``wtd_rec_mask_asc`` is built from ``wtd_rec_mask_dsc`` one line after
       that array was overwritten, so the "ascending" weights are really
       ``(1:max_op) .* (max_op:-1:1)`` -- a symmetric bow, not a ramp.

    With ``fix`` True, ``max_op`` comes from the filtered rows and the
    ascending weights are built from the unweighted mask.
    """
    base = rec_mask.astype(np.float64)
    wtd_dsc = base.copy()
    wtd_asc = base.copy()
    nrec = base.copy()

    source = recalls_sel if fix else recalls_all

    for tri in range(wtd_dsc.shape[0]):
        if tri >= source.shape[0]:
            # Cannot happen once fixed; under the bug the filtered row count is
            # always <= the unfiltered one, so this is unreachable either way.
            break
        positive = np.flatnonzero(source[tri] > 0)
        if positive.size == 0:
            # MATLAB's find(...,'last') returns empty here and every indexed
            # assignment below becomes a no-op.
            continue
        max_op = int(positive[-1]) + 1  # count of output positions, 1-based

        ramp_down = np.arange(max_op, 0, -1, dtype=np.float64)
        ramp_up = np.arange(1, max_op + 1, dtype=np.float64)

        if fix:
            wtd_asc[tri, :max_op] = ramp_up * base[tri, :max_op]
            wtd_dsc[tri, :max_op] = ramp_down * base[tri, :max_op]
        else:
            wtd_dsc[tri, :max_op] = ramp_down * wtd_dsc[tri, :max_op]
            # Bug 2: reads the line above's output, not the plain mask.
            wtd_asc[tri, :max_op] = ramp_up * wtd_dsc[tri, :max_op]

        nrec[tri, :max_op] = rec_mask[tri, :max_op].sum() * nrec[tri, :max_op]

    return wtd_dsc, wtd_asc, nrec


def _build_nonrec_mask(data, inds, rec_mask, pres_mask):
    """Mask selecting the studied-but-NOT-recalled items of each list.

    Recalled item numbers are matched back into ``pres_itemnos`` and the
    result inverted. The match is exact on this dataset (3092/3092 recalled
    items), which is also the check that the two encodings agree.
    """
    pres_itemnos = data.pres_itemnos[inds]
    rec_itemnos = data.rec_itemnos[inds]

    nonrec_mask = pres_mask.copy()
    for trial in range(pres_itemnos.shape[0]):
        recalled = rec_itemnos[trial, rec_mask[trial]]
        found = np.isin(pres_itemnos[trial], recalled)
        if int(found.sum()) != len(set(recalled.tolist())):
            raise AnalysisConsistencyError(
                f"trial {trial}: {found.sum()} presented items matched "
                f"{len(set(recalled.tolist()))} distinct recalled item numbers"
            )
        nonrec_mask[trial] = ~found
    return nonrec_mask


def _check_values_contiguous(uni_vals):
    expected = np.arange(1, int(uni_vals.max()) + 1)
    if not np.array_equal(uni_vals, expected):
        raise AnalysisConsistencyError(
            "prec_byval indexes columns by value, which requires the values to "
            f"be exactly 1..{int(uni_vals.max())}; got {uni_vals.size} distinct"
        )


def _prec_byval(data, inds, subjects, rec_mask, pres_mask, list_length):
    """Recall probability by item value.

    The SPC restricted to items of one value, then averaged over serial
    positions. Column ``v - 1`` holds value ``v`` -- the matrix is indexed by
    value, not by position, which is only well defined because the values
    present are exactly 1..max.

    The value set comes from the FULL dataset rather than the condition subset,
    so the columns mean the same thing in ``both``, ``rand`` and ``temp``.
    """
    uni_vals = np.unique(data.pres_values)
    _check_values_contiguous(uni_vals)

    recalls = data.recalls[inds]
    rec_values = data.rec_values_act[inds]
    pres_values = data.pres_values[inds]

    n_subs = len(unique_subjects(subjects))
    out = np.full((n_subs, int(uni_vals.max())), np.nan)

    for v in uni_vals:
        # spc returns NaN where a subject never saw that value at a position;
        # that is what lets the nanmean below skip those positions.
        by_val = spc(
            recalls,
            subjects,
            list_length,
            rec_mask & (rec_values == v),
            pres_mask & (pres_values == v),
        )
        out[:, int(v) - 1] = _nan_reduce(np.nanmean, by_val, 1)
    return out


def _stray_act_nonrec_list(act_nrec_list, list_length):
    """Reproduce ana_vavi_e2frop.m:132, stored as ``act_nonrec_stray``.

    ``res.act_nonrec_list(res.act_nrec_list==data.list_length) = NaN`` assigns
    into a field that is never otherwise created. MATLAB grows it from nothing
    using a logical index, giving a column-major flat array that is NaN at the
    selected cells, 0 elsewhere, and truncated after the last selected cell.

    The value is never read by anything downstream. It is reproduced only so
    the field list matches. The name ``act_nonrec_list`` is used for the real
    measure -- the mean value of the non-recalled items -- which this MATLAB
    line was evidently reaching for but never computed.
    """
    flat = (act_nrec_list == list_length).ravel(order="F")
    hits = np.flatnonzero(flat)
    if hits.size == 0:
        return np.zeros(0)
    out = np.zeros(int(hits[-1]) + 1)
    out[hits] = np.nan
    return out


def _inner_ana_vavi(data, inds, incs0, fix_op_weights):
    """One condition's worth of analysis (the MATLAB nested function)."""
    res = {}

    recalls = data.recalls[inds]
    subjects = data.subject[inds]

    rec_mask = make_clean_recalls_mask2d(recalls)
    pres_mask = np.ones(data.pres_itemnos[inds].shape, dtype=bool)
    ll = data.list_length

    # -- recall probabilities ------------------------------------------------
    res["spcs"] = spc(recalls, subjects, ll, rec_mask, pres_mask)
    res["pfrs"] = pfr(recalls, subjects, ll, rec_mask, pres_mask)
    res["prec_byval"] = _prec_byval(
        data, inds, subjects, rec_mask, pres_mask, ll
    )

    # -- recall transitions --------------------------------------------------
    res["crps"] = crp(
        recalls, subjects, ll, rec_mask, rec_mask, pres_mask, pres_mask
    )
    res["temp_clusts"] = temp_fact(recalls, subjects, ll, rec_mask, pres_mask)

    # -- value estimation ----------------------------------------------------
    maxnlist = 55 if incs0 else 50

    res["est_list"] = value_from_mask(
        data.est_values[inds], subjects, np.ones(subjects.shape), maxnlist
    )
    res["act_rec_list"] = value_from_mask(
        data.rec_values_act[inds], subjects, rec_mask, maxnlist
    )
    res["est_rec_list"] = value_from_mask(
        data.rec_values_est[inds], subjects, rec_mask, maxnlist
    )
    res["abserr_rec_list_item"] = value_from_mask(
        np.abs(data.rec_values_act[inds] - data.rec_values_est[inds]),
        subjects,
        rec_mask,
        maxnlist,
    )

    # Mean value of the items that were studied but NOT recalled. This is the
    # measure ana_vavi_e2frop.m:132 gestured at but never computed; the stray
    # array that line does produce is kept as `act_nonrec_stray` below.
    res["act_nonrec_list"] = value_from_mask(
        data.pres_values[inds],
        subjects,
        _build_nonrec_mask(data, inds, rec_mask, pres_mask),
        maxnlist,
    )

    wtd_dsc, wtd_asc, nrec_mask = _output_position_weights(
        data.recalls, recalls, rec_mask, fix_op_weights
    )
    res["act_weightrec_list"] = value_from_mask(
        data.rec_values_act[inds], subjects, wtd_dsc, maxnlist
    )
    res["act_wtascrec_list"] = value_from_mask(
        data.rec_values_act[inds], subjects, wtd_asc, maxnlist
    )
    res["act_nrec_list"] = value_from_mask(
        nrec_mask, subjects, rec_mask.astype(np.float64), maxnlist
    )
    res["act_nonrec_stray"] = _stray_act_nonrec_list(res["act_nrec_list"], ll)

    # A list where everything was recalled has no non-recalled items to
    # average, and value_from_mask marks that with its empty-mask Inf.
    #
    # vcbeh nulls these via `act_nrec_list == ll`, but act_nrec_list is built
    # through the output-position weighting, so under the default (MATLAB-
    # faithful) row-misalignment bug it is unreliable for rand/temp. The Inf
    # sentinel comes straight from the non-recalled mask and is therefore
    # correct in every condition.
    res["act_nonrec_list"] = np.where(
        np.isinf(res["act_nonrec_list"]), np.nan, res["act_nonrec_list"]
    )

    res["act_pres_list"] = value_from_mask(
        data.pres_values[inds], subjects, pres_mask, maxnlist
    )

    # -- alignment checks (MATLAB dropped to `keyboard` here) ----------------
    if not np.array_equal(
        np.isnan(res["est_list"]), np.isnan(res["act_pres_list"])
    ):
        raise AnalysisConsistencyError(
            "est_list and act_pres_list disagree on which lists exist"
        )

    if np.isinf(res["act_rec_list"]).any():
        inf_ref = np.isinf(res["act_rec_list"])
        # est_list is NaN only for lists a subject does not have; act_rec_list
        # marks its no-recall lists with Inf, so the two NaN patterns agree.
        # Disagreement means the Inf sentinel did not land where expected.
        if np.array_equal(
            np.isnan(res["est_list"]), np.isnan(res["act_rec_list"])
        ):
            companions = {
                "act_weightrec_list": res["act_weightrec_list"],
                "act_wtascrec_list": res["act_wtascrec_list"],
                "abserr_rec_list_item": res["abserr_rec_list_item"],
                "act_nrec_list": res["act_nrec_list"],
            }
            if not np.array_equal(
                np.isnan(res["act_rec_list"]),
                np.isnan(res["act_weightrec_list"]),
            ):
                raise AnalysisConsistencyError(
                    "act_rec_list and act_weightrec_list disagree on NaN placement"
                )
            for name, arr in companions.items():
                if not np.array_equal(inf_ref, np.isinf(arr)):
                    raise AnalysisConsistencyError(
                        f"act_rec_list and {name} disagree on Inf placement"
                    )
            # Aligned, so the no-recall sentinel can become NaN and let the
            # omitnan means below do the right thing.
            #
            # est_rec_list is deliberately NOT in this list, matching
            # ana_vavi_e2frop.m:165-169: its Inf survives into the result.
            # Nothing downstream reads it, but a nanmean over it would give
            # Inf rather than skipping the no-recall list.
            for name in ("act_rec_list", *companions):
                res[name] = np.where(np.isinf(res[name]), np.nan, res[name])
        else:
            raise AnalysisConsistencyError(
                "act_rec_list has Inf sentinels but its NaN pattern does not "
                "match est_list"
            )

    # -- derived error measures ---------------------------------------------
    res["signerr_pres_list"] = res["est_list"] - res["act_pres_list"]
    res["abserr_rec_list"] = np.abs(res["act_rec_list"] - res["est_list"])
    res["abserr_pres_list"] = np.abs(res["act_pres_list"] - res["est_list"])

    res["abserr_wtrec_list"] = np.abs(
        res["act_weightrec_list"] - res["est_list"]
    )
    res["baseerr_rec_list"] = np.abs(res["act_pres_list"] - VALUE_BASELINE)

    res["abserr_pres_sub"] = _nan_reduce(np.nanmean, res["abserr_pres_list"], 1)
    res["abserr_rec_sub"] = _nan_reduce(np.nanmean, res["abserr_rec_list"], 1)
    res["abserr_rec_item_sub"] = _nan_reduce(
        np.nanmean, res["abserr_rec_list_item"], 1
    )
    res["abserr_wtrec_sub"] = _nan_reduce(np.nanmean, res["abserr_wtrec_list"], 1)
    res["mm_abserr_pres_sub"] = np.column_stack(
        [
            _nan_reduce(np.nanmin, res["abserr_pres_list"], 1),
            _nan_reduce(np.nanmax, res["abserr_pres_list"], 1),
        ]
    )
    res["mm_signerr_pres_sub"] = np.column_stack(
        [
            _nan_reduce(np.nanmin, res["signerr_pres_list"], 1),
            _nan_reduce(np.nanmax, res["signerr_pres_list"], 1),
        ]
    )

    res["act_rec_sub"] = _nan_reduce(np.nanmean, res["act_rec_list"], 1)
    res["act_pres_sub"] = _nan_reduce(np.nanmean, res["act_pres_list"], 1)

    # -- by-session error ----------------------------------------------------
    sessnos = data.session[inds].ravel()
    subwind = subjects.ravel()
    sublist = unique_subjects(subwind)
    session_matrix = np.full((len(sublist), maxnlist), np.nan)
    for si, subj in enumerate(sublist):
        these = sessnos[subwind == subj]
        session_matrix[si, : len(these)] = these

    nsess = len(np.unique(sessnos))
    res["abserr_rec_sess"] = mean_abserr_sess(
        res["abserr_rec_list"], session_matrix, sublist, nsess
    )
    res["abserr_pres_sess"] = mean_abserr_sess(
        res["abserr_pres_list"], session_matrix, sublist, nsess
    )

    return res


def ana_vavi_e2frop(datapath, bysess=False, fix_op_weights=False):
    """Run the analysis for all lists and for each list type.

    Parameters
    ----------
    datapath : str, Path, or Data
        A ``data_*.mat`` file, or an already-loaded :class:`~vcfrop.matio.Data`
        (for instance from :func:`vcfrop.rawio.load_raw`).
    bysess : bool
        Treat each subject-session as its own subject, by folding session into
        the subject id as ``subject + 1000 * session``.
    fix_op_weights : bool
        Correct the two output-position weighting bugs described in
        :func:`_output_position_weights`. Default False reproduces MATLAB.

    Returns
    -------
    dict
        ``{"unilist": ..., "both": {...}, "rand": {...}, "temp": {...}}``.
    """
    data = datapath if isinstance(datapath, Data) else load_data(datapath)

    if bysess:
        data = data.copy()
        data["subject_orig"] = data.subject
        data["subject"] = data.subject + 1000 * data.session

    res = {"unilist": unique_subjects(data.subject)}

    incs0 = bool(np.any(data.session == 0))
    pres_types = data.pres_types.ravel()

    conditions = {
        "both": np.ones(pres_types.shape, dtype=bool),
        "rand": pres_types == 1,
        "temp": pres_types == 2,
    }
    for name, inds in conditions.items():
        res[name] = _inner_ana_vavi(data, inds, incs0, fix_op_weights)

    return res
