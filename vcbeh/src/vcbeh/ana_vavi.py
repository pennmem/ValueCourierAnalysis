"""Port of ``ana_vavi.m`` -- the analysis hub.

Runs the full set of recall and value-estimation measures three times: over all
lists, over random-value lists only, and over autocorrelated-value lists only.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from .checks import (
    check_est_pres_alignment,
    check_inf_alignment,
    check_nonrec_mask,
    check_session_counts,
)
from .embam import crp, make_clean_recalls_mask2d, pfr, spc, temp_fact
from .primitives import mean_abserr_sess, value_from_mask
from .types import AnaResult, CondResult, Trials

# Midpoint of the 1-50 value range, used as the no-information baseline.
VALUE_BASELINE = 25

# Recall-derived measures whose Inf "no recalls" sentinels must agree.
_INF_ALIGNED = (
    "act_rec_list",
    "act_weightrec_list",
    "act_wtascrec_list",
    "act_wtdscrec_list",
    "act_nrec_list",
    "act_nonrec_list",
)


def ana_vavi(
    trials: Trials,
    bysess: bool = False,
    ascending_weight_bug: bool = True,
) -> AnaResult:
    """Run every analysis for all lists and for each list type.

    Parameters
    ----------
    trials:
        The loaded dataset.
    bysess:
        Treat each subject-session as its own "subject".  **This diverges from
        MATLAB.**  There, ``data.session`` is ``uint8`` so ``1000*session``
        saturates at 255 and every session received the same offset, meaning
        the split silently never happened.  Here it works as intended, so
        results will not match a MATLAB run with ``bysess=true``.
    ascending_weight_bug:
        Reproduce the defect at ``ana_vavi.m:138``, where the "ascending"
        output-position weight is built from the already-overwritten descending
        weight and so comes out as a symmetric tent, ``op*(max_op-op+1)``.
        Left on by default because it is what every saved result was computed
        with; set ``False`` for a true ascending weight, which changes
        ``act_wtascrec_list`` by up to ~8 value units.
    """
    if bysess:
        trials = dataclasses.replace(
            trials, subject=trials.subject + 1000 * trials.session
        )

    unilist = np.unique(trials.subject)
    incs0 = bool(np.any(trials.session == 0))

    conditions = {
        "both": np.ones(trials.ntrials, dtype=bool),
        "rand_": trials.pres_types == 1,
        "temp": trials.pres_types == 2,
    }
    results = {
        name: _inner_ana_vavi(trials, inds, incs0, ascending_weight_bug)
        for name, inds in conditions.items()
    }
    return AnaResult(unilist=unilist, **results)


def _inner_ana_vavi(
    data: Trials,
    inds: np.ndarray,
    incs0: bool,
    ascending_weight_bug: bool,
) -> CondResult:
    t = data.subset(inds)
    ll = data.list_length
    subject = t.subject

    rec_mask = make_clean_recalls_mask2d(t.recalls)
    pres_mask = np.ones(t.pres_itemnos.shape, dtype=bool)

    # -- recall probabilities -------------------------------------------------
    spcs = spc(t.recalls, subject, ll, rec_mask, pres_mask)
    pfrs = pfr(t.recalls, subject, ll, rec_mask, pres_mask)

    # Note both of these come from the *full* dataset, not the condition
    # subset, exactly as the MATLAB original did.
    sublist_all = np.unique(data.subject)
    uni_vals = np.unique(data.pres_values)
    _check_values_contiguous(uni_vals)

    # prec_byval is indexed by value, not by position -- column v-1 holds
    # value v.  This is only well-defined because values are exactly 1..max.
    prec_byval = np.full((sublist_all.size, int(uni_vals.max())), np.nan)
    for v in uni_vals:
        by_val = spc(
            t.recalls,
            subject,
            ll,
            rec_mask & (t.rec_values == v),
            pres_mask & (t.pres_values == v),
        )
        prec_byval[:, int(v) - 1] = _nanmean(by_val, axis=1)

    # -- recall transitions ---------------------------------------------------
    crps = crp(t.recalls, subject, ll, rec_mask, rec_mask, pres_mask, pres_mask)
    temp_clusts = temp_fact(t.recalls, subject, ll, rec_mask, pres_mask)

    # -- value estimation -----------------------------------------------------
    maxnlist = 55 if incs0 else 50
    ones_col = np.ones((t.ntrials, 1), dtype=bool)

    est_list = value_from_mask(
        t.est_values[:, None].astype(np.float64), subject, ones_col, maxnlist
    )
    act_rec_list = value_from_mask(t.rec_values, subject, rec_mask, maxnlist)

    nonrec_mask = _build_nonrec_mask(t, rec_mask, pres_mask)
    act_nonrec_list = value_from_mask(
        t.pres_values.astype(np.float64), subject, nonrec_mask, maxnlist
    )

    wtd_dsc, wtd_asc, nrec_weights = _output_position_weights(
        t.recalls, rec_mask, ascending_weight_bug
    )
    act_weightrec_list = value_from_mask(t.rec_values, subject, wtd_dsc, maxnlist)
    act_wtascrec_list = value_from_mask(t.rec_values, subject, wtd_asc, maxnlist)
    act_wtdscrec_list = value_from_mask(t.rec_values, subject, wtd_dsc, maxnlist)
    # Averaging a constant-per-list matrix against the plain recall mask is a
    # roundabout way of counting correct recalls, but it keeps the result
    # aligned with every other per-list measure.
    act_nrec_list = value_from_mask(
        nrec_weights, subject, rec_mask.astype(np.float64), maxnlist
    )

    # A list where everything was recalled has no non-recalled items to average.
    act_nonrec_list[act_nrec_list == ll] = np.nan

    act_pres_list = value_from_mask(
        t.pres_values.astype(np.float64), subject, pres_mask, maxnlist
    )

    check_est_pres_alignment(est_list, act_pres_list)
    check_session_counts(data.subject, data.session, incs0)

    fields = {
        "act_rec_list": act_rec_list,
        "act_weightrec_list": act_weightrec_list,
        "act_wtascrec_list": act_wtascrec_list,
        "act_wtdscrec_list": act_wtdscrec_list,
        "act_nrec_list": act_nrec_list,
        "act_nonrec_list": act_nonrec_list,
    }
    if np.isinf(act_rec_list).any():
        check_inf_alignment(fields, est_list)
        for arr in fields.values():
            arr[np.isinf(arr)] = np.nan

    # -- error measures -------------------------------------------------------
    signerr_pres_list = est_list - act_pres_list
    abserr_rec_list = np.abs(act_rec_list - est_list)
    abserr_pres_list = np.abs(act_pres_list - est_list)
    abserr_wtrec_list = np.abs(act_weightrec_list - est_list)
    baseerr_rec_list = np.abs(act_pres_list - VALUE_BASELINE)

    # MATLAB's min/max skip NaN, so these are nanmin/nanmax.
    mm_abserr_pres_sub = np.column_stack(
        (_nanmin(abserr_pres_list), _nanmax(abserr_pres_list))
    )
    mm_signerr_pres_sub = np.column_stack(
        (_nanmin(signerr_pres_list), _nanmax(signerr_pres_list))
    )

    # -- per-session aggregation ---------------------------------------------
    session_matrix = _build_session_matrix(subject, t.session, maxnlist)
    sublist = np.unique(subject)
    nsess = np.unique(t.session).size
    abserr_rec_sess = mean_abserr_sess(
        abserr_rec_list, session_matrix, sublist, nsess
    )
    abserr_pres_sess = mean_abserr_sess(
        abserr_pres_list, session_matrix, sublist, nsess
    )

    return CondResult(
        spcs=spcs,
        pfrs=pfrs,
        prec_byval=prec_byval,
        crps=crps,
        temp_clusts=temp_clusts,
        est_list=est_list,
        act_rec_list=act_rec_list,
        act_nonrec_list=act_nonrec_list,
        act_weightrec_list=act_weightrec_list,
        act_wtascrec_list=act_wtascrec_list,
        act_wtdscrec_list=act_wtdscrec_list,
        act_nrec_list=act_nrec_list,
        act_pres_list=act_pres_list,
        signerr_pres_list=signerr_pres_list,
        abserr_rec_list=abserr_rec_list,
        abserr_pres_list=abserr_pres_list,
        abserr_wtrec_list=abserr_wtrec_list,
        baseerr_rec_list=baseerr_rec_list,
        abserr_pres_sub=_nanmean(abserr_pres_list, axis=1)[:, None],
        abserr_rec_sub=_nanmean(abserr_rec_list, axis=1)[:, None],
        abserr_wtrec_sub=_nanmean(abserr_wtrec_list, axis=1)[:, None],
        mm_abserr_pres_sub=mm_abserr_pres_sub,
        mm_signerr_pres_sub=mm_signerr_pres_sub,
        act_rec_sub=_nanmean(act_rec_list, axis=1)[:, None],
        act_pres_sub=_nanmean(act_pres_list, axis=1)[:, None],
        abserr_rec_sess=abserr_rec_sess,
        abserr_pres_sess=abserr_pres_sess,
    )


def _build_nonrec_mask(
    t: Trials, rec_mask: np.ndarray, pres_mask: np.ndarray
) -> np.ndarray:
    """Mask selecting the *studied but not recalled* items of each list.

    Derived by matching recalled item numbers back to the presentation list,
    which is also where the original verified that the two agree.
    """
    nonrec_mask = pres_mask.copy()
    for trial in range(t.ntrials):
        recalled = t.rec_itemnos[trial, rec_mask[trial]]
        found = np.isin(t.pres_itemnos[trial], recalled)
        check_nonrec_mask(trial, t.pres_itemnos[trial][found], recalled)
        nonrec_mask[trial] = ~found
    return nonrec_mask


def _output_position_weights(
    recalls: np.ndarray, rec_mask: np.ndarray, ascending_weight_bug: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Output-position weightings of the recall mask.

    Returns the descending weight, the ascending weight, and a mask carrying
    each list's correct-recall count.  Output positions run to the *last*
    response of the trial, so intrusions still consume positions -- deliberate
    in the original, since an error still occupies a slot in the output order.
    """
    base = rec_mask.astype(np.float64)
    wtd_dsc = base.copy()
    wtd_asc = base.copy()
    nrec_weights = base.copy()

    for trial in range(recalls.shape[0]):
        responded = np.nonzero(recalls[trial] > 0)[0]
        if responded.size == 0:
            continue
        max_op = int(responded[-1]) + 1
        sl = slice(0, max_op)

        wtd_dsc[trial, sl] = np.arange(max_op, 0, -1) * wtd_dsc[trial, sl]
        if ascending_weight_bug:
            # ana_vavi.m:138 reads the descending weight just written on the
            # line above instead of the plain mask, giving a tent shape.
            wtd_asc[trial, sl] = np.arange(1, max_op + 1) * wtd_dsc[trial, sl]
        else:
            wtd_asc[trial, sl] = np.arange(1, max_op + 1) * base[trial, sl]
        nrec_weights[trial, sl] = rec_mask[trial, sl].sum() * nrec_weights[trial, sl]

    return wtd_dsc, wtd_asc, nrec_weights


def _build_session_matrix(
    subject: np.ndarray, session: np.ndarray, maxnlist: int
) -> np.ndarray:
    """Session number of each list slot; ``(nsub, maxnlist)``, NaN-padded."""
    sublist = np.unique(subject)
    out = np.full((sublist.size, maxnlist), np.nan)
    for si, subj in enumerate(sublist):
        these = session[subject == subj]
        out[si, : these.size] = these
    return out


def _check_values_contiguous(uni_vals: np.ndarray) -> None:
    expected = np.arange(1, int(uni_vals.max()) + 1)
    if not np.array_equal(uni_vals, expected):
        raise ValueError(
            "prec_byval indexes columns by value, which requires values to be "
            f"exactly 1..{uni_vals.max()}; got {uni_vals.size} distinct values"
        )


def _nanmean(a: np.ndarray, axis: int = 1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
        return np.nanmean(a, axis=axis)


def _nanmin(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(a, axis=1)


def _nanmax(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(a, axis=1)
