"""The consistency checks that were ``keyboard`` statements in ``ana_vavi.m``.

The MATLAB original dropped into the debugger at five points where an
invariant failed.  That is unusable non-interactively, so each becomes a named
function raising `ParityError` with enough context to find the offending
subject or trial.
"""

from __future__ import annotations

import numpy as np

# Subject exempted from the session-count check in the MATLAB original
# (ana_vavi.m:172), presumably a known data-collection irregularity.
SESSION_CHECK_EXEMPT = (615,)


class ParityError(AssertionError):
    """An invariant the MATLAB pipeline asserted has been violated."""


def check_nonrec_mask(trial: int, presented: np.ndarray, recalled: np.ndarray) -> None:
    """ana_vavi.m:119 -- every recalled item must be findable among the presented.

    If the recalled item numbers cannot be matched one-for-one against the
    presentation list, the non-recalled mask would be silently wrong.
    """
    if not np.array_equal(np.sort(presented), np.sort(recalled)):
        missing = sorted(set(recalled.tolist()) - set(presented.tolist()))
        raise ParityError(
            f"trial {trial}: recalled item numbers do not match the presented "
            f"list; {len(recalled)} recalled vs {len(presented)} matched"
            + (f", unmatched item numbers {missing}" if missing else "")
        )


def check_est_pres_alignment(est_list: np.ndarray, act_pres_list: np.ndarray) -> None:
    """ana_vavi.m:165 -- estimated and presented value must define the same lists."""
    if not np.array_equal(np.isnan(est_list), np.isnan(act_pres_list)):
        bad = np.argwhere(np.isnan(est_list) != np.isnan(act_pres_list))
        raise ParityError(
            "est_list and act_pres_list disagree on which lists exist at "
            f"{len(bad)} position(s); first at (subject row, list) "
            f"{tuple(bad[0])}"
        )


def check_session_counts(
    subject: np.ndarray, session: np.ndarray, incs0: bool
) -> None:
    """ana_vavi.m:173 -- list count per subject must match their session count.

    Expects ``5`` lists for session 0 when present, plus ``10`` per numbered
    session.  Uses the full dataset, not the condition subset, exactly as the
    original did.
    """
    offenders = []
    for subj in np.unique(subject):
        these = session[subject == subj]
        nsess = np.unique(these[these > 0]).size
        expected = int(incs0) * 5 + 10 * nsess
        if expected != these.size and int(subj) not in SESSION_CHECK_EXEMPT:
            offenders.append((int(subj), these.size, expected))
    if offenders:
        detail = "; ".join(
            f"subject {s}: {got} lists, expected {exp}" for s, got, exp in offenders
        )
        raise ParityError(f"session/list count mismatch -- {detail}")


def check_inf_alignment(fields: dict[str, np.ndarray], est_list: np.ndarray) -> None:
    """ana_vavi.m:195 and :199 -- the Inf "no recalls" sentinels must agree.

    ``act_rec_list`` marks lists with no valid recalls as Inf.  Before those are
    folded into NaN, every other *recall-derived* measure must carry Inf in
    exactly the same places, and the NaN pattern must still match ``est_list``.

    ``act_nonrec_list`` is deliberately **not** held to that pattern, which is
    where this diverges from the MATLAB original.  Its Inf means the opposite
    thing -- every presented item was recalled -- and those entries have
    already been set to NaN by the ``act_nrec_list == list_length`` sweep.  The
    two conditions are mutually exclusive for any non-empty list, so requiring
    the patterns to be equal (as ``ana_vavi.m:186`` does) is only satisfiable
    when no list has zero recalls.  ``data_050526_45sess.mat`` happens to have
    none, so the original passes there; ``data_050526_xs0.mat`` has one, and
    the MATLAB code drops into the debugger on it.  What is actually required
    is that no Inf survives in ``act_nonrec_list``.
    """
    reference = fields["act_rec_list"]

    if not np.array_equal(np.isnan(est_list), np.isnan(reference)):
        raise ParityError(
            "act_rec_list has Inf sentinels but its NaN pattern no longer "
            "matches est_list, so the two disagree on which lists exist"
        )

    leftover = np.argwhere(np.isinf(fields["act_nonrec_list"]))
    if leftover.size:
        raise ParityError(
            f"act_nonrec_list still has {len(leftover)} Inf sentinel(s) after "
            "the full-recall sweep; first at (subject row, list) "
            f"{tuple(leftover[0])}. Every list where all items were recalled "
            "should already have been set to NaN."
        )

    ref_inf = np.isinf(reference)
    for name, arr in fields.items():
        if name in ("act_rec_list", "act_nonrec_list"):
            continue
        if not np.array_equal(np.isinf(arr), ref_inf):
            extra = np.argwhere(np.isinf(arr) & ~ref_inf)
            missing = np.argwhere(ref_inf & ~np.isinf(arr))
            raise ParityError(
                f"{name} disagrees with act_rec_list on which lists had no "
                f"recalls: {len(extra)} extra Inf, {len(missing)} missing"
            )

    if not np.array_equal(np.isnan(fields["act_weightrec_list"]), np.isnan(reference)):
        raise ParityError(
            "act_weightrec_list and act_rec_list disagree on their NaN pattern"
        )
