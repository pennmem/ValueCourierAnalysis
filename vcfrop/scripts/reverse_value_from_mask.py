"""Phase 0 gate: recover the definition of the missing ``value_from_mask``.

The MATLAB helper is not on this machine, so its behavior is inferred by testing
candidate definitions against the four outputs saved in res_083026_xs0.mat that
were produced by it:

    est_list           values=est_values (n,1),      mask=all true
    act_pres_list      values=pres_values (n,15),    mask=all true
    act_rec_list       values=rec_values_act (n,60), mask=clean-recall mask
    act_weightrec_list values=rec_values_act (n,60), mask=descending OP weights

The last one is the discriminating case: its mask is non-binary, so only a
weighted mean reproduces it.

Run:  python scripts/reverse_value_from_mask.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcfrop.masks import make_clean_recalls_mask2d, unique_subjects
from vcfrop.matio import load_data, load_res

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-12


# ---------------------------------------------------------------- candidates


def _by_subject(values, subjects, mask, maxnlist, combine):
    """Shared scaffolding: per subject, one number per list, NaN-padded.

    Masked-out entries are zeroed rather than multiplied through, because
    rec_values_act carries NaN outside the recalled slots and NaN*0 would
    poison the sum.
    """
    subjects = np.asarray(subjects).ravel()
    subs = unique_subjects(subjects)
    out = np.full((len(subs), maxnlist), np.nan)

    for si, subj in enumerate(subs):
        rows = np.flatnonzero(subjects == subj)
        for li, row in enumerate(rows):
            m = mask[row]
            v = np.where(m != 0, values[row], 0.0)
            out[si, li] = combine(v, m)
    return out


def cand_weighted_mean_inf(values, subjects, mask, maxnlist):
    """Weighted mean; Inf when the mask is empty for that list."""

    def combine(v, m):
        den = np.sum(m)
        if den == 0:
            return np.inf
        return np.sum(v * m) / den

    return _by_subject(values, subjects, mask, maxnlist, combine)


def cand_weighted_mean_nan(values, subjects, mask, maxnlist):
    """Weighted mean; NaN when the mask is empty."""

    def combine(v, m):
        den = np.sum(m)
        if den == 0:
            return np.nan
        return np.sum(v * m) / den

    return _by_subject(values, subjects, mask, maxnlist, combine)


def cand_masked_sum(values, subjects, mask, maxnlist):
    """Plain masked sum, no normalization."""
    return _by_subject(
        values, subjects, mask, maxnlist, lambda v, m: np.sum(v * m)
    )


def cand_mean_over_nnz(values, subjects, mask, maxnlist):
    """Sum of masked values divided by the COUNT of nonzero mask entries.

    Differs from the weighted mean only for non-binary masks.
    """

    def combine(v, m):
        nnz = np.count_nonzero(m)
        if nnz == 0:
            return np.inf
        return np.sum(v * m) / nnz

    return _by_subject(values, subjects, mask, maxnlist, combine)


CANDIDATES = {
    "weighted_mean_inf": cand_weighted_mean_inf,
    "weighted_mean_nan": cand_weighted_mean_nan,
    "masked_sum": cand_masked_sum,
    "mean_over_nnz": cand_mean_over_nnz,
}


# ------------------------------------------------------------------ harness


def descending_op_weights(recalls, rec_mask):
    """Reproduce the ``wtd_rec_mask_dsc`` construction (ana_vavi_e2frop.m:106-120).

    Uses the same row indexing as the MATLAB source. This script only ever runs
    on the unfiltered ``both`` condition, where the source's row-index bug is
    inert, so the two agree here.
    """
    weights = rec_mask.astype(np.float64)
    for tri in range(recalls.shape[0]):
        positive = np.flatnonzero(recalls[tri] > 0)
        if positive.size == 0:
            continue
        max_op = positive[-1] + 1  # 1-based count of output positions
        weights[tri, :max_op] = np.arange(max_op, 0, -1) * weights[tri, :max_op]
    return weights


def compare(got, want):
    """Max absolute difference, treating NaN==NaN and Inf==Inf as equal."""
    if got.shape != want.shape:
        return None, f"shape {got.shape} vs {want.shape}"

    both_nan = np.isnan(got) & np.isnan(want)
    both_pinf = np.isposinf(got) & np.isposinf(want)
    both_ninf = np.isneginf(got) & np.isneginf(want)
    special = both_nan | both_pinf | both_ninf

    mismatched_special = (
        np.isnan(got) | np.isinf(got) | np.isnan(want) | np.isinf(want)
    ) & ~special
    if mismatched_special.any():
        n = int(mismatched_special.sum())
        return None, f"{n} NaN/Inf placement mismatches"

    finite = ~special
    if not finite.any():
        return 0.0, None
    return float(np.max(np.abs(got[finite] - want[finite]))), None


def main():
    data = load_data(ROOT / "data_083026_xs0.mat")
    res = load_res(ROOT / "res_083026_xs0.mat")
    both = res["both"]

    subjects = data.subject
    rec_mask = make_clean_recalls_mask2d(data.recalls)
    maxnlist = both["est_list"].shape[1]

    print(f"maxnlist inferred from saved res: {maxnlist}")
    print(f"trials={data.subject.shape[0]}  subjects={len(unique_subjects(subjects))}")
    print()

    cases = [
        (
            "est_list",
            data.est_values,
            np.ones_like(data.est_values, dtype=np.float64),
            both["est_list"],
        ),
        (
            "act_pres_list",
            data.pres_values,
            np.ones_like(data.pres_values, dtype=np.float64),
            both["act_pres_list"],
        ),
        (
            "act_rec_list",
            data.rec_values_act,
            rec_mask.astype(np.float64),
            both["act_rec_list"],
        ),
        (
            "act_weightrec_list",
            data.rec_values_act,
            descending_op_weights(data.recalls, rec_mask),
            both["act_weightrec_list"],
        ),
    ]

    header = f"{'candidate':<20}" + "".join(f"{name:>22}" for name, *_ in cases)
    print(header)
    print("-" * len(header))

    winners = []
    for cand_name, func in CANDIDATES.items():
        cells = []
        all_pass = True
        for _, values, mask, want in cases:
            got = func(values, subjects, mask, maxnlist)
            diff, problem = compare(got, want)
            if problem is not None:
                cells.append(f"{problem:>22}")
                all_pass = False
            else:
                cells.append(f"{diff:>22.3e}")
                if diff > TOL:
                    all_pass = False
        print(f"{cand_name:<20}" + "".join(cells))
        if all_pass:
            winners.append(cand_name)

    print()

    # Disambiguation -------------------------------------------------------
    #
    # weighted_mean_inf and weighted_mean_nan differ on exactly one cell: the
    # single list in this dataset with zero clean recalls (row 34, all
    # intrusions). The saved res has NaN there, which matches _nan directly.
    #
    # But ana_vavi_e2frop.m:155-156 documents the opposite contract -- "these
    # might not align if no recalls, which were set to Inf to distinguish from
    # default NaN" -- and lines 157-169 convert those Infs to NaN before the
    # result is saved. So the faithful reading is that value_from_mask returns
    # Inf and the analysis converts it. Both reproduce the saved file; only the
    # Inf version exercises the control flow the source actually describes.
    #
    # Confirm that Inf + conversion lands on the saved values.
    ok = True
    for name, values, mask, want in cases:
        got = cand_weighted_mean_inf(values, subjects, mask, maxnlist)
        n_inf = int(np.isinf(got).sum())
        got = np.where(np.isinf(got), np.nan, got)
        diff, problem = compare(got, want)
        status = problem if problem is not None else f"max|diff|={diff:.3e}"
        print(f"  inf+convert  {name:<20} infs={n_inf:<3} {status}")
        if problem is not None or diff > TOL:
            ok = False

    print()
    if ok:
        print(
            "GATE PASSED: weighted mean, Inf on empty mask, Inf->NaN conversion\n"
            f"             in the analysis. Reproduces all four outputs within {TOL:g}."
        )
        return 0
    print("GATE FAILED: Inf + conversion does not reproduce the saved outputs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
