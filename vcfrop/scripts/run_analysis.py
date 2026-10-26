"""Run the ported analysis and compare it against a saved MATLAB result.

    python scripts/run_analysis.py data_083026_xs0.mat [res_083026_xs0.mat]

With a reference given, prints a per-field pass/fail table. Fields the
reference predates (act_nrec_list, act_nonrec_list) are reported as "no
reference" rather than counted as failures.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcfrop.analysis import ana_vavi_e2frop
from vcfrop.matio import load_res

RTOL = 1e-9
ATOL = 1e-12


def compare_field(got, want):
    """Return (ok, detail) for one field."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)

    if got.shape != want.shape:
        return False, f"shape {got.shape} vs {want.shape}"

    same_nan = np.isnan(got) == np.isnan(want)
    if not same_nan.all():
        return False, f"{int((~same_nan).sum())} NaN placement mismatches"

    # est_rec_list keeps an Inf sentinel in both the port and the reference
    # (see the analysis module); compare those cells for placement only, since
    # inf - inf is not a meaningful difference.
    same_inf = np.isinf(got) == np.isinf(want)
    if not same_inf.all():
        return False, f"{int((~same_inf).sum())} Inf placement mismatches"

    finite = ~np.isnan(got) & ~np.isinf(got)
    n_inf = int(np.isinf(got).sum())
    note = f" (+{n_inf} matching Inf)" if n_inf else ""

    if not finite.any():
        return True, "all NaN/Inf"

    ok = np.allclose(got[finite], want[finite], rtol=RTOL, atol=ATOL)
    diff = float(np.max(np.abs(got[finite] - want[finite])))
    return ok, f"max|diff|={diff:.3e}{note}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("reference", nargs="?")
    ap.add_argument("--bysess", action="store_true")
    ap.add_argument("--fix-op-weights", action="store_true")
    args = ap.parse_args()

    res = ana_vavi_e2frop(
        args.data, bysess=args.bysess, fix_op_weights=args.fix_op_weights
    )
    print(f"analysed {args.data}"
          f"  (bysess={args.bysess}, fix_op_weights={args.fix_op_weights})")
    print(f"subjects: {len(res['unilist'])}")

    if not args.reference:
        for cond in ("both", "rand", "temp"):
            print(f"\n[{cond}]")
            for name, arr in res[cond].items():
                print(f"  {name:<22} {np.shape(arr)}")
        return 0

    ref = load_res(args.reference)
    print(f"reference: {args.reference}\n")

    n_pass = n_fail = n_noref = 0
    for cond in ("both", "rand", "temp"):
        print(f"[{cond}]")
        for name, arr in res[cond].items():
            if name not in ref[cond]:
                print(f"  {'--':<4} {name:<22} no reference")
                n_noref += 1
                continue
            ok, detail = compare_field(arr, ref[cond][name])
            print(f"  {'PASS' if ok else 'FAIL':<4} {name:<22} {detail}")
            n_pass += ok
            n_fail += not ok
        print()

    print(f"{n_pass} passed, {n_fail} failed, {n_noref} without a reference")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
