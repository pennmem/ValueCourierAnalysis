"""Port of stats_vaviv2.m and linreg_estval_presrecallval.m.

    python scripts/run_stats.py data_083026_xs0.mat
    python scripts/run_stats.py --from-res res_083026_xs0.mat
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcfrop.analysis import ana_vavi_e2frop
from vcfrop.matio import load_res
from vcfrop.regression import run_regression
from vcfrop.stats import run_stats


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("data", nargs="?", help="a data_*.mat to analyse")
    src.add_argument("--from-res", help="a saved res_*.mat to read instead")
    ap.add_argument("--latex", default=None, help="write the LaTeX table here")
    ap.add_argument("--fix-op-weights", action="store_true")
    args = ap.parse_args()

    if args.from_res:
        res = load_res(args.from_res)
    else:
        res = ana_vavi_e2frop(args.data, fix_op_weights=args.fix_op_weights)

    print("=" * 62)
    print("stats_vaviv2")
    print("=" * 62)
    run_stats(res)

    print()
    print("=" * 62)
    print("linreg_estval_presrecallval")
    print("=" * 62)
    run_regression(res, latex_path=args.latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
