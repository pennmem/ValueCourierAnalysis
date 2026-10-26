"""Command line entry point: ``python -m vcbeh <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ana_vavi, describe_data, load_data, load_res, save_res
from .io import structural_problems
from .rawio import load_raw

# Extensions that mean raw event data rather than a built .mat.
_RAW_SUFFIXES = {".csv", ".tsv"}


def _load(path: str, *, include_session0: bool = False):
    """Load a dataset, accepting either a built .mat or the raw events.

    A directory or a .csv/.tsv is read as raw event data; anything else is
    treated as a .mat.  Rebuilding from raw is preferable where possible -- the
    saved .mat files are stale, and one of them is unusable.
    """
    target = Path(path)
    if target.is_dir() or target.suffix.lower() in _RAW_SUFFIXES:
        trials = load_raw(target, include_session0=include_session0)
    else:
        trials = load_data(target)
    for problem in structural_problems(trials):
        print(f"warning: {problem}", file=sys.stderr)
    return trials


def cmd_analyze(args) -> int:
    trials = _load(args.data)
    print(describe_data(trials), file=sys.stderr)
    result = ana_vavi(
        trials, bysess=args.bysess, ascending_weight_bug=not args.fix_ascending_weight
    )
    save_res(result, args.output)
    print(f"wrote {args.output}")
    return 0


def cmd_stats(args) -> int:
    from .stats import (
        all_vs_recalled,
        condition_contrast,
        condition_models,
        format_all_vs_recalled,
        format_condition_models,
        format_contrast,
        format_report,
        format_sweep,
        recalled_unweighted,
        recalled_vs_nonrecalled,
    )

    trials = _load(args.data, include_session0=args.include_session0)
    result = ana_vavi(trials)
    ll = trials.list_length
    print(format_report(result))
    print()

    for corrected, label in ((False, "uncorrected"), (True, "recall-corrected")):
        which = "recalled + non-recalled value" if corrected else "actual list value"
        print(format_condition_models(
            condition_models(result, ll, corrected=corrected),
            f"Estimate ~ {which}, by list type ({label})",
        ))
        print()
        print(format_contrast(
            condition_contrast(result, ll, corrected=corrected),
            f"Does it differ by list type? ({label})",
        ))
        print()

    print(format_all_vs_recalled(
        all_vs_recalled(result),
        "Estimate ~ 1 + ValueAll * ValueRecalled + (1|Subject), pooled",
    ))
    print()
    print(format_all_vs_recalled(
        all_vs_recalled(result, by_condition=True),
        "Estimate ~ 1 + ValueAll * ValueRecalled + (1|Subject), by list type",
    ))
    print()
    print(format_sweep(
        recalled_vs_nonrecalled(result, trials.list_length),
        "Value estimate ~ recalled + non-recalled value (both weighted by list share)",
    ))
    print()
    print(format_sweep(
        recalled_unweighted(result, trials.list_length),
        "Value estimate ~ recalled * non-recalled value (recalled term unweighted)",
    ))
    return 0


def cmd_plot(args) -> int:
    from .plots import res_forpres

    trials = _load(args.data)
    result = ana_vavi(trials)
    written = res_forpres.save_all(result, args.outdir, fmt=args.format)

    if args.pres_values:
        from .plots import pres_values

        written += pres_values.save_all(trials, Path(args.outdir) / "pres_values",
                                        fmt=args.format)
    for path in written:
        print(path)
    return 0


def cmd_describe(args) -> int:
    print(describe_data(_load(args.data)))
    if args.res:
        saved = load_res(args.res)
        for condition in ("both", "rand", "temp"):
            fields = sorted(saved.get(condition, {}))
            print(f"\n{args.res} [{condition}]: {len(fields)} fields")
            print("  " + ", ".join(fields))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vcbeh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="run the analysis and write a res .mat")
    p.add_argument("data")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--bysess", action="store_true",
                   help="treat each subject-session as its own subject "
                        "(diverges from MATLAB, which silently never split)")
    p.add_argument("--fix-ascending-weight", action="store_true",
                   help="use a true ascending output-position weight instead of "
                        "reproducing the defect at ana_vavi.m:138")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("stats", help="print contrasts, correlation and regressions")
    p.add_argument("data")
    p.add_argument("--include-session0", action="store_true",
                   help="raw input only: keep the 5-list intro session. Excluding "
                        "it also drops any subject who did nothing else")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("plot", help="render figures")
    p.add_argument("data")
    p.add_argument("-o", "--outdir", default="figs")
    p.add_argument("--format", default="png")
    p.add_argument("--pres-values", action="store_true",
                   help="also render one value-sequence figure per subject-session")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("describe", help="summarize a data file")
    p.add_argument("data")
    p.add_argument("--res", help="also list the fields of a saved result file")
    p.set_defaults(func=cmd_describe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
