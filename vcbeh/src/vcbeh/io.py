"""Loading and saving of the MATLAB ``.mat`` files.

All dtype normalization happens here, once.  That matters: in the MATLAB
original ``data.session`` is ``uint8``, so ``1000*data.session`` saturates at
255 and ``ana_vavi``'s ``bysess`` path silently fails to separate sessions
(bug B2).  Casting to ``int64`` at load time neutralizes that class of bug for
the whole pipeline instead of at ~20 call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from .types import AnaResult, Trials

# Per-trial scalars in the MATLAB struct, stored as (n, 1) columns there.
_SCALAR_FIELDS = ("subject", "session", "pres_types", "est_values", "trial")
_SCALAR_FLOAT_FIELDS = ("act_values",)
_MATRIX_INT_FIELDS = (
    "pres_itemnos",
    "pres_values",
    "rec_itemnos",
    "recalls",
    "intrusions",
)
_MATRIX_FLOAT_FIELDS = ("rec_values",)


def load_data(path: str | Path) -> Trials:
    """Read a ``data_*.mat`` file into a `Trials`.

    Integer fields become ``int64`` and value fields ``float64``; per-trial
    scalars are flattened to 1-D.
    """
    mat = loadmat(str(path), struct_as_record=False, squeeze_me=False)
    if "data" not in mat:
        raise KeyError(f"{path}: no 'data' variable (found {_varnames(mat)})")
    s = mat["data"][0, 0]
    present = set(s._fieldnames)

    kw: dict[str, Any] = {"list_length": int(np.asarray(s.list_length).ravel()[0])}
    for name in _SCALAR_FIELDS:
        kw[name] = (
            np.asarray(getattr(s, name)).ravel().astype(np.int64)
            if name in present
            else None
        )
    for name in _SCALAR_FLOAT_FIELDS:
        kw[name] = np.asarray(getattr(s, name)).ravel().astype(np.float64)
    for name in _MATRIX_INT_FIELDS:
        kw[name] = np.atleast_2d(np.asarray(getattr(s, name))).astype(np.int64)
    for name in _MATRIX_FLOAT_FIELDS:
        kw[name] = np.atleast_2d(np.asarray(getattr(s, name))).astype(np.float64)

    # `trial` is absent from some files and degenerate (all zeros) in others;
    # nothing in the pipeline reads it, so fill rather than fail.
    if kw.get("trial") is None:
        kw["trial"] = np.zeros(kw["subject"].shape[0], dtype=np.int64)

    return Trials(**kw)


def describe_data(trials: Trials) -> str:
    """Human-readable summary, including structural problems worth knowing about."""
    lines = [
        f"trials       : {trials.ntrials}",
        f"list_length  : {trials.list_length}",
        f"subjects     : {np.unique(trials.subject).size}",
        f"sessions     : {np.unique(trials.session).tolist()}",
        f"list types   : {np.unique(trials.pres_types).tolist()}",
        f"recall cols  : {trials.n_recall_cols}",
    ]
    for problem in structural_problems(trials):
        lines.append(f"WARNING      : {problem}")
    return "\n".join(lines)


def structural_problems(trials: Trials) -> list[str]:
    """Structural mismatches that will break the pipeline downstream.

    ``data_050526_ws0.mat`` trips the first of these: its ``recalls`` is
    1285x65 while ``rec_values`` is 1285x60, so masks built from ``recalls``
    cannot index ``rec_values``.
    """
    problems = []
    ncols = trials.recalls.shape[1]
    for name in ("rec_itemnos", "intrusions", "rec_values"):
        got = getattr(trials, name).shape[1]
        if got != ncols:
            problems.append(
                f"{name} has {got} columns but recalls has {ncols}; "
                "recall masks cannot be applied to it"
            )
    return problems


def load_res(path: str | Path) -> dict[str, Any]:
    """Read a ``res_*.mat`` file into nested plain dicts, for golden comparison.

    Only the fields actually present are returned -- the saved files are stale
    relative to the current ``ana_vavi`` and each is missing a different subset.
    """
    mat = loadmat(str(path), struct_as_record=False, squeeze_me=False)
    if "res" not in mat:
        raise KeyError(f"{path}: no 'res' variable (found {_varnames(mat)})")
    r = mat["res"][0, 0]
    out: dict[str, Any] = {"unilist": np.asarray(r.unilist).ravel().astype(np.int64)}
    for cond in ("both", "rand", "temp"):
        if cond not in r._fieldnames:
            continue
        c = getattr(r, cond)[0, 0]
        out[cond] = {
            f: np.atleast_2d(np.asarray(getattr(c, f))).astype(np.float64)
            for f in c._fieldnames
        }
    return out


def save_res(result: AnaResult, path: str | Path) -> None:
    """Write an `AnaResult` as a MATLAB-consumable ``res`` struct."""
    savemat(str(path), {"res": result.to_matlab()}, do_compression=True)


def _varnames(mat: dict) -> list[str]:
    return [k for k in mat if not k.startswith("__")]
