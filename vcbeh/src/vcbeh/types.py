"""Data containers for the vcbeh pipeline.

`Trials` replaces the MATLAB ``data`` struct; `CondResult`/`AnaResult` replace
``res``.  All integer fields are ``int64`` and all value fields ``float64`` --
see `vcbeh.io` for why that normalization has to happen exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

# Fields of `Trials` that carry one row per trial and must be subset together.
_TRIAL_FIELDS = (
    "subject",
    "session",
    "pres_itemnos",
    "pres_values",
    "pres_types",
    "rec_itemnos",
    "recalls",
    "intrusions",
    "rec_values",
    "est_values",
    "act_values",
    "trial",
)


@dataclass(frozen=True)
class Trials:
    """One row per study/test list.

    Scalars-per-trial (``subject``, ``session``, ``pres_types``, ``est_values``,
    ``act_values``, ``trial``) are 1-D of length ``ntrials``.  The rest are 2-D:
    ``pres_*`` are ``(ntrials, list_length)`` and ``rec_*``/``recalls``/
    ``intrusions`` are ``(ntrials, n_recall_cols)``.

    In ``recalls``, entries are serial positions ``1..list_length``, ``0`` is
    padding, and ``-1`` marks an intrusion.
    """

    list_length: int
    subject: np.ndarray
    session: np.ndarray
    pres_itemnos: np.ndarray
    pres_values: np.ndarray
    pres_types: np.ndarray
    rec_itemnos: np.ndarray
    recalls: np.ndarray
    intrusions: np.ndarray
    rec_values: np.ndarray
    est_values: np.ndarray
    act_values: np.ndarray
    trial: np.ndarray

    @property
    def ntrials(self) -> int:
        return int(self.subject.shape[0])

    @property
    def n_recall_cols(self) -> int:
        return int(self.recalls.shape[1])

    def subset(self, inds: np.ndarray) -> "Trials":
        """Rows selected by a boolean or integer index.

        This is the single audited place where the MATLAB ``data.X(inds,:)``
        idiom -- repeated ~15 times in ``inner_ana_vavi`` -- is applied, so a
        field can never be left un-subset by accident.
        """
        inds = np.asarray(inds)
        if inds.dtype == bool and inds.shape[0] != self.ntrials:
            raise ValueError(
                f"boolean index has {inds.shape[0]} entries, expected {self.ntrials}"
            )
        kw: dict[str, Any] = {"list_length": self.list_length}
        for name in _TRIAL_FIELDS:
            kw[name] = getattr(self, name)[inds]
        return Trials(**kw)

    def __post_init__(self) -> None:
        n = self.subject.shape[0]
        for name in _TRIAL_FIELDS:
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"field {name!r} has {arr.shape[0]} rows, expected {n}"
                )


# Every field ana_vavi's inner function writes, in assignment order.
COND_FIELDS = (
    "spcs",
    "pfrs",
    "prec_byval",
    "crps",
    "temp_clusts",
    "est_list",
    "act_rec_list",
    "act_nonrec_list",
    "act_weightrec_list",
    "act_wtascrec_list",
    "act_wtdscrec_list",
    "act_nrec_list",
    "act_pres_list",
    "signerr_pres_list",
    "abserr_rec_list",
    "abserr_pres_list",
    "abserr_wtrec_list",
    "baseerr_rec_list",
    "abserr_pres_sub",
    "abserr_rec_sub",
    "abserr_wtrec_sub",
    "mm_abserr_pres_sub",
    "mm_signerr_pres_sub",
    "act_rec_sub",
    "act_pres_sub",
    "abserr_rec_sess",
    "abserr_pres_sess",
)


@dataclass
class CondResult:
    """Results for one condition (all lists, random lists, or autocorrelated)."""

    spcs: np.ndarray
    pfrs: np.ndarray
    prec_byval: np.ndarray
    crps: np.ndarray
    temp_clusts: np.ndarray
    est_list: np.ndarray
    act_rec_list: np.ndarray
    act_nonrec_list: np.ndarray
    act_weightrec_list: np.ndarray
    act_wtascrec_list: np.ndarray
    act_wtdscrec_list: np.ndarray
    act_nrec_list: np.ndarray
    act_pres_list: np.ndarray
    signerr_pres_list: np.ndarray
    abserr_rec_list: np.ndarray
    abserr_pres_list: np.ndarray
    abserr_wtrec_list: np.ndarray
    baseerr_rec_list: np.ndarray
    abserr_pres_sub: np.ndarray
    abserr_rec_sub: np.ndarray
    abserr_wtrec_sub: np.ndarray
    mm_abserr_pres_sub: np.ndarray
    mm_signerr_pres_sub: np.ndarray
    act_rec_sub: np.ndarray
    act_pres_sub: np.ndarray
    abserr_rec_sess: np.ndarray
    abserr_pres_sess: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class AnaResult:
    """Top-level result: the three conditions plus the subject list."""

    unilist: np.ndarray
    both: CondResult
    rand_: CondResult  # named rand_ to avoid shadowing; exported to MATLAB as "rand"
    temp: CondResult

    def to_matlab(self) -> dict[str, Any]:
        """Struct-shaped dict for ``scipy.io.savemat``, using MATLAB's names."""
        return {
            "unilist": self.unilist.reshape(-1, 1),
            "both": self.both.to_dict(),
            "rand": self.rand_.to_dict(),
            "temp": self.temp.to_dict(),
        }

    def conditions(self) -> dict[str, CondResult]:
        return {"both": self.both, "rand": self.rand_, "temp": self.temp}
