"""Build a `Trials` directly from the raw event-level behavioural data.

The ``.mat`` files in this project were produced from these events by a MATLAB
step that is not part of the repository.  This module reconstructs the same
matrices in Python, which removes the dependence on those files -- two of which
are stale, and one of which (``ws0``) is unusable.

Verified against ``data_050526_45sess.mat`` and ``data_050526_xs0.mat``: every
matrix is reproduced exactly.  See ``tests/test_rawio.py``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .types import Trials

DEFAULT_EXPERIMENT = "VCBehOnly"

# Event types we read.  WORD is presentation, REC_WORD a recall attempt, and
# VALUE_RECALL the subject's single whole-list value estimate.
PRESENTATION_EVENT = "WORD"
RECALL_EVENT = "REC_WORD"
ESTIMATE_EVENT = "VALUE_RECALL"

# `storepointtype` carries the list type and is constant within a trial.
LIST_TYPE_CODES = {"random": 1, "Temporal": 2}

# Sentinel the .mat uses for an extra-list intrusion, whose item is unknown.
UNKNOWN_ITEM = -1


def load_raw(
    path: str | Path,
    *,
    include_session0: bool = False,
    experiment: str | None = DEFAULT_EXPERIMENT,
) -> Trials:
    """Read the raw behavioural events into a `Trials`.

    Parameters
    ----------
    path:
        Either the data directory (every ``sub-*/*.tsv`` under it is read, and
        the ``experiment`` column is honoured) or a single concatenated
        ``.csv``/``.tsv`` export.
    include_session0:
        Session 0 is a short intro session of 5 lists; numbered sessions have
        10.  **Excluding it also drops any subject who did nothing else** --
        8 of the 34 subjects in this dataset.
    experiment:
        Keep only rows whose ``experiment`` matches.  Ignored when the column
        is absent, as it is in the concatenated CSV export.  Pass ``None`` to
        disable the filter entirely.
    """
    events = _read_events(path, experiment)

    if not include_session0:
        events = events[events["session"] > 0]
    if events.empty:
        raise ValueError(f"{path}: no events left after filtering")

    presentations = events[events["trial_type"] == PRESENTATION_EVENT]
    if presentations.empty:
        raise ValueError(f"{path}: no {PRESENTATION_EVENT} events found")

    list_length = int(
        presentations.groupby(_TRIAL_KEYS).size().mode().iloc[0]
    )
    pres_by_trial = {
        key: group.sort_values("serialpos")
        for key, group in presentations.groupby(_TRIAL_KEYS)
    }
    # Trials missing presentation events are unusable; they cannot define a
    # value sequence to be estimated.
    keys = sorted(k for k, g in pres_by_trial.items() if len(g) == list_length)
    if not keys:
        raise ValueError(f"{path}: no complete {list_length}-item lists")

    recalls_by_trial = {
        key: group.sort_values("mstime")
        for key, group in events[events["trial_type"] == RECALL_EVENT].groupby(_TRIAL_KEYS)
    }
    estimates_by_trial = {
        key: group for key, group in events[events["trial_type"] == ESTIMATE_EVENT].groupby(_TRIAL_KEYS)
    }

    # One width for every recall matrix, so they can never disagree -- the
    # defect that makes data_050526_ws0.mat unusable.
    n_recall_cols = max(
        (len(recalls_by_trial[k]) for k in keys if k in recalls_by_trial), default=0
    )

    return _assemble(
        keys,
        pres_by_trial,
        recalls_by_trial,
        estimates_by_trial,
        list_length,
        n_recall_cols,
    )


_TRIAL_KEYS = ["subject_id", "session", "trial"]

_USED_COLUMNS = (
    "mstime trial_type experiment subject session trial serialpos itemno "
    "itemno_all itemvalue actualvalue intrusion storepointtype valuerecall"
).split()


def _read_events(path: str | Path, experiment: str | None) -> pd.DataFrame:
    """Load the events, honouring the experiment filter.

    The two on-disk forms are not interchangeable, and each is missing what the
    other has:

    * The **concatenated CSV export** is the only source of ``itemno_all``, the
      item numbering the ``.mat`` files use.  The per-session TSVs carry no item
      number at all on ``WORD`` rows.
    * The **TSVs** are the only source of the ``experiment`` column.

    So the CSV supplies the data and the TSVs supply the labelling, joined on
    subject and session.
    """
    path = Path(path)
    if path.is_dir():
        exports = sorted(path.glob("*.csv"))
        if not exports:
            raise FileNotFoundError(
                f"{path}: no concatenated .csv export. The per-session TSVs "
                "cannot be used on their own -- they carry no item numbers on "
                "presentation rows."
            )
        events = pd.read_csv(exports[0], low_memory=False)
        allowed = _sessions_for_experiment(path, experiment)
    else:
        sep = "\t" if path.suffix == ".tsv" else ","
        events = pd.read_csv(path, sep=sep, low_memory=False)
        allowed = None

    for column in _USED_COLUMNS:
        if column not in events.columns:
            events[column] = np.nan

    events = events.copy()
    events["subject_id"] = _subject_number(events["subject"])
    for column in ("session", "trial"):
        events[column] = events[column].astype("int64")

    if experiment is not None and events["experiment"].notna().any():
        events = events[events["experiment"].astype(str).str.strip() == experiment]
    elif allowed is not None:
        keep = pd.MultiIndex.from_arrays(
            [events["subject_id"], events["session"]]
        ).isin(allowed)
        events = events[keep]

    if events.empty:
        raise ValueError(f"{path}: no rows left for experiment {experiment!r}")
    return events


def _sessions_for_experiment(directory: Path, experiment: str | None):
    """``(subject_id, session)`` pairs whose TSV declares this experiment.

    Returns ``None`` only when there is nothing to filter *with* -- no TSVs, or
    none carrying an ``experiment`` column.  When TSVs do exist but none match,
    the result is an empty index, so an unrecognised experiment name filters
    everything out rather than silently passing it all through.
    """
    if experiment is None:
        return None
    files = sorted(glob.glob(str(directory / "sub-*" / "*.tsv")))
    if not files:
        return None

    pairs = set()
    usable = False
    for file in files:
        header = set(pd.read_csv(file, sep="\t", nrows=0).columns)
        if not {"experiment", "subject", "session"} <= header:
            continue
        usable = True
        table = pd.read_csv(
            file, sep="\t", usecols=["experiment", "subject", "session"], low_memory=False
        )
        table = table[table["experiment"].astype(str).str.strip() == experiment]
        if table.empty:
            continue
        subjects = _subject_number(table["subject"])
        pairs.update(zip(subjects.tolist(), table["session"].astype("int64").tolist()))

    if not usable:
        return None
    return pd.MultiIndex.from_tuples(sorted(pairs)) if pairs else pd.MultiIndex.from_tuples(
        [], names=["subject_id", "session"]
    )


def _subject_number(subject: pd.Series) -> pd.Series:
    """``LTP615`` -> ``615``; numeric identifiers pass through unchanged."""
    text = subject.astype(str).str.extract(r"(\d+)", expand=False)
    if text.isna().any():
        bad = sorted(subject[text.isna()].unique())[:5]
        raise ValueError(f"subject identifiers without a number: {bad}")
    return text.astype("int64")


def _assemble(
    keys,
    pres_by_trial,
    recalls_by_trial,
    estimates_by_trial,
    list_length: int,
    n_recall_cols: int,
) -> Trials:
    n = len(keys)
    subject = np.empty(n, dtype=np.int64)
    session = np.empty(n, dtype=np.int64)
    trial = np.empty(n, dtype=np.int64)
    pres_types = np.zeros(n, dtype=np.int64)
    est_values = np.zeros(n, dtype=np.int64)
    act_values = np.full(n, np.nan)

    pres_itemnos = np.zeros((n, list_length), dtype=np.int64)
    pres_values = np.zeros((n, list_length), dtype=np.int64)
    recalls = np.zeros((n, n_recall_cols), dtype=np.int64)
    rec_itemnos = np.zeros((n, n_recall_cols), dtype=np.int64)
    intrusions = np.zeros((n, n_recall_cols), dtype=np.int64)
    rec_values = np.full((n, n_recall_cols), np.nan)

    for row, key in enumerate(keys):
        shown = pres_by_trial[key]
        subject[row], session[row], trial[row] = key
        pres_itemnos[row] = shown["itemno_all"].to_numpy()
        pres_values[row] = shown["itemvalue"].to_numpy()
        act_values[row] = float(shown["actualvalue"].iloc[0])
        pres_types[row] = _list_type(shown["storepointtype"].iloc[0], key)

        estimate = estimates_by_trial.get(key)
        if estimate is not None and not estimate["valuerecall"].isna().all():
            est_values[row] = int(estimate["valuerecall"].iloc[0])

        responses = recalls_by_trial.get(key)
        if responses is None:
            continue
        _fill_recalls(
            responses,
            pres_itemnos[row],
            pres_values[row],
            recalls[row],
            rec_itemnos[row],
            intrusions[row],
            rec_values[row],
        )

    return Trials(
        list_length=list_length,
        subject=subject,
        session=session,
        trial=trial,
        pres_itemnos=pres_itemnos,
        pres_values=pres_values,
        pres_types=pres_types,
        rec_itemnos=rec_itemnos,
        recalls=recalls,
        intrusions=intrusions,
        rec_values=rec_values,
        est_values=est_values,
        act_values=act_values,
    )


def _list_type(storepointtype, key) -> int:
    try:
        return LIST_TYPE_CODES[str(storepointtype).strip()]
    except KeyError:
        raise ValueError(
            f"trial {key}: unknown storepointtype {storepointtype!r}; "
            f"expected one of {sorted(LIST_TYPE_CODES)}"
        ) from None


def _fill_recalls(
    responses: pd.DataFrame,
    pres_itemnos: np.ndarray,
    pres_values: np.ndarray,
    recalls: np.ndarray,
    rec_itemnos: np.ndarray,
    intrusions: np.ndarray,
    rec_values: np.ndarray,
) -> None:
    """Encode one trial's responses into the four recall matrices.

    ``intrusion`` drives the encoding: ``0`` is a correct recall, a positive
    value is a prior-list intrusion counting lists back, and ``-1`` is an
    extra-list intrusion.

    A handful of rows carry no ``intrusion`` **and** no ``serialpos`` while
    still naming an item.  Those are correct recalls with an unrecorded serial
    position, and are resolved by looking the item up in the presentation list
    -- which is what the MATLAB pipeline did.
    """
    for column, (_, event) in enumerate(responses.iterrows()):
        if column >= recalls.size:
            break  # matrices are sized to the longest trial, so this is a guard

        item = event["itemno_all"]
        if pd.isna(item):
            item = event["itemno"]
        rec_itemnos[column] = UNKNOWN_ITEM if pd.isna(item) else int(item)

        serial_pos = _serial_position(event, rec_itemnos[column], pres_itemnos)
        if serial_pos is not None:
            recalls[column] = serial_pos
            rec_values[column] = pres_values[serial_pos - 1]
        else:
            recalls[column] = UNKNOWN_ITEM
            intrusions[column] = -1 if pd.isna(event["intrusion"]) else int(event["intrusion"])


def _serial_position(event, itemno: int, pres_itemnos: np.ndarray) -> int | None:
    """Serial position of a correct recall, or None when it is an intrusion."""
    intrusion = event["intrusion"]

    if pd.isna(intrusion):
        # No intrusion code recorded: correct if the item was in this list.
        match = np.flatnonzero(pres_itemnos == itemno)
        return int(match[0]) + 1 if match.size else None

    if int(intrusion) != 0:
        return None

    serial_pos = event["serialpos"]
    if pd.isna(serial_pos):
        match = np.flatnonzero(pres_itemnos == itemno)
        return int(match[0]) + 1 if match.size else None
    return int(serial_pos)
