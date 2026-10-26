"""Build trial matrices from the raw vcfrop (e2frop) event log.

The log is BIDS-style, one TSV per subject-session, under a directory of
``sub-LTP*/`` folders::

    sub-LTP657/sub-LTP657_ses-1_task-vcfrop_beh.tsv
    sub-LTP657/unmod-sub-LTP657_ses-5_task-vcfrop_beh.tsv   <- backup, ignored

Files prefixed ``unmod-`` are pre-curation backups of a session that was later
hand-corrected; ``modifiedtrials.xlsx`` in the same directory logs what changed
and why. They are skipped unless ``modified=False``.

Row types that matter:

===================== =========================================================
``WORD``              one per studied item: ``serialpos``, ``item``,
                      ``itemvaluecorrect``, ``storepointtype``
``REC_WORD``          one per recall attempt, in time order: ``serialpos``
                      (already resolved), ``intrusion``, ``item``
``REC_WORD_VV``       a vocalization, ``<>`` -- not a recall attempt
``ITEM_VALUE_RECALL`` the value the subject gave for the item they just
                      recalled; ``itemvalueguess``
``AVG_VALUE_RECALL``  the one whole-list estimate: ``avgvalueguess``,
                      ``avgvaluecorrect``
===================== =========================================================

A value is attached to the recall it *follows in time*, rather than by counting
positions, so a vocalization or a stray event between the two does not shift
every later pairing.

``itemno`` is ``n/a`` on ``WORD`` rows, so item numbers are assigned here from
the word strings (a sorted wordpool over the whole dataset). They are internally
consistent -- ``rec_itemnos[i]`` equals ``pres_itemnos[recalls[i] - 1]`` for
every clean recall, which is all anything downstream needs -- but they do NOT
match the numbering in ``data_083026_xs0.mat``, which came from a wordpool file
that is not in this tree.
"""

import csv
import re
from pathlib import Path

import numpy as np

from .matio import Data

#: storepointtype -> pres_types code
LIST_TYPE_CODES = {"random": 1, "temporal": 2}

_FILENAME = re.compile(
    r"^(?P<unmod>unmod-)?sub-(?P<subject>LTP\d+)_ses-(?P<session>\d+)"
    r"_task-vcfrop_beh\.tsv$"
)

_PRESENT = "WORD"
_RECALL = "REC_WORD"
_VOCALIZATION = "REC_WORD_VV"
_ITEM_VALUE = "ITEM_VALUE_RECALL"
_LIST_VALUE = "AVG_VALUE_RECALL"


def _number(text, default=np.nan):
    if text is None:
        return default
    text = text.strip()
    if text in ("", "n/a", "NaN"):
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _session_files(root, modified=True):
    """Every session TSV, as ``(subject_number, session, path)``, sorted."""
    found = []
    for path in sorted(Path(root).glob("sub-*/*.tsv")):
        m = _FILENAME.match(path.name)
        if m is None:
            continue
        is_backup = bool(m.group("unmod"))
        if is_backup == modified:
            continue
        found.append(
            (int(re.sub(r"\D", "", m.group("subject"))), int(m.group("session")), path)
        )
    return sorted(found)


def _read_events(path):
    with open(path, newline="") as fh:
        return [
            row
            for row in csv.DictReader(fh, delimiter="\t")
            if row.get("trial_type") and not row["trial_type"].startswith("PRACTICE")
        ]


def _trial_key(row):
    trial = row.get("trial")
    return None if trial in (None, "", "n/a") else int(float(trial))


def _pair_values_to_recalls(events):
    """Attach each ITEM_VALUE_RECALL to the recall it follows.

    Returns a list, parallel to the trial's REC_WORD rows, of the value the
    subject gave for each one (NaN where they gave none).
    """
    values = []
    current = None
    started = False

    for row in events:
        kind = row["trial_type"]
        if kind in (_RECALL, _VOCALIZATION):
            if started:
                values.append(current)
            # A vocalization is not a recall, but it does reset which recall a
            # following value belongs to.
            started = kind == _RECALL
            current = np.nan
        elif kind == _ITEM_VALUE and started:
            current = _number(row.get("itemvalueguess"))

    if started:
        values.append(current)
    return values


def load_raw(root, include_session0=False, modified=True):
    """Load the raw event log into the same structure as :func:`load_data`.

    Parameters
    ----------
    root : str or Path
        Directory holding the ``sub-LTP*/`` folders.
    include_session0 : bool
        Session 0 is a shorter practice session. Excluding it (the default)
        reproduces the ``xs0`` convention used everywhere else here, and drops
        any subject who has no other session.
    modified : bool
        Read the hand-curated files (default) or the ``unmod-`` backups.

    Returns
    -------
    Data
    """
    files = _session_files(root, modified=modified)
    if not files:
        raise FileNotFoundError(f"no vcfrop session TSVs under {root}")
    if not include_session0:
        files = [f for f in files if f[1] != 0]

    # One wordpool over everything, so item numbers mean the same thing in
    # every trial.
    vocabulary = set()
    per_session = []
    for subject, session, path in files:
        events = _read_events(path)
        vocabulary.update(
            row["item"] for row in events if row["trial_type"] == _PRESENT
        )
        per_session.append((subject, session, events))

    itemnos = {word: i + 1 for i, word in enumerate(sorted(vocabulary))}

    trials = []
    for subject, session, events in per_session:
        by_trial = {}
        for row in events:
            key = _trial_key(row)
            if key is not None:
                by_trial.setdefault(key, []).append(row)

        for trial in sorted(by_trial):
            built = _build_trial(by_trial[trial], itemnos)
            if built is None:
                continue
            # The TSV numbers trials from 0; data_083026_xs0.mat numbers them
            # from 1. Match the .mat so the two sources are interchangeable.
            built.update(subject=subject, session=session, trial=trial + 1)
            trials.append(built)

    if not trials:
        raise ValueError(f"no usable trials under {root}")
    return _assemble(trials, itemnos)


def _build_trial(events, itemnos):
    presented = [r for r in events if r["trial_type"] == _PRESENT]
    if not presented:
        return None
    presented.sort(key=lambda r: int(float(r["serialpos"])))

    list_type = presented[0]["storepointtype"].strip().lower()
    if list_type not in LIST_TYPE_CODES:
        raise ValueError(f"unknown storepointtype {presented[0]['storepointtype']!r}")

    recalls_rows = [r for r in events if r["trial_type"] == _RECALL]
    guesses = _pair_values_to_recalls(events)
    if len(guesses) != len(recalls_rows):
        raise ValueError(
            f"paired {len(guesses)} values to {len(recalls_rows)} recalls"
        )

    list_value = next((r for r in events if r["trial_type"] == _LIST_VALUE), None)

    pres_values = [_number(r["itemvaluecorrect"]) for r in presented]
    recalls, intrusions, rec_itemnos, rec_act = [], [], [], []
    for row in recalls_rows:
        intrusion = _number(row.get("intrusion"), default=0.0)
        serialpos = _number(row.get("serialpos"))
        correct = intrusion == 0 and not np.isnan(serialpos)

        # Intrusions are marked -1 in `recalls`; the `intrusions` matrix keeps
        # the original code (>0 prior-list, -1 extra-list).
        recalls.append(serialpos if correct else -1.0)
        intrusions.append(intrusion)
        rec_itemnos.append(
            itemnos.get(row["item"], -1) if correct else -1.0
        )
        rec_act.append(pres_values[int(serialpos) - 1] if correct else np.nan)

    return {
        "pres_itemnos": [itemnos[r["item"]] for r in presented],
        "pres_values": pres_values,
        "pres_types": LIST_TYPE_CODES[list_type],
        "recalls": recalls,
        "intrusions": intrusions,
        "rec_itemnos": rec_itemnos,
        "rec_values_act": rec_act,
        "rec_values_est": guesses,
        "est_values": _number(list_value["avgvalueguess"]) if list_value else np.nan,
        "act_values": _number(list_value["avgvaluecorrect"]) if list_value else np.nan,
    }


def _assemble(trials, itemnos):
    list_length = max(len(t["pres_values"]) for t in trials)
    kept = [t for t in trials if len(t["pres_values"]) == list_length]
    n_rec = max(len(t["recalls"]) for t in kept)
    n = len(kept)

    def column(field):
        return np.array([[t[field]] for t in kept], dtype=np.float64)

    def matrix(field, width, fill):
        out = np.full((n, width), fill, dtype=np.float64)
        for i, t in enumerate(kept):
            values = t[field]
            out[i, : len(values)] = values
        return out

    return Data(
        {
            "list_length": int(list_length),
            "subject": column("subject"),
            "session": column("session"),
            "trial": column("trial"),
            "pres_types": column("pres_types"),
            "est_values": column("est_values"),
            "act_values": column("act_values"),
            "pres_itemnos": matrix("pres_itemnos", list_length, 0.0),
            "pres_values": matrix("pres_values", list_length, np.nan),
            "recalls": matrix("recalls", n_rec, 0.0),
            "intrusions": matrix("intrusions", n_rec, 0.0),
            "rec_itemnos": matrix("rec_itemnos", n_rec, 0.0),
            "rec_values_act": matrix("rec_values_act", n_rec, np.nan),
            "rec_values_est": matrix("rec_values_est", n_rec, np.nan),
        }
    )
