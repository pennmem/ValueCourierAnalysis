"""Value-estimation analogs of the SPC, lag-CRP and PFR curves.

These have no MATLAB counterpart -- they are new analyses, not a port.

At recall the subject gives a value estimate for each item they recall, so
every recalled item carries both an actual value and an estimated one. That
makes it possible to ask the questions the recall curves ask, but about
*estimation accuracy* rather than recall probability:

* by **serial position** -- the SPC analog
* by **lag** between adjacent recalls -- the lag-CRP analog
* by **output position** -- the PFR analog, whose output position 1 is the
  literal PFR case

Values are z-scored within each list so lists are comparable::

    mu, sd   = mean, sd of that list's 15 presented values
    z_act(i) = (actual value of recalled item i    - mu) / sd
    z_est(i) = (subject's estimate of that item    - mu) / sd
    signed   = z_est - z_act        # positive = overestimate
    absolute = |z_est - z_act|

Centring on the list's own mean value puts every list on a common scale
without flattening the bias: a list-wide over- or under-estimate survives the
transform, which is the point.

Aggregation is hierarchical and unweighted at each level -- **lists within a
session, then sessions within a subject, then subjects**. That is not the same
as pooling every observation: a session in which the subject recalled more
items does not thereby count for more. Session columns are ordinals, matching
the convention used by :func:`vcfrop.value.mean_abserr_sess`.
"""

import numpy as np

from .masks import make_clean_recalls_mask2d, unique_subjects

#: The four per-item quantities the curves can be built from.
MEASURES = ("signed", "absolute", "z_act", "z_est")


def item_z_scores(data, inds=None):
    """Per-item z-scored values and estimation errors.

    Parameters
    ----------
    data : Data
        As returned by :func:`vcfrop.matio.load_data`.
    inds : bool array, optional
        Row mask selecting a condition. Defaults to all rows.

    Returns
    -------
    dict
        ``z_act``, ``z_est``, ``signed``, ``absolute`` -- each
        ``(n_trials, n_output_positions)`` with NaN outside scored items --
        plus ``valid``, the boolean mask of items that contribute (a clean,
        non-repeated recall of a studied item that the subject also estimated).
    """
    if inds is None:
        inds = np.ones(data.subject.shape[0], dtype=bool)

    recalls = data.recalls[inds]
    pres_values = data.pres_values[inds]
    act = data.rec_values_act[inds]
    est = data.rec_values_est[inds]

    rec_mask = make_clean_recalls_mask2d(recalls)
    valid = rec_mask & ~np.isnan(act) & ~np.isnan(est)

    # One centre and scale per list, from that list's presented values.
    mu = pres_values.mean(axis=1, keepdims=True)
    sd = pres_values.std(axis=1, ddof=1, keepdims=True)
    if np.any(sd == 0):
        raise ValueError(
            "a list has zero variance in its presented values; z-scoring it "
            "would divide by zero"
        )

    z_act = np.where(valid, (act - mu) / sd, np.nan)
    z_est = np.where(valid, (est - mu) / sd, np.nan)
    signed = z_est - z_act

    return {
        "z_act": z_act,
        "z_est": z_est,
        "signed": signed,
        "absolute": np.abs(signed),
        "valid": valid,
    }


# ------------------------------------------------------------- aggregation


def _per_trial_bins(trial_idx, bin_idx, values, n_trials, n_bins):
    """Mean of ``values`` per (trial, bin); NaN for empty cells.

    Several observations can share a bin within one trial -- two transitions
    at the same lag, say -- so they are averaged before the trial takes part
    in the hierarchy.
    """
    total = np.zeros((n_trials, n_bins))
    count = np.zeros((n_trials, n_bins))
    if len(trial_idx):
        np.add.at(total, (trial_idx, bin_idx), values)
        np.add.at(count, (trial_idx, bin_idx), 1.0)

    out = np.full((n_trials, n_bins), np.nan)
    nz = count > 0
    out[nz] = total[nz] / count[nz]
    return out


def _aggregate(per_trial, subjects, sessions):
    """Average lists -> sessions -> subjects, unweighted at each level."""
    subjects = np.asarray(subjects).ravel()
    sessions = np.asarray(sessions).ravel()
    subs = unique_subjects(subjects)

    out = np.full((len(subs), per_trial.shape[1]), np.nan)
    with np.errstate(invalid="ignore"):
        for si, subj in enumerate(subs):
            rows = subjects == subj
            per_session = []
            for sess in np.unique(sessions[rows]):
                cell = per_trial[rows & (sessions == sess)]
                if cell.size:
                    per_session.append(_quiet_nanmean(cell, axis=0))
            if per_session:
                out[si] = _quiet_nanmean(np.array(per_session), axis=0)
    return out


def _quiet_nanmean(arr, axis):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _context(data, inds):
    if inds is None:
        inds = np.ones(data.subject.shape[0], dtype=bool)
    return (
        inds,
        data.recalls[inds],
        data.subject[inds],
        data.session[inds],
        data.list_length,
    )


# ----------------------------------------------------------------- curves


def error_by_serial_position(data, inds=None, measure="signed"):
    """The SPC analog: estimation error against the item's serial position.

    Returns
    -------
    (n_subjects, list_length) array
        Column ``p - 1`` holds serial position ``p``.
    """
    _check_measure(measure)
    inds, recalls, subjects, sessions, ll = _context(data, inds)
    z = item_z_scores(data, inds)
    values = z[measure]

    trials, cols = np.nonzero(z["valid"])
    sps = recalls[trials, cols].astype(int)
    keep = (sps >= 1) & (sps <= ll)

    per_trial = _per_trial_bins(
        trials[keep], sps[keep] - 1, values[trials[keep], cols[keep]],
        recalls.shape[0], ll,
    )
    return _aggregate(per_trial, subjects, sessions)


def error_by_output_position(data, inds=None, measure="signed", n_positions=None):
    """The PFR analog: estimation error against output position.

    Output position 1 is the literal PFR case; later positions show how
    estimation error evolves across the recall sequence.

    Parameters
    ----------
    n_positions : int, optional
        Column count. Defaults to the last output position anywhere in the
        **full** dataset that carries a scored item, so that the three
        conditions share columns. Intrusions can push this past the list
        length (it is 20 here, for a list length of 15).
    """
    _check_measure(measure)
    inds, recalls, subjects, sessions, ll = _context(data, inds)

    if n_positions is None:
        full = item_z_scores(data, None)["valid"]
        n_positions = int(np.flatnonzero(full.any(axis=0)).max()) + 1

    z = item_z_scores(data, inds)
    trials, cols = np.nonzero(z["valid"])
    keep = cols < n_positions

    per_trial = _per_trial_bins(
        trials[keep], cols[keep], z[measure][trials[keep], cols[keep]],
        recalls.shape[0], n_positions,
    )
    return _aggregate(per_trial, subjects, sessions)


def error_by_lag(data, inds=None, measure="signed"):
    """The lag-CRP analog: estimation error against the lag jumped.

    Each *adjacent* pair of scored recalls contributes its lag, and the error
    scored is that of the item transitioned **to**. Adjacency follows the same
    rule as :func:`vcfrop.recall.crp` and ``temp_fact`` -- an intrusion or a
    repeat breaks the pair rather than being skipped over -- so all three
    measures are defined over the same set of transitions.

    Returns
    -------
    (n_subjects, 2 * list_length - 1) array
        Column ``lag + list_length - 1`` holds that lag; lag 0 is NaN.
    """
    _check_measure(measure)
    inds, recalls, subjects, sessions, ll = _context(data, inds)
    z = item_z_scores(data, inds)
    valid, values = z["valid"], z[measure]

    n_bins = 2 * ll - 1
    trial_idx, bin_idx, obs = [], [], []

    for tri in range(recalls.shape[0]):
        row = recalls[tri]
        for col in range(len(row) - 1):
            if not valid[tri, col] or not valid[tri, col + 1]:
                continue
            sp_from, sp_to = int(row[col]), int(row[col + 1])
            if not (1 <= sp_from <= ll and 1 <= sp_to <= ll):
                continue
            trial_idx.append(tri)
            bin_idx.append(sp_to - sp_from + ll - 1)
            obs.append(values[tri, col + 1])

    per_trial = _per_trial_bins(
        np.array(trial_idx, dtype=int), np.array(bin_idx, dtype=int),
        np.array(obs, dtype=float), recalls.shape[0], n_bins,
    )
    out = _aggregate(per_trial, subjects, sessions)
    out[:, ll - 1] = np.nan  # lag 0 is undefined
    return out


def _check_measure(measure):
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")


def observation_counts(data, inds=None, binning="serial_position", n_positions=None):
    """How many scored items land in each bin, pooled over every list.

    The curves themselves are hierarchical means, so a bin fed by two
    observations looks exactly like one fed by two hundred. Report these
    alongside any curve: the output-position tail past the list length is
    fed by a handful of items and should not be read as signal.

    Returns
    -------
    (n_bins,) integer array
    """
    inds, recalls, _, _, ll = _context(data, inds)
    z = item_z_scores(data, inds)
    trials, cols = np.nonzero(z["valid"])

    if binning == "serial_position":
        sps = recalls[trials, cols].astype(int)
        sps = sps[(sps >= 1) & (sps <= ll)]
        return np.bincount(sps - 1, minlength=ll)

    if binning == "output_position":
        if n_positions is None:
            full = item_z_scores(data, None)["valid"]
            n_positions = int(np.flatnonzero(full.any(axis=0)).max()) + 1
        return np.bincount(cols[cols < n_positions], minlength=n_positions)

    if binning == "lag":
        valid = z["valid"]
        lags = []
        for tri in range(recalls.shape[0]):
            row = recalls[tri]
            for col in range(len(row) - 1):
                if not valid[tri, col] or not valid[tri, col + 1]:
                    continue
                a, b = int(row[col]), int(row[col + 1])
                if 1 <= a <= ll and 1 <= b <= ll:
                    lags.append(b - a + ll - 1)
        return np.bincount(np.array(lags, dtype=int), minlength=2 * ll - 1)

    raise ValueError(
        "binning must be 'serial_position', 'output_position' or 'lag'; "
        f"got {binning!r}"
    )


def value_curves(data, inds=None, measures=("signed", "absolute", "z_act", "z_est")):
    """Every curve for one condition, as ``{binning: {measure: array}}``."""
    return {
        "serial_position": {
            m: error_by_serial_position(data, inds, m) for m in measures
        },
        "output_position": {
            m: error_by_output_position(data, inds, m) for m in measures
        },
        "lag": {m: error_by_lag(data, inds, m) for m in measures},
    }
