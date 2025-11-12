# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import pickle
from typing import Dict, List, Tuple, Union, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import cmlreaders as cml
import cmldask.CMLDask as da
from dask.distributed import wait

from scipy.signal import butter, filtfilt, hilbert
from matplotlib.colors import to_rgb


# --- TOP-LEVEL worker (must be at module scope) ---
def _compute_subject_worker(subject, df, config):
    """
    Stateless worker: re-create a tiny ERPGenerator without a client
    and call compute_subject. Avoids pickling the original `self`.
    """
    gen = ERPGenerator(
        subjects=[subject],
        experiment=config["experiment"],
        create_client=False
    )
    kwargs = {k: v for k, v in config.items() if k != "experiment"}
    return gen.compute_subject(df=df, subject=subject, experiment=config["experiment"], **kwargs)


class ERPGenerator:
    """
    ERP/ERBP (band power) workflow focused on selecting electrodes and frequency bands.

    - Choose electrodes by list, regex, or 'all'
    - Optional bandpass filter (freq_band=(f_lo, f_hi) in Hz)
    - Optional Hilbert power (envelope^2) after bandpass
    - Baseline z per channel (using a time window)
    - Average across events and sessions
    - Parallel across subjects (via cmldask/dask)

    Returns per-subject arrays with dims: (channel, time)
    """

    # ---------- construction ----------
    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,                  # now optional
        experiment: Optional[Union[str, Iterable[str]]] = None,    # now optional (str or list[str])
        dask_args: Optional[dict] = None,
        create_client: bool = True,
    ):
        self.subjects = None if subjects is None else list(subjects)

        # normalize experiments to a list or None
        if experiment is None:
            self.experiments = None
        elif isinstance(experiment, (list, tuple, set)):
            self.experiments = [str(e) for e in experiment]
        else:
            self.experiments = [str(experiment)]

        self.client = None
        if create_client:
            if dask_args is None:
                dask_args = {
                    'job_name': 'erp_jobs',
                    'memory_per_job': '8GB',
                    'max_n_jobs': 8,
                    'log_directory': './cluster_logs',
                }
            os.makedirs(dask_args.get('log_directory', './cluster_logs'), exist_ok=True)
            self.client = da.new_dask_client(**dask_args)

    # ---------- utils ----------
    @staticmethod
    def save_dict(d: dict, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved → {path}")

    @staticmethod
    def load_dict(path: str) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _select_channels(eeg: xr.DataArray, channels: Union[str, List[str], re.Pattern]) -> xr.DataArray:
        """
        eeg dims: (event, channel, time) or (channel, time) later.
        channels can be:
          - 'all'
          - list of exact names
          - regex string or compiled pattern
        """
        ch_names = eeg['channel'].values.astype(str)
        if channels == 'all' or channels is None:
            keep = ch_names
        elif isinstance(channels, (list, tuple)):
            keep = [c for c in ch_names if c in set(channels)]
        else:
            pat = re.compile(channels) if isinstance(channels, str) else channels
            keep = [c for c in ch_names if pat.search(c)]
        if len(keep) == 0:
            raise ValueError("No channels matched your selection.")
        return eeg.sel(channel=keep)

    @staticmethod
    def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
        nyq = 0.5 * fs
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return b, a

    @staticmethod
    def _apply_bandpass(data: np.ndarray, fs: float, band: Tuple[float, float], order: int = 4) -> np.ndarray:
        """
        data shape: (..., time)
        """
        b, a = ERPGenerator._butter_bandpass(band[0], band[1], fs, order)
        return filtfilt(b, a, data, axis=-1, method="gust")

    @staticmethod
    def _to_hilbert_power(data: np.ndarray) -> np.ndarray:
        """Instantaneous band power: |hilbert(x)|^2"""
        analytic = hilbert(data, axis=-1)
        return np.abs(analytic) ** 2

    @staticmethod
    def _baseline_z(eeg: xr.DataArray, brange: Tuple[int, int]) -> xr.DataArray:
        """
        z-score using baseline window per channel.
        eeg dims: (event, channel, time)
        """
        base = eeg.sel(time=slice(brange[0], brange[1]))
        mu = base.mean(dim='event').mean(dim='time')         # (channel,)
        sd = base.mean(dim='event').std(dim='time', ddof=1)  # (channel,)
        return (eeg - mu) / sd

    # ---------- discovery helpers ----------
    def _resolve_subjects_experiments(
        self,
        df: pd.DataFrame,
        subjects: Optional[Iterable[str]] = None,
        experiments: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve subjects/experiments to concrete lists using `df`.
        - If subjects is None → all subjects (optionally filtered by experiments)
        - If experiments is None → all experiments for those subjects
        """
        # resolve subjects
        if subjects is None:
            if experiments is not None:
                subs = df[df["experiment"].isin(list(experiments))]["subject"].dropna().unique()
            else:
                subs = df["subject"].dropna().unique()
            subjects_resolved = sorted(map(str, subs))
        else:
            subjects_resolved = sorted(map(str, subjects))

        # resolve experiments
        if experiments is None:
            exps = (
                df[df["subject"].isin(subjects_resolved)]["experiment"]
                .dropna().unique()
            )
            experiments_resolved = sorted(map(str, exps))
        else:
            experiments_resolved = sorted(map(str, experiments))

        return subjects_resolved, experiments_resolved

    # ---------- IO for a session ----------
    def _load_session(
        self,
        df: pd.DataFrame,
        subject: str,
        session: int,
        trange: Tuple[int, int],
        experiment: str,   # <- explicit
        clean: bool = True,
        evs_type: Optional[Union[str, List[str], Tuple[str, ...], set]] = 'WORD',
    ) -> Tuple[Optional[xr.DataArray], Optional[pd.DataFrame]]:
        """
        Return (eeg, events) or (None, None).
        """
        try:
            sel = (df['subject'] == subject) & (df['experiment'] == experiment) & (df['session'] == session)
            df_sel = df.loc[sel]
            if df_sel.empty:
                raise ValueError(f"No matching row for {subject} {experiment} session {session}")

            row = df_sel.iloc[0]
            reader = cml.CMLReader(subject=row['subject'], experiment=row['experiment'], session=row['session'])
            evs = reader.load("events")
            if evs_type is not None:
                if isinstance(evs_type, str):
                    evs = evs[evs['type'] == evs_type]
                else:
                    evs = evs[evs['type'].isin(evs_type)]

            eeg = reader.load_eeg(events=evs, rel_start=trange[0], rel_stop=trange[1], clean=clean).to_ptsa()
            return eeg, evs
        except Exception as e:
            print(f"[load fail] {subject} s{session} ({experiment}): {e}")
            return None, None

    # ---------- core: one subject ----------
    def compute_subject(
        self,
        df: pd.DataFrame,
        subject: str,
        trange: Tuple[int, int],
        brange: Tuple[int, int],
        channels: Union[str, List[str], re.Pattern] = 'all',
        sample_rate: int = 500,
        exclude_sessions: Optional[List[int]] = None,
        freq_band: Optional[Tuple[float, float]] = None,   # e.g., (4, 8)
        band_order: int = 4,
        hilbert_power: bool = False,                       # if True, convert filtered data to power
        outdir: str = "results",
        fname_prefix: str = "erp_subject_",
        event_average: bool = True,                        # average over events before session-avg
        experiment: Optional[str] = None,                  # <- explicit experiment required here
    ) -> Optional[xr.DataArray]:
        """
        Returns channel x time (xarray) averaged across sessions (and events if requested).
        Saves pickle on success.
        """
        if experiment is None:
            raise ValueError("compute_subject requires a concrete `experiment` string.")

        if exclude_sessions is None:
            exclude_sessions = [24]

        # find sessions for this subject+experiment
        sub_df = df.query('subject == @subject and experiment == @experiment')
        sub_df = sub_df.loc[~sub_df['session'].isin(exclude_sessions)].sort_values('session')

        if sub_df.empty:
            print(f"[skip] {subject} ({experiment}): no sessions")
            return None

        # canonical time vector (helps enforce alignment)
        n_samp = int((trange[1] - trange[0]) * sample_rate / 1000) + 1
        time = np.linspace(trange[0], trange[1], n_samp)

        session_arrays: List[xr.DataArray] = []

        for _, row in sub_df.iterrows():
            s = int(row['session'])
            eeg, evs = self._load_session(df, subject, s, trange, experiment=experiment, clean=True, evs_type='WORD')
            if eeg is None or evs is None or len(evs) == 0:
                print(f"[skip] {subject} s{s} ({experiment}): load failure or no events")
                continue

            # alignment sanity
            if evs['eegoffset'].max() < 0:
                print(f"[skip] {subject} s{s} ({experiment}): no aligned EEG")
                continue
            if len(np.unique(evs['session'])) != 1 or int(evs.iloc[0]['session']) != s:
                print(f"[skip] {subject} s{s} ({experiment}): session mismatch")
                continue

            # resample if needed
            sr_now = int(eeg.samplerate.values)
            if sr_now != sample_rate:
                eeg = eeg.resampled(sample_rate)

            # enforce length & time coordinates (keep your existing policy)
            if len(eeg['time']) != len(time):
                print(f"[skip] {subject} s{s} ({experiment}): time misalignment after resample")
                continue
            eeg = eeg.assign_coords(time=time)

            # choose electrodes
            eeg = self._select_channels(eeg, channels)  # (event, channel, time)

            # optional frequency processing
            if freq_band is not None:
                arr = eeg.values  # (event, channel, time)
                arr = self._apply_bandpass(arr, sample_rate, freq_band, order=band_order)
                if hilbert_power:
                    arr = self._to_hilbert_power(arr)
                eeg = xr.DataArray(arr, coords=eeg.coords, dims=eeg.dims, attrs=eeg.attrs)

            # baseline z per channel
            eeg = self._baseline_z(eeg, brange)  # (event, channel, time)

            # average over events -> (channel, time)
            if event_average:
                eeg = eeg.mean(dim='event')

            session_arrays.append(eeg)
            print(f"[ok] {subject} s{s} ({experiment}): {eeg.shape}")

        if len(session_arrays) == 0:
            print(f"[skip] {subject} ({experiment}): no valid sessions")
            return None

        # average over sessions
        stack = xr.concat(session_arrays, dim='session')  # (session, channel, time)
        out = stack.mean(dim='session') if event_average else stack.mean(dim=('event', 'session'))

        # save (experiment-aware default prefix)
        os.makedirs(outdir, exist_ok=True)
        if fname_prefix is None or fname_prefix == "erp_subject_":
            fname_prefix = f"erp_{experiment}_subject_"
        path = os.path.join(outdir, f"{fname_prefix}{subject}.pkl")
        self.save_dict({'data': out}, path)
        return out

    # ---------- run many subjects ----------
    def compute_all_subjects(
        self,
        df: pd.DataFrame,
        subjects: Optional[Iterable[str]] = None,        # allow overrides
        experiments: Optional[Iterable[str]] = None,     # allow overrides
        **kwargs,
    ) -> None:
        """
        Resolve subjects/experiments if None, then compute per subject for each experiment.
        Uses Dask if a client exists.
        """
        # resolve from args or from self.* or from df
        subj_basis = subjects if subjects is not None else self.subjects
        exp_basis  = experiments if experiments is not None else self.experiments
        subjects_resolved, experiments_resolved = self._resolve_subjects_experiments(df, subj_basis, exp_basis)

        if not subjects_resolved:
            print("[compute_all_subjects] No subjects found.")
            return
        if not experiments_resolved:
            print("[compute_all_subjects] No experiments found for those subjects.")
            return

        # serial path
        if self.client is None:
            for exp in experiments_resolved:
                for s in subjects_resolved:
                    try:
                        # ensure experiment-aware default prefix unless user provided one
                        local_kwargs = dict(kwargs)
                        if "fname_prefix" not in local_kwargs or local_kwargs["fname_prefix"] == "erp_subject_":
                            local_kwargs["fname_prefix"] = f"erp_{exp}_subject_"
                        self.compute_subject(df=df, subject=s, experiment=exp, **local_kwargs)
                    except Exception as e:
                        print(f"[err] {s} ({exp}): {e}")
            return

        # dask path: submit a batch per experiment
        for exp in experiments_resolved:
            cfg = {"experiment": exp, **kwargs}
            if "fname_prefix" not in cfg or cfg["fname_prefix"] == "erp_subject_":
                cfg["fname_prefix"] = f"erp_{exp}_subject_"
            futures = self.client.map(_compute_subject_worker, subjects_resolved, df=df, config=cfg)
            wait(futures)
            try:
                errs = da.get_exceptions(futures, range(len(subjects_resolved)))
                print(f"[dask] {exp} errors:", errs)
            except Exception:
                print(f"[dask] {exp} completed without reported exceptions")

    # ---------- group load / concat ----------
    def load_group(
        self,
        subjects: Iterable[str],
        indir: str = "results",               # default aligned with compute_subject outdir
        fname_prefix: str = "erp_subject_",
        outfile: str = "erp_group.pkl",
    ) -> Optional[xr.DataArray]:
        """
        Load per-subject pickles and concat on new 'subject' dim.
        Returns (subject, channel, time) or None.

        Tip: when you ran multi-experiment batches, your files are
        named like erp_<EXP>_subject_<SUBJ>.pkl — pass fname_prefix accordingly.
        """
        arrs = []
        keep_subjects = []
        for s in subjects:
            path = os.path.join(indir, f"{fname_prefix}{s}.pkl")
            try:
                d = self.load_dict(path)
                da = d['data']
                da = da.expand_dims(subject=[s])  # (subject, channel, time)
                arrs.append(da)
                keep_subjects.append(s)
            except FileNotFoundError:
                print(f"[skip] {s}: not found at {path}")
            except Exception as e:
                print(f"[skip] {s}: {e}")

        if not arrs:
            print("no valid subject files to group")
            return None

        grp = xr.concat(arrs, dim='subject')
        self.save_dict({'data': grp}, os.path.join(indir, outfile))
        print(f"✅ saved group with {len(keep_subjects)} subjects → {os.path.join(indir, outfile)}")
        return grp

    # ---------- plotting ----------
    @staticmethod
    def _darken(color, factor=0.8):
        rgb = np.array(to_rgb(color))
        return tuple(rgb * factor)

    @staticmethod
    def plot(
        data: Union[xr.DataArray, np.ndarray],
        trange: Tuple[int, int],
        ci: Optional[Union[float, np.ndarray]] = None,
        label: str = "ERP/ERBP",
        channel_reduce: str = "mean",  # "mean", "median", or None
    ) -> None:
        """
        Plot a single waveform. If data is (channel, time), reduce over channel first.
        """
        if hasattr(data, "values"):
            y = data.values
        else:
            y = np.asarray(data)

        if y.ndim == 2 and channel_reduce:
            if channel_reduce == "mean":
                y = y.mean(axis=0)
            elif channel_reduce == "median":
                y = np.median(y, axis=0)
            else:
                raise ValueError("channel_reduce must be 'mean', 'median', or None")

        time_vals = np.linspace(trange[0], trange[1], y.shape[-1])

        plt.figure(figsize=(7.5, 3))
        (line,) = plt.plot(time_vals, y, linewidth=0.9, label=label)
        if ci is not None:
            c = np.asarray(ci)
            if c.ndim == 0:
                c = np.full_like(y, float(ci))
            upper, lower = y + c, y - c
            plt.fill_between(time_vals, lower, upper, color=ERPGenerator._darken(line.get_color(), 0.8), alpha=0.25)

        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        plt.vlines([0], ymin=ymin, ymax=ymax, colors='k')
        plt.hlines([0], xmin=trange[0], xmax=trange[1], linestyles='--', colors='k')
        plt.xlabel("Time (ms)")
        plt.ylabel("Baseline Z-scored Voltage" + (" (Power)" if label.lower().startswith("power") else ""))
        plt.legend()
        plt.tight_layout()
        plt.show()
