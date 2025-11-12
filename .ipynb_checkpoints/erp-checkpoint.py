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
    """

    # ---------- construction ----------
    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,                  # may be None (to resolve later)
        experiment: Optional[Union[str, Iterable[str]]] = None,    # str, list[str], or None
        dask_args: Optional[dict] = None,
        create_client: bool = True,
    ):
        self.subjects = None if subjects is None else list(subjects)
        # store experiments as a normalized list (or None)
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
        b, a = ERPGenerator._butter_bandpass(band[0], band[1], fs, order)
        return filtfilt(b, a, data, axis=-1, method="gust")

    @staticmethod
    def _to_hilbert_power(data: np.ndarray) -> np.ndarray:
        analytic = hilbert(data, axis=-1)
        return np.abs(analytic) ** 2

    @staticmethod
    def _baseline_z(eeg: xr.DataArray, brange: Tuple[int, int]) -> xr.DataArray:
        base = eeg.sel(time=slice(brange[0], brange[1]))
        mu = base.mean(dim='event').mean(dim='time')
        sd = base.mean(dim='event').std(dim='time', ddof=1)
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
            exps = df[df["subject"].isin(subjects_resolved)]["experiment"].dropna().unique()
            experiments_resolved = sorted(map(str, exps))
        else:
            experiments_resolved = sorted(map(str, experiments))

        return subjects_resolved, experiments_resolved

    # NEW: optional public resolver that also updates fields
    def resolve_from_df(
        self,
        df: pd.DataFrame,
        subjects: Optional[Iterable[str]] = None,
        experiments: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        subs, exps = self._resolve_subjects_experiments(
            df,
            self.subjects if subjects is None else subjects,
            self.experiments if experiments is None else experiments,
        )
        # persist resolved values on the instance
        self.subjects = subs
        self.experiments = exps
        return subs, exps

    # ---------- IO for a session ----------
    def _load_session(
        self,
        df: pd.DataFrame,
        subject: str,
        session: int,
        trange: Tuple[int, int],
        experiment: str,
        clean: bool = True,
        evs_type: Optional[Union[str, List[str], Tuple[str, ...], set]] = 'WORD',
    ) -> Tuple[Optional[xr.DataArray], Optional[pd.DataFrame]]:
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
        freq_band: Optional[Tuple[float, float]] = None,
        band_order: int = 4,
        hilbert_power: bool = False,
        outdir: str = "results",
        fname_prefix: str = "erp_subject_",
        event_average: bool = True,
        experiment: Optional[str] = None,
    ) -> Optional[xr.DataArray]:
        if experiment is None:
            raise ValueError("compute_subject requires a concrete `experiment` string.")

        if exclude_sessions is None:
            exclude_sessions = [24]

        sub_df = df.query('subject == @subject and experiment == @experiment')
        sub_df = sub_df.loc[~sub_df['session'].isin(exclude_sessions)].sort_values('session')

        if sub_df.empty:
            print(f"[skip] {subject} ({experiment}): no sessions")
            return None

        n_samp = int((trange[1] - trange[0]) * sample_rate / 1000) + 1
        time = np.linspace(trange[0], trange[1], n_samp)

        session_arrays: List[xr.DataArray] = []

        for _, row in sub_df.iterrows():
            s = int(row['session'])
            eeg, evs = self._load_session(df, subject, s, trange, experiment=experiment, clean=True, evs_type='WORD')
            if eeg is None or evs is None or len(evs) == 0:
                print(f"[skip] {subject} s{s} ({experiment}): load failure or no events")
                continue

            if evs['eegoffset'].max() < 0:
                print(f"[skip] {subject} s{s} ({experiment}): no aligned EEG")
                continue
            if len(np.unique(evs['session'])) != 1 or int(evs.iloc[0]['session']) != s:
                print(f"[skip] {subject} s{s} ({experiment}): session mismatch")
                continue

            sr_now = int(eeg.samplerate.values)
            if sr_now != sample_rate:
                eeg = eeg.resampled(sample_rate)

            if len(eeg['time']) != len(time):
                print(f"[skip] {subject} s{s} ({experiment}): time misalignment after resample")
                continue
            eeg = eeg.assign_coords(time=time)

            eeg = self._select_channels(eeg, channels)

            if freq_band is not None:
                arr = eeg.values
                arr = self._apply_bandpass(arr, sample_rate, freq_band, order=band_order)
                if hilbert_power:
                    arr = self._to_hilbert_power(arr)
                eeg = xr.DataArray(arr, coords=eeg.coords, dims=eeg.dims, attrs=eeg.attrs)

            eeg = self._baseline_z(eeg, brange)
            if event_average:
                eeg = eeg.mean(dim='event')

            session_arrays.append(eeg)
            print(f"[ok] {subject} s{s} ({experiment}): {eeg.shape}")

        if len(session_arrays) == 0:
            print(f"[skip] {subject} ({experiment}): no valid sessions")
            return None

        stack = xr.concat(session_arrays, dim='session')
        out = stack.mean(dim='session') if event_average else stack.mean(dim=('event', 'session'))

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
        subjects: Optional[Iterable[str]] = None,
        experiments: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        """
        Resolve subjects/experiments if None, persist them on self, then compute.
        Uses Dask if a client exists.
        """
        # resolve from args or from self.* or from df
        subj_basis = subjects if subjects is not None else self.subjects
        exp_basis  = experiments if experiments is not None else self.experiments
        subjects_resolved, experiments_resolved = self._resolve_subjects_experiments(df, subj_basis, exp_basis)

        # NEW: persist the resolved values onto the instance
        self.subjects = subjects_resolved
        self.experiments = experiments_resolved

        if not self.subjects:
            print("[compute_all_subjects] No subjects found.")
            return
        if not self.experiments:
            print("[compute_all_subjects] No experiments found for those subjects.")
            return

        # serial path
        if self.client is None:
            for exp in self.experiments:
                for s in self.subjects:
                    try:
                        local_kwargs = dict(kwargs)
                        if "fname_prefix" not in local_kwargs or local_kwargs["fname_prefix"] == "erp_subject_":
                            local_kwargs["fname_prefix"] = f"erp_{exp}_subject_"
                        self.compute_subject(df=df, subject=s, experiment=exp, **local_kwargs)
                    except Exception as e:
                        print(f"[err] {s} ({exp}): {e}")
            return

        # dask path: submit a batch per experiment
        for exp in self.experiments:
            cfg = {"experiment": exp, **kwargs}
            if "fname_prefix" not in cfg or cfg["fname_prefix"] == "erp_subject_":
                cfg["fname_prefix"] = f"erp_{exp}_subject_"
            futures = self.client.map(_compute_subject_worker, self.subjects, df=df, config=cfg)
            wait(futures)
            try:
                errs = da.get_exceptions(futures, range(len(self.subjects)))
                print(f"[dask] {exp} errors:", errs)
            except Exception:
                print(f"[dask] {exp} completed without reported exceptions")

    # ---------- group load / concat ----------
    def load_group(
        self,
        subjects,
        indir: str = "results",
        fname_prefix: str = "erp_subject_",
        outfile: str = "erp_group.pkl",
    ) -> Optional[xr.DataArray]:
        arrs = []
        keep_subjects = []
        subjects = self.subjects if subjects is None else list(subjects)
        for s in subjects:
            path = os.path.join(indir, f"{fname_prefix}{s}.pkl")
            try:
                d = self.load_dict(path)
                da = d['data']
                da = da.expand_dims(subject=[s])
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
        channel_reduce: str = "mean",
    ) -> None:
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

    @staticmethod
    def plot_matrix(
        data: Union[xr.DataArray, np.ndarray],
        trange: Tuple[int, int],
        *,
        mode: str = "overlay",                 # "overlay" | "subplots"
        row_labels: Optional[Iterable[str]] = None,
        show_aggregate: bool = True,
        aggregate: str = "mean",               # "mean" | "median"
        ci: Optional[Union[float, np.ndarray]] = None,  # scalar or array of shape (time,)
        aggregate_ci: Optional[Union[float, np.ndarray]] = None,
        individual_alpha: float = 0.35,
        linewidth_individual: float = 0.7,
        linewidth_aggregate: float = 1.6,
        figsize: Optional[Tuple[float, float]] = None,
        max_cols: int = 4,                     # for subplots grid
        sharey: bool = True,
        title: Optional[str] = None,
        legend: bool = True,
    ) -> None:
        """
        Plot each row of a 2D matrix (rows x time), either overlayed or as subplots.
        Optionally overlay an aggregate (mean/median) and CI for individuals and/or aggregate.
        """
        # ----- normalize to np.ndarray (rows x time) and capture labels -----
        if isinstance(data, xr.DataArray):
            # Try to find the time dim
            dims = list(data.dims)
            if "time" in dims:
                time_axis = data.sizes["time"]
                # Candidate row dims = everything except "time"
                row_dims = [d for d in dims if d != "time"]
                if len(row_dims) == 0:
                    Y = data.values.reshape(1, time_axis)
                    default_labels = ["row0"]
                elif len(row_dims) == 1:
                    Y = data.values
                    default_labels = [str(v) for v in data.coords[row_dims[0]].values]
                else:
                    # Flatten all non-time dims into rows
                    Y = data.stack(row=row_dims).transpose("row", "time").values
                    default_labels = [str(v) for v in data["row"].values]
            else:
                # assume last axis is time
                arr = data.values
                if arr.ndim == 1:
                    Y = arr.reshape(1, -1)
                    default_labels = ["row0"]
                elif arr.ndim == 2:
                    Y = arr
                    default_labels = [f"row{i}" for i in range(Y.shape[0])]
                else:
                    raise ValueError("xarray input must have a 'time' dim or be (rows x time).")
        else:
            arr = np.asarray(data)
            if arr.ndim == 1:
                Y = arr.reshape(1, -1)
            elif arr.ndim == 2:
                Y = arr
            else:
                raise ValueError("data must be 1D or 2D (rows x time).")
            default_labels = [f"row{i}" for i in range(Y.shape[0])]

        n_rows, T = Y.shape
        t = np.linspace(trange[0], trange[1], T)

        if row_labels is None:
            row_labels = default_labels
        else:
            row_labels = list(row_labels)
            if len(row_labels) != n_rows:
                raise ValueError("row_labels length must match number of rows.")

        # ----- choose figure size -----
        if figsize is None:
            if mode == "overlay":
                figsize = (8.5, 3.5)
            else:
                # scale by number of rows
                ncols = min(max_cols, n_rows)
                nrows = int(np.ceil(n_rows / ncols))
                figsize = (3.2 * ncols, 2.2 * nrows)

        # ----- plotting -----
        if mode == "overlay":
            plt.figure(figsize=figsize)
            ax = plt.gca()
            # plot individuals
            for i in range(n_rows):
                (line_i,) = ax.plot(t, Y[i], linewidth=linewidth_individual, alpha=individual_alpha, label=row_labels[i])
            # aggregate
            if show_aggregate:
                if aggregate == "mean":
                    agg = Y.mean(axis=0)
                elif aggregate == "median":
                    agg = np.median(Y, axis=0)
                else:
                    raise ValueError("aggregate must be 'mean' or 'median'")

                (line_a,) = ax.plot(t, agg, linewidth=linewidth_aggregate, label=f"{aggregate.title()} (all)")
                if aggregate_ci is not None:
                    aci = np.asarray(aggregate_ci)
                    if aci.ndim == 0:
                        aci = np.full_like(agg, float(aggregate_ci))
                    upper, lower = agg + aci, agg - aci
                    ax.fill_between(t, lower, upper,
                                    color=ERPGenerator._darken(line_a.get_color(), 0.8), alpha=0.25)

            ymin, ymax = ax.get_ylim()
            ax.vlines([0], ymin=ymin, ymax=ymax, colors='k')
            ax.hlines([0], xmin=trange[0], xmax=trange[1], linestyles='--', colors='k')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Value")
            if title:
                ax.set_title(title)
            if legend:
                ax.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.show()
            return

        # --- subplots mode ---
        ncols = min(max_cols, n_rows)
        nrows = int(np.ceil(n_rows / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=sharey, sharex=True)
        axes = np.array(axes).reshape(-1)

        for i in range(n_rows):
            ax = axes[i]
            (line_i,) = ax.plot(t, Y[i], linewidth=linewidth_individual, label=row_labels[i])
            # per-row ci (optional)
            if ci is not None:
                ci_row = np.asarray(ci)
                if ci_row.ndim == 0:
                    ci_row = np.full_like(Y[i], float(ci))
                upper, lower = Y[i] + ci_row, Y[i] - ci_row
                ax.fill_between(t, lower, upper, color=ERPGenerator._darken(line_i.get_color(), 0.8), alpha=0.25)
            ymin, ymax = ax.get_ylim()
            ax.vlines([0], ymin=ymin, ymax=ymax, colors='k')
            ax.hlines([0], xmin=trange[0], xmax=trange[1], linestyles='--', colors='k')
            ax.set_title(row_labels[i], fontsize=9)

        # remove any unused axes
        for j in range(n_rows, nrows * ncols):
            fig.delaxes(axes[j])

        fig.text(0.5, 0.02, "Time (ms)", ha="center")
        fig.text(0.02, 0.5, "Value", va="center", rotation="vertical")
        if title:
            fig.suptitle(title, y=1.02)
        fig.tight_layout()
        plt.show()

    def plot_group_subjects(
        self,
        trange: Tuple[int, int],
        *,
        subjects: Optional[Iterable[str]] = None,     # if None, use self.subjects
        indir: str = "results",
        fname_prefix: str = "erp_subject_",           # e.g., "erp_FR1_subject_" for a specific experiment
        reduce_over_channels: Optional[str] = "mean", # "mean" | "median" | None (None means plot channels individually)
        mode: str = "overlay",                        # "overlay" | "subplots"  (per subject)
        show_group_aggregate: bool = True,
        group_aggregate: str = "mean",                # "mean" | "median"
        group_ci: Optional[str] = None,               # None | "sem" | "sd"
        ci_scale: float = 1.0,                        # multiplier for CI (e.g., 1.96 for ~95% if using SEM)
        individual_alpha: float = 0.35,
        linewidth_individual: float = 0.7,
        linewidth_aggregate: float = 1.6,
        title: Optional[str] = None,
        legend: bool = False,
        max_cols: int = 4,
    ) -> None:
        """
        Load per-subject files, reduce channels if requested, and plot either:
          - overlay of all subjects (+ optional group mean/median and CI), or
          - subplots (one panel per subject).

        CI options:
          - group_ci="sem": CI = (std / sqrt(n)) * ci_scale
          - group_ci="sd" : CI =  std           * ci_scale
        """
        # ----- resolve subjects -----
        subj_list = list(subjects) if subjects is not None else (self.subjects or [])
        if not subj_list:
            raise ValueError("No subjects provided and self.subjects is empty. Run compute_all_subjects(...) or pass a list.")

        # ----- load subjects -----
        waves = []     # each element: (label, 1D time series)
        for s in subj_list:
            path = os.path.join(indir, f"{fname_prefix}{s}.pkl")
            try:
                d = self.load_dict(path)
                da = d["data"]  # expected (channel, time) or (time,)
                y = da.values
                if y.ndim == 2:
                    if reduce_over_channels == "mean":
                        y = y.mean(axis=0)
                    elif reduce_over_channels == "median":
                        y = np.median(y, axis=0)
                    elif reduce_over_channels is None:
                        # We'll flatten channels as separate rows with labels 'SUBJ:CHx'
                        for ch_idx in range(y.shape[0]):
                            waves.append((f"{s}:ch{ch_idx}", y[ch_idx]))
                        continue
                    else:
                        raise ValueError("reduce_over_channels must be 'mean', 'median', or None.")
                elif y.ndim == 1:
                    pass
                else:
                    raise ValueError(f"Unexpected data shape for {s} at {path}: {y.shape}")
                waves.append((s, y))
            except FileNotFoundError:
                print(f"[skip] {s}: not found at {path}")
            except Exception as e:
                print(f"[skip] {s}: {e}")

        if not waves:
            print("No valid subject files to plot.")
            return

        # align by min length (simple, robust)
        min_T = min(len(y) for _, y in waves)
        waves = [(lab, y[:min_T]) for lab, y in waves]
        Y = np.stack([y for _, y in waves], axis=0)  # (subjects_or_subject_channels, time)
        labels = [lab for lab, _ in waves]

        # ----- compute group aggregate and CI if requested -----
        agg_curve = None
        agg_ci = None
        if show_group_aggregate:
            if group_aggregate == "mean":
                agg_curve = Y.mean(axis=0)
                spread = Y.std(axis=0, ddof=1)
            elif group_aggregate == "median":
                agg_curve = np.median(Y, axis=0)
                # use MAD approx as spread proxy if median requested (optional)
                spread = None
            else:
                raise ValueError("group_aggregate must be 'mean' or 'median'.")

            if group_ci is not None:
                if group_ci == "sem":
                    n = Y.shape[0]
                    spread = Y.std(axis=0, ddof=1) / np.sqrt(max(n, 1))
                elif group_ci == "sd":
                    spread = Y.std(axis=0, ddof=1)
                else:
                    raise ValueError("group_ci must be 'sem', 'sd', or None.")
                if spread is not None:
                    agg_ci = spread * float(ci_scale)

        # ----- delegate to plot_matrix -----
        ERPGenerator.plot_matrix(
            Y,
            trange,
            mode=mode,
            row_labels=labels,
            show_aggregate=show_group_aggregate,
            aggregate=group_aggregate,
            aggregate_ci=agg_ci,
            individual_alpha=individual_alpha,
            linewidth_individual=linewidth_individual,
            linewidth_aggregate=linewidth_aggregate,
            max_cols=max_cols,
            title=title,
            legend=legend if mode == "overlay" else False,
        )
