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

# --- TOP-LEVEL worker (put at module scope, not inside the class) ---
def _compute_subject_worker(subject, df, config):
    """
    Stateless worker: re-create a tiny ERPGenerator without a client
    and call compute_subject. Avoids pickling the original `self`.
    """
    gen = ERPGenerator([subject], experiment=config["experiment"], create_client=False)
    # fan out kwargs except experiment
    kwargs = {k: v for k, v in config.items() if k != "experiment"}
    return gen.compute_subject(df=df, subject=subject, **kwargs)


class ERPGenerator:
    """
    ERP/ERBP (band power) workflow focused on selecting electrodes and frequency bands.

    Key features
    ------------
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
        subjects: Iterable[str],
        experiment: str,
        dask_args: Optional[dict] = None,
        create_client: bool = True,
    ):
        self.subjects = list(subjects)
        self.experiment = experiment
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
        print(ch_names)
        if channels == 'all' or channels is None:
            keep = ch_names
        elif isinstance(channels, (list, tuple)):
            keep = [c for c in ch_names if c in set(channels)]
        else:
            # regex
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
        """
        Instantaneous band power: |hilbert(x)|^2
        """
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

    # ---------- IO for a session ----------
    def _load_session(
        self,
        df: pd.DataFrame,
        subject: str,
        session: int,
        trange: Tuple[int, int],
        clean: bool = True,
        evs_type: Optional[Union[str, List[str], Tuple[str, ...], set]] = 'WORD',
    ) -> Tuple[Optional[xr.DataArray], Optional[pd.DataFrame]]:
        """
        Return (eeg, events) or (None, None).
        """
        try:
            sel = (df['subject'] == subject) & (df['experiment'] == self.experiment) & (df['session'] == session)
            df_sel = df.loc[sel]
            if df_sel.empty:
                raise ValueError(f"No matching row for {subject} {self.experiment} session {session}")

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
            print(f"[load fail] {subject} s{session}: {e}")
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
    ) -> Optional[xr.DataArray]:
        """
        Returns channel x time (xarray) averaged across sessions (and events if requested).
        Saves pickle on success.
        """
        if exclude_sessions is None:
            exclude_sessions = [24]

        # find sessions for this subject in df
        sub_df = df.query('subject == @subject and experiment == @self.experiment')
        sub_df = sub_df.loc[~sub_df['session'].isin(exclude_sessions)].sort_values('session')

        if sub_df.empty:
            print(f"[skip] {subject}: no sessions")
            return None

        # canonical time vector (helps enforce alignment)
        n_samp = int((trange[1] - trange[0]) * sample_rate / 1000) + 1
        time = np.linspace(trange[0], trange[1], n_samp)
        print(len(time))
        # time = eeg['time'] 
        # time = None

        session_arrays: List[xr.DataArray] = []

        for _, row in sub_df.iterrows():
            s = int(row['session'])
            eeg, evs = self._load_session(df, subject, s, trange, clean=True, evs_type='WORD')
            print(len(eeg['time']))
            # time = eeg['time']
            if eeg is None or evs is None or len(evs) == 0:
                print(f"[skip] {subject} s{s}: load failure or no events")
                continue

            # alignment sanity
            if evs['eegoffset'].max() < 0:
                print(f"[skip] {subject} s{s}: no aligned EEG")
                continue
            if len(np.unique(evs['session'])) != 1 or int(evs.iloc[0]['session']) != s:
                print(f"[skip] {subject} s{s}: session mismatch")
                continue

            # resample if needed
            sr_now = int(eeg.samplerate.values)
            if sr_now != sample_rate:
                eeg = eeg.resampled(sample_rate)

            # enforce length & time coordinates
            if len(eeg['time']) != len(time):
                print(f"[skip] {subject} s{s}: time misalignment after resample")
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

            print(f"[ok] {subject} s{s}: {eeg.shape}")

        if len(session_arrays) == 0:
            print(f"[skip] {subject}: no valid sessions")
            return None

        # average over sessions
        stack = xr.concat(session_arrays, dim='session')  # (session, channel, time) or (session, event, channel, time) if event_average=False
        if event_average:
            out = stack.mean(dim='session')               # (channel, time)
        else:
            # event-average across sessions then sessions
            out = stack.mean(dim=('event', 'session'))

        # save
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{fname_prefix}{subject}.pkl")
        self.save_dict({'data': out}, path)
        return out

    # ---------- run many subjects ----------
    # inside ERPGenerator
    def compute_all_subjects(self, df: pd.DataFrame, **kwargs) -> None:
        if self.client is None:
            for s in self.subjects:
                try:
                    self.compute_subject(df=df, subject=s, **kwargs)
                except Exception as e:
                    print(f"[err] {s}: {e}")
            return

        config = {"experiment": self.experiment, **kwargs}
        futures = self.client.map(_compute_subject_worker, self.subjects, df=df, config=config)
        wait(futures)
        try:
            errs = da.get_exceptions(futures, range(len(self.subjects)))
            print(errs)
        except Exception:
            print("completed without reported exceptions")



    # ---------- group load / concat ----------
    def load_group(
        self,
        subjects: Iterable[str],
        indir: str = "Assignment_4",
        fname_prefix: str = "erp_subject_",
        outfile: str = "erp_group.pkl",
    ) -> Optional[xr.DataArray]:
        """
        Load per-subject pickles and concat on new 'subject' dim.
        Returns (subject, channel, time) or None.
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
                print(f"[skip] {s}: not found")
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
