import numpy as np
import xarray as xr

# ----------------------------
# USER SETTINGS
# ----------------------------
RTOL = 1e-6
ATOL = 1e-9
CHANNEL_DIM = "channel"
STRIP_EVENT_METADATA_COORDS = True
MAX_MISMATCHES_PER_CHANNEL = 10

PRINT_FULL_RAW_IF_SMALL = True
MAX_ELEMENTS_FOR_FULL_PRINT = 50_000
PRINT_SLICE = True
SLICE_CFG = {"event": 0, "channel": slice(0, 5), "time": slice(0, 20)}


import numpy as np
import xarray as xr
import pyedflib

# ------------------------------------------------------------
# PYEDFLIB LOADER -> xarray with (event, channel, time) IN VOLTS
# ------------------------------------------------------------
def load_bdf_as_xarray(path: str, *, event_dim_name="event") -> xr.DataArray:
    f = pyedflib.EdfReader(path)
    try:
        n_channels = f.signals_in_file
        ch_names = list(f.getSignalLabels())
        sfreq = float(f.getSampleFrequency(0))
        n_samples = int(f.getNSamples()[0])

        # read + convert each channel explicitly to volts
        data_V = []
        for ch in range(n_channels):
            # print(ch)
            x = f.readSignal(ch)
            unit = f.getPhysicalDimension(ch).lower()

            if unit in ("uv", "µv"):
                x = x * 1e-6          # µV → V
            elif unit == "mv":
                x = x * 1e-3          # mV → V
            elif unit == "v":
                pass                 # already volts
            else:
                print(f"Unknown or invalid physical unit '{unit}' \n" + f"for channel {ch} ({ch_names[ch]})")

            data_V.append(x)

        # shape: (channel, time)
        data = np.vstack(data_V)

        times = np.arange(n_samples) / sfreq

        # add singleton event dimension
        data = data[None, :, :]  # (event=1, channel, time)

        da = xr.DataArray(
            data,
            dims=(event_dim_name, "channel", "time"),
            coords={
                event_dim_name: [0],
                "channel": ch_names,
                "time": times,
                "samplerate": sfreq,
            },
            name="eeg",
            attrs={
                "units": "V",
                "source": "pyedflib (explicitly converted to volts)",
            },
        )

        return da

    finally:
        f.close()


# ----------------------------
# HELPERS
# ----------------------------

def _strip_event_metadata(da: xr.DataArray) -> xr.DataArray:
    if "event" not in da.dims:
        return da
    drop = [c for c in da.coords if ("event" in da.coords[c].dims and c != "event")]
    return da.drop_vars(drop) if drop else da


def _summarize_channel_coords(a: xr.DataArray, b: xr.DataArray, channel_dim: str):
    a_ch = list(a[channel_dim].values) if channel_dim in a.dims else []
    b_ch = list(b[channel_dim].values) if channel_dim in b.dims else []

    set_a, set_b = set(a_ch), set(b_ch)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    common = [ch for ch in a_ch if ch in set_b]  # preserve a's order for common

    print("\n--- Channel coord check ---")
    print("BIDS #channels:", len(a_ch))
    print("CML  #channels:", len(b_ch))
    print("Common channels:", len(common))
    if only_a:
        print("Only in BIDS (first 20):", only_a[:20])
    if only_b:
        print("Only in CML  (first 20):", only_b[:20])

    # order check on common channels
    b_common_in_b_order = [ch for ch in b_ch if ch in set_a]
    if common != b_common_in_b_order:
        print("Channel order differs across objects (on common channels).")
    else:
        print("Channel order matches (on common channels).")

    return a_ch, b_ch, common


def _report_mismatches(channel_name, da_a: xr.DataArray, da_b: xr.DataArray, rtol: float, atol: float, max_n: int):
    """
    Prints first max_n mismatch locations for:
      - exact mismatch (NaN-safe)
      - allclose mismatch (rtol/atol, NaN-safe)

    Works for 1D or ND arrays.
    """
    a = np.asarray(da_a.data)
    b = np.asarray(da_b.data)

    # squeeze singleton dims so (1, T) -> (T,)
    a = np.squeeze(a)
    b = np.squeeze(b)

    if a.shape != b.shape:
        print(f"\n[{channel_name}] cannot report mismatches: shape bids {a.shape} vs cml {b.shape}")
        return

    both_nan = np.isnan(a) & np.isnan(b)
    exact_bad = ~((a == b) | both_nan)
    close_bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    def _print_bad(mask, label):
        if not np.any(mask):
            return
        print(f"\n[{channel_name}] {label} mismatches (up to {max_n})")
        inds = np.argwhere(mask)

        for idx_arr in inds[:max_n]:
            idx = tuple(idx_arr.tolist())

            # coords: try to print time (common case)
            coord_info = {}
            if da_a.ndim == 1 and "time" in da_a.dims:
                # idx is (time_index,)
                ti = idx[0]
                try:
                    coord_info["time"] = float(da_a["time"].values[ti])
                except Exception:
                    coord_info["time_index"] = ti
            else:
                # ND case: map each dim
                for d, i in zip(da_a.dims, idx):
                    if d in da_a.coords and da_a.coords[d].ndim == 1:
                        try:
                            coord_info[d] = da_a.coords[d].values[i]
                        except Exception:
                            coord_info[f"{d}_index"] = i
                    else:
                        coord_info[f"{d}_index"] = i

            aval = a[idx]
            bval = b[idx]
            if label == "ALLCLOSE":
                err = np.abs(aval - bval)
                print(" index:", idx, "coords:", coord_info, "bids:", aval, "cml:", bval, "abs_err:", err)
            else:
                print(" index:", idx, "coords:", coord_info, "bids:", aval, "cml:", bval)

    _print_bad(exact_bad, "EXACT")
    _print_bad(close_bad, "ALLCLOSE")


def _print_raw_data(name: str, da: xr.DataArray):
    data = da.data
    n_elem = data.size

    print(f"\n--- {name} raw data ---")
    print("dims:", da.dims)
    print("shape:", da.shape)

    if PRINT_FULL_RAW_IF_SMALL and n_elem <= MAX_ELEMENTS_FOR_FULL_PRINT:
        print(data)
        return

    if PRINT_SLICE:
        sel = {dim: spec for dim, spec in SLICE_CFG.items() if dim in da.dims}
        try:
            sliced = da.isel(**sel)
            print(f"(showing slice {sel})")
            print(sliced.data)
        except Exception as e:
            print(f"(could not slice with {sel}: {e})")
            flat = data.ravel()
            print("(printing first 200 flattened values)")
            print(flat[: min(200, flat.size)])
    else:
        print("(raw data printing disabled; set PRINT_SLICE or PRINT_FULL_RAW_IF_SMALL)")

def _ensure_dims(da: xr.DataArray, *, event_dim="event", channel_dim="channel", time_dim="time") -> xr.DataArray:
    for d in (event_dim, channel_dim, time_dim):
        if d not in da.dims:
            raise ValueError(f"Expected dim '{d}' not found. Have dims={da.dims}")
    return da.transpose(event_dim, channel_dim, time_dim)

def _common_channels(*das: xr.DataArray, channel_dim="channel"):
    sets = [set(map(str, da[channel_dim].values)) for da in das]
    inter = set.intersection(*sets) if sets else set()
    return np.array(sorted(inter), dtype=object)

def _crop_time_to_min(a: np.ndarray, b: np.ndarray):
    """
    Crop two arrays along the last axis to the min length.
    Returns cropped arrays and min_len.
    """
    la = a.shape[-1]
    lb = b.shape[-1]
    m = min(la, lb)
    return a[..., :m], b[..., :m], m

def _compare_pair(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
    rows = []
    exact_fail = []
    close_fail = []

    # Only compare channels that exist in BOTH
    common_ch = np.intersect1d(
        da_a[CHANNEL_DIM].astype(str).values,
        da_b[CHANNEL_DIM].astype(str).values,
    )

    for ch in common_ch:
        da1 = da_a.sel({CHANNEL_DIM: ch})
        da2 = da_b.sel({CHANNEL_DIM: ch})

        a = np.squeeze(np.asarray(da1.data))  # typically (time,)
        b = np.squeeze(np.asarray(da2.data))

        # Handle arbitrary shapes; compare on last axis as time
        try:
            a2, b2, m = _crop_time_to_min(a, b)
        except Exception:
            exact_fail.append(ch)
            close_fail.append(ch)
            rows.append(dict(
                comparison=f"{label_a} vs {label_b}",
                channel=ch,
                n_exact_diff=np.nan,
                n_close_diff=np.nan,
                mean_abs_diff=np.nan,
                max_abs_diff=np.nan,
                mean_signed_diff=np.nan,
                std_diff=np.nan,
                mse_channel=np.nan,
                note="shape/crop failure",
            ))
            continue

        both_nan = np.isnan(a2) & np.isnan(b2)
        exact_bad = ~((a2 == b2) | both_nan)
        close_bad = ~np.isclose(a2, b2, rtol=RTOL, atol=ATOL, equal_nan=True)

        diff = a2 - b2
        invalid = both_nan | np.isnan(a2) | np.isnan(b2)
        diff = np.where(invalid, np.nan, diff)
        abs_diff = np.abs(diff)

        n_exact = int(np.sum(exact_bad))
        n_close = int(np.sum(close_bad))
        mean_abs = float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan
        max_abs  = float(np.nanmax(abs_diff))  if np.isfinite(abs_diff).any() else np.nan
        mean_signed = float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan
        std_diff = float(np.nanstd(diff)) if np.isfinite(diff).any() else np.nan
        mse_channel = float(np.nanmean(diff**2)) if np.isfinite(diff).any() else np.nan

        rows.append(dict(
            comparison=f"{label_a} vs {label_b}",
            channel=ch,
            n_exact_diff=n_exact,
            n_close_diff=n_close,
            mean_abs_diff=mean_abs,
            max_abs_diff=max_abs,
            mean_signed_diff=mean_signed,
            std_diff=std_diff,
            mse_channel=mse_channel,
            time_compared_samples=int(m),
        ))

        if n_exact != 0:
            exact_fail.append(ch)
        if n_close != 0:
            close_fail.append(ch)

        if (n_exact != 0) or (n_close != 0):
            # For reporting: make 1D "time" views (event singleton -> isel(0))
            da1_rep = da1.isel(event=0) if ("event" in da1.dims and da1.sizes.get("event", 0) == 1) else da1
            da2_rep = da2.isel(event=0) if ("event" in da2.dims and da2.sizes.get("event", 0) == 1) else da2

            # Crop the DataArrays for mismatch reporting too (by index)
            # Use isel on time dimension so we don't touch coords.
            if "time" in da1_rep.dims and "time" in da2_rep.dims:
                m2 = min(da1_rep.sizes["time"], da2_rep.sizes["time"])
                da1_rep = da1_rep.isel(time=slice(0, m2))
                da2_rep = da2_rep.isel(time=slice(0, m2))

            _report_mismatches(
                f"{ch} [{label_a} vs {label_b}]",
                da1_rep,
                da2_rep,
                RTOL,
                ATOL,
                MAX_MISMATCHES_PER_CHANNEL
            )

    df = pd.DataFrame(rows)
    return df, exact_fail, close_fail, common_ch

# ----------------------------
# GENERIC N-WAY COMPARISON (NO ALIGN)
# ----------------------------
def compare_eeg_sources(eegs, names, *, channel_dim=CHANNEL_DIM):
    """
    Compare N EEG sources WITHOUT xarray alignment.

    Behavior:
      - Standardizes dim order (event, channel, time)
      - Strips event metadata (optional)
      - Reports channel overlap (pairwise)
      - Pairwise comparisons:
          * channels: intersection by name
          * time: compare by sample index (crop to min length)
    """
    if len(eegs) != len(names):
        raise ValueError(f"len(eegs)={len(eegs)} must match len(names)={len(names)}")
    if len(eegs) < 2:
        raise ValueError("Need at least 2 sources to compare.")

    eegs_std = []
    for da in eegs:
        stripped_da = _strip_event_metadata(da) if STRIP_EVENT_METADATA_COORDS
        eegs_std.append(_ensure_dims(stripped_da))

    # Pre summaries
    print("\n================ CHANNEL SUMMARY (pre-compare; no alignment) ================")
    for i in range(len(eegs_std)):
        for j in range(i + 1, len(eegs_std)):
            _summarize_channel_coords(eegs_std[i], eegs_std[j], channel_dim)

    # Pairwise comparisons
    print("\n================ PAIRWISE COMPARISONS (no alignment) ================")
    stats_frames = []
    summary = {}

    for i in range(len(eegs_std)):
        for j in range(i + 1, len(eegs_std)):
            a_name = names[i]
            b_name = names[j]
            df_pair, exact_fail, close_fail, common_ch = _compare_pair(
                a_name, eegs_std[i], b_name, eegs_std[j]
            )
            stats_frames.append(df_pair)
            summary[(a_name, b_name)] = {
                "common_channels": list(common_ch),
                "exact": list(exact_fail),
                "close": list(close_fail),
            }

            # quick diagnostic: time lengths
            ta = int(eegs_std[i].sizes["time"])
            tb = int(eegs_std[j].sizes["time"])
            print(f"{a_name} vs {b_name}: time lengths {ta} vs {tb} (comparing first {min(ta, tb)} samples)")

    df_stats = pd.concat(stats_frames, axis=0, ignore_index=True) if stats_frames else pd.DataFrame()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n================ PER-CHANNEL STATS (ALL PAIRS; no alignment) ================")
    if len(df_stats) == 0:
        print("[WARN] No stats produced (no common channels across any pairs?)")
    else:
        print(df_stats.to_string(index=False))

    print("\n================ SUMMARY (mismatch channels) ================")
    for (a, b), d in summary.items():
        print(f"{a} vs {b} | #common_channels={len(d['common_channels'])}")
        print(f"{a} vs {b} | exact={d['exact']}")
        print(f"{a} vs {b} | close={d['close']}")

    # Optional raw prints
    for nm, da in zip(names, eegs_std):
        _print_raw_data(nm, da)

    return df_stats, summary
