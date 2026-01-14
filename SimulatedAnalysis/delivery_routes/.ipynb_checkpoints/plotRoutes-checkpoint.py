from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional, Callable, Any


def plot_routes_colored(
    routes: List[List[str]],
    store_pos_dict: Dict[str, Tuple[float, float, float]],
    cmap_name: str = "viridis",
    figsize: Tuple[float, float] = (9, 9),
    show_store_names: bool = True,
    show_step_numbers: bool = False,
    alpha_route: float = 0.9,
    linewidth: float = 3.0,
    jitter_scale: float = 4.0,
    jitter_seed: int = 8023453280,
    alpha_jitter_scale: float = 0.25,  # <-- added: how much to vary alpha per route
    alpha_min: float = 0.15,           # <-- added: clamp
    alpha_max: float = 1.0,            # <-- added: clamp
):
    """
    Plot store layout once, then draw multiple routes.

    - Each route gets ONE color (different route => different color).
    - Background store markers and labels shown once (NOT jittered).
    - Route geometry can be slightly offset (jittered) to reduce overlap,
      but route point markers are NOT drawn (so there is only 1 node per store).
    - Opacity (alpha) can be jittered per route to help disambiguate overlaps.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # ---- 1) Plot all stores as background (ONE node per store) ----
    for name, (x, y, z) in store_pos_dict.items():
        ax.scatter(-x, -z, s=80, alpha=1, color="gray")
        if show_store_names:
            ax.text(-x, -z, f" {name}", fontsize=12, alpha=1, ha="left", va="center")

    cmap = cm.get_cmap(cmap_name)

    # ---- 2) Draw each route ----
    for r_idx, route in enumerate(routes):
        if not route or len(route) < 2:
            continue

        xs: List[float] = []
        zs: List[float] = []
        for store in route:
            store_clean = store.strip()
            if store_clean not in store_pos_dict:
                raise KeyError(f"Store '{store_clean}' missing from store_pos_dict!")
            x, y, z = store_pos_dict[store_clean]
            xs.append(-x)
            zs.append(-z)

        n_segments = len(xs) - 1
        if n_segments <= 0:
            continue

        # ---- ONE COLOR PER ROUTE ----
        route_color = cmap(r_idx / max(1, len(routes) - 1))

        # ---- JITTER: constant offset per route (perpendicular to route direction) ----
        rng = np.random.default_rng(jitter_seed + r_idx)
        dx = xs[-1] - xs[0]
        dz = zs[-1] - zs[0]
        norm = np.hypot(dx, dz)

        if norm > 0:
            perp = np.array([-dz, dx]) / norm
        else:
            perp = rng.normal(size=2)
            perp /= np.linalg.norm(perp)

        offset = perp * float(jitter_scale)

        # ---- OPACITY JITTER per route ----
        alpha_route_j = float(
            np.clip(
                alpha_route + rng.uniform(-alpha_jitter_scale, alpha_jitter_scale),
                alpha_min,
                alpha_max,
            )
        )

        # Segments (jittered), but NO per-route node markers
        for i in range(n_segments):
            ax.plot(
                [xs[i] + offset[0], xs[i + 1] + offset[0]],
                [zs[i] + offset[1], zs[i + 1] + offset[1]],
                color=route_color,
                linewidth=linewidth,
                alpha=alpha_route_j,
            )

        # Optional: step labels at true store locations (NOT jittered)
        if show_step_numbers:
            for i, (x, z) in enumerate(zip(xs, zs)):
                ax.text(
                    x,
                    z,
                    f" {r_idx}:{i}",
                    fontsize=16,
                    color=route_color,
                    weight="bold",
                    ha="left",
                    va="bottom",
                    alpha=alpha_route_j,
                )

    # ---- 3) Nice plot defaults ----
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"Routes Colored By Route ({cmap_name})", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal")

    # Expand axis limits based on *all* stores
    all_x = [-pos[0] for pos in store_pos_dict.values()]
    all_z = [-pos[2] for pos in store_pos_dict.values()]
    if all_x and all_z:
        margin = 15
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    plt.tight_layout()
    plt.show()



# -------------------------
# Helpers: compute series + stats
# -------------------------

def _compute_per_route_series(
    routes: List[List[str]],
    series_from_route: Callable[[List[str]], List[float]],
) -> Tuple[List[List[float]], int]:
    """Compute per-route 1D series and return (all_series, max_len)."""
    all_series: List[List[float]] = []
    max_len = 0
    for route in routes:
        vals = series_from_route(route)
        if vals:
            all_series.append(vals)
            max_len = max(max_len, len(vals))
    return all_series, max_len


def _pad_to_matrix(all_series: List[List[float]], max_len: int) -> np.ndarray:
    padded = np.full((len(all_series), max_len), np.nan, dtype=float)
    for i, s in enumerate(all_series):
        padded[i, : len(s)] = s
    return padded


def _per_index_stats(padded: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-index mean/std/skew (skew requires >=2 vals at that index)."""
    mean_vals = np.nanmean(padded, axis=0)
    std_vals = np.nanstd(padded, axis=0)

    max_len = padded.shape[1]
    skew_vals = np.full(max_len, np.nan, dtype=float)
    for t in range(max_len):
        col = padded[:, t]
        vals = col[~np.isnan(col)]
        if len(vals) >= 2:
            mu = vals.mean()
            sigma = vals.std()
            if sigma > 0:
                skew_vals[t] = np.mean(((vals - mu) / sigma) ** 3)
            else:
                skew_vals[t] = 0.0
    return mean_vals, std_vals, skew_vals


def _global_stats(flat_vals: np.ndarray) -> Tuple[float, float, float]:
    if len(flat_vals) >= 2:
        g_mean = float(np.mean(flat_vals))
        g_std = float(np.std(flat_vals))
        if g_std > 0:
            g_skew = float(np.mean(((flat_vals - g_mean) / g_std) ** 3))
        else:
            g_skew = 0.0
        return g_mean, g_std, g_skew
    return float("nan"), float("nan"), float("nan")


# -------------------------
# dist_df-based route series extractor
# -------------------------

def _series_from_route_dist_df(route: List[str], dist_df: pd.DataFrame) -> List[float]:
    """
    Return per-transition distances for a route using labeled dist_df.
    Raises if a transition is missing or non-finite.
    """
    vals: List[float] = []
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        try:
            d = float(dist_df.loc[a, b])
        except KeyError as e:
            raise KeyError(f"Missing label in dist_df for transition ({a} -> {b}).") from e

        if not np.isfinite(d):
            raise ValueError(f"Non-finite distance in dist_df for transition ({a} -> {b}): {d}")

        vals.append(d)
    return vals


# -------------------------
# Core plotter
# -------------------------

def plot_transition_series(
    *,
    # --- Option A: give routes and a function that yields per-transition values ---
    routes: Optional[List[List[str]]] = None,
    series_from_route: Optional[Callable[[List[str]], List[float]]] = None,
    # --- Option B: give algorithm-level mean/std directly ---
    algorithm_means: Optional[List[np.ndarray]] = None,
    algorithm_labels: Optional[List[str]] = None,
    algorithm_stds: Optional[List[Optional[np.ndarray]]] = None,
    # --- Shared plotting options ---
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,  # identity by default
    cmap_name: str = "tab10",
    figsize: Tuple[float, float] = (9, 5),
    title: str = "Per-Transition Series",
    ylabel: str = "Value",
    show_route_lines: bool = True,
    route_alpha: float = 0.25,
    route_linewidth: float = 1.2,
    route_markersize: float = 3.0,
    show_mean: bool = True,
    show_std: bool = True,
    mean_color: str = "black",
) -> Dict[str, Any]:
    """
    One function that can do BOTH:
      (1) Route-level plot with faint per-route lines + mean/std/skew (computed from routes)
      (2) Algorithm-level plot of multiple mean curves (+ optional SD bands)

    Provide either:
      - routes + series_from_route
    and/or:
      - algorithm_means + algorithm_labels (+ optional algorithm_stds)

    Returns a dict with computed route-level stats (if routes provided).
    """
    if routes is None and algorithm_means is None:
        print("Nothing to plot: provide routes or algorithm_means.")
        return {}

    # ---------- ROUTE-LEVEL COMPUTATION ----------
    route_mean = route_std = route_skew = None
    max_len_routes = 0
    all_series = None
    g_mean = g_std = g_skew = None

    if routes is not None:
        if not routes:
            print("No routes provided.")
            return {}

        if series_from_route is None:
            raise ValueError("If routes are provided, series_from_route must be provided.")

        all_series, max_len_routes = _compute_per_route_series(routes, series_from_route)
        if max_len_routes == 0:
            print("All routes are too short to compute transitions.")
            return {}

        padded = _pad_to_matrix(all_series, max_len_routes)
        padded_t = transform(padded)

        route_mean, route_std, route_skew = _per_index_stats(padded_t)
        flat_vals = padded_t[~np.isnan(padded_t)]
        g_mean, g_std, g_skew = _global_stats(flat_vals)

    # ---------- ALGORITHM-LEVEL VALIDATION ----------
    if algorithm_means is not None:
        if algorithm_labels is None or len(algorithm_labels) != len(algorithm_means):
            raise ValueError("algorithm_labels must be provided and match len(algorithm_means).")
        if algorithm_stds is not None and len(algorithm_stds) != len(algorithm_means):
            raise ValueError("algorithm_stds must be None or match len(algorithm_means).")

    # ---------- DETERMINE GLOBAL X-AXIS LENGTH ----------
    lengths: List[int] = []
    if max_len_routes:
        lengths.append(max_len_routes)
    if algorithm_means is not None:
        lengths.extend([len(m) for m in algorithm_means])
    max_len = max(lengths) if lengths else 0
    x_full = np.arange(max_len)

    # ---------- PLOTTING ----------
    cmap = cm.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=figsize)

    # Route-level faint lines
    if routes is not None and show_route_lines and all_series is not None:
        colors = cmap(np.linspace(0, 1, len(all_series)))
        for idx, s in enumerate(all_series):
            s_arr = transform(np.asarray(s, dtype=float))
            x = np.arange(len(s_arr))
            ax.plot(
                x,
                s_arr,
                "-o",
                color=colors[idx],
                linewidth=route_linewidth,
                markersize=route_markersize,
                alpha=route_alpha,
            )

    # Route-level mean + SD band
    handles, legend_labels = [], []
    if routes is not None and show_mean and route_mean is not None:
        mean_pad = np.full(max_len, np.nan)
        std_pad = np.full(max_len, np.nan)
        mean_pad[: len(route_mean)] = route_mean
        std_pad[: len(route_std)] = route_std

        mean_line = ax.plot(
            x_full,
            mean_pad,
            "-o",
            color=mean_color,
            linewidth=3,
            markersize=6,
        )[0]
        handles.append(mean_line)
        legend_labels.append(f"Route-mean (μ={g_mean:.2f}, σ={g_std:.2f}, skew={g_skew:.2f})")

        if show_std:
            upper = mean_pad + std_pad
            lower = mean_pad - std_pad
            sd_patch = ax.fill_between(
                x_full,
                lower,
                upper,
                color=mean_color,
                alpha=0.15,
            )
            handles.append(sd_patch)
            legend_labels.append("Route ±1 SD")

    # Algorithm-level mean curves (+ optional SD bands)
    if algorithm_means is not None:
        alg_colors = cmap(np.linspace(0, 1, len(algorithm_means)))
        for idx, (m, lab) in enumerate(zip(algorithm_means, algorithm_labels)):
            m_t = transform(np.asarray(m, dtype=float))
            x = np.arange(len(m_t))
            ax.plot(
                x,
                m_t,
                "-o",
                color=alg_colors[idx],
                linewidth=2.5,
                markersize=5,
                label=lab,
            )

            if algorithm_stds is not None and algorithm_stds[idx] is not None:
                s = algorithm_stds[idx]
                if len(s) != len(m):
                    raise ValueError(f"algorithm_stds[{idx}] length does not match algorithm_means[{idx}] length")
                s_t = transform(np.asarray(s, dtype=float))
                ax.fill_between(x, m_t - s_t, m_t + s_t, color=alg_colors[idx], alpha=0.15)

    # Cosmetics
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Transition Index")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.5, max_len - 0.5)

    # Legends
    if handles:
        if algorithm_means is not None:
            leg1 = ax.legend(handles, legend_labels, loc="best", fontsize=9)
            ax.add_artist(leg1)
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.legend(handles, legend_labels, loc="best", fontsize=9)
    else:
        if algorithm_means is not None:
            ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()

    return {
        "route_mean": route_mean,
        "route_std": route_std,
        "route_skew": route_skew,
        "global_mean": g_mean,
        "global_std": g_std,
        "global_skew": g_skew,
        "max_len": max_len,
    }


# -------------------------
# Convenience wrappers (NOW USING dist_df)
# -------------------------

def plot_transition_distances(
    routes: List[List[str]],
    dist_df: pd.DataFrame,
    *,
    title: str = "Distance per Transition Across Routes",
    ylabel: str = "Distance",
    **kwargs,
) -> Dict[str, Any]:
    return plot_transition_series(
        routes=routes,
        series_from_route=lambda route: _series_from_route_dist_df(route, dist_df),
        title=title,
        ylabel=ylabel,
        **kwargs,
    )


def plot_transition_times(
    routes: List[List[str]],
    dist_df: pd.DataFrame,
    *,
    player_speed: float = 15.0,
    title: str = "Time per Transition Across Routes",
    ylabel: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if ylabel is None:
        ylabel = f"Time (s) [distance / {player_speed}]"

    return plot_transition_series(
        routes=routes,
        series_from_route=lambda route: [d / player_speed for d in _series_from_route_dist_df(route, dist_df)],
        title=title,
        ylabel=ylabel,
        **kwargs,
    )


def plot_algorithm_distance_means(
    means: List[np.ndarray],
    labels: List[str],
    *,
    stds: Optional[List[Optional[np.ndarray]]] = None,
    title: str = "Mean Distance per Transition by Routing Algorithm",
    ylabel: str = "Distance",
    **kwargs,
) -> Dict[str, Any]:
    return plot_transition_series(
        algorithm_means=means,
        algorithm_labels=labels,
        algorithm_stds=stds,
        title=title,
        ylabel=ylabel,
        **kwargs,
    )


def plot_algorithm_time_means(
    means: List[np.ndarray],
    labels: List[str],
    *,
    stds: Optional[List[Optional[np.ndarray]]] = None,
    player_speed: float = 15.0,
    title: str = "Mean Time per Transition by Routing Algorithm",
    ylabel: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if ylabel is None:
        ylabel = f"Time (s) [distance / {player_speed}]"

    return plot_transition_series(
        algorithm_means=means,
        algorithm_labels=labels,
        algorithm_stds=stds,
        transform=lambda x: x / player_speed,  # means/stds are distances; convert to time
        title=title,
        ylabel=ylabel,
        **kwargs,
    )
