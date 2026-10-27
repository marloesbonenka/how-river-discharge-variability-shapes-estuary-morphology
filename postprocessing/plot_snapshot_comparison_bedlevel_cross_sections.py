# %% Compare cross-section bed profiles and channel-depth metrics
"""
Compare the final morphology of the constant scenario, pm3_n5 and pm5_n1.
Data loading follows the pm/n sensitivity script: model folders are found with
find_variability_model_folders, stitched run paths come from
get_stitched_map_run_paths, and the bed-level cache is read (and, if needed,
created / appended) with load_or_update_map_cache_multi.

For the timestep nearest to TARGET_DATE, each cross-section is sampled on a
regular y-grid (nearest face centre via a KDTree) and two variants are made:

- detrended: bed level minus the scenario's own t=0 centreline reference
             (build_centerline_reference);
- raw:       the actual bed level [m], no detrending.

Percentile metrics (p5, p95) are taken on the raw bed of a sample population chosen
per metric in MASK_MODE:
  'frozen'   samples below CHANNEL_INIT_THRESHOLD at t=0 (original method);
  'union'    below CHANNEL_INIT_THRESHOLD at t=0 OR at the snapshot (adds widening);
  'envelope' below water level(x) + WET_MARGIN_M at t=0 OR at the snapshot, plus dry
             stretches narrower than MAX_BAR_WIDTH_M bridged by wet samples (keeps
             mid-channel bars above the limit).
The same populations are used for the raw and detrended profile lines.

Per-section profile figures (one subplot per scenario) are made for both variants.
The along-estuary summary plots and the CSV use the raw bed level only: the metrics
are bed levels [m] (more negative = deeper), not depths.
"""

# %%
from pathlib import Path
import re
import sys
import warnings

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from functions.F_general import (
    apply_plot_style,
    find_variability_model_folders,
    get_snapshot_matches_by_target_dates,
    get_variability_map,
)
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_map_cache import (
    _get_face_coords,
    cache_tag_from_bbox,
    load_or_update_map_cache_multi,
)
from functions.F_morphological_activity import build_centerline_reference


# %%
# =============================================================================
# SETTINGS
# =============================================================================

DISCHARGE = 500
BASE_DIRECTORY = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
)
MODEL_DIR = BASE_DIRECTORY / "Model_Output" / f"Q{DISCHARGE}"
CACHE_DIR = MODEL_DIR / "cached_data"
TIMED_OUT_DIR = MODEL_DIR / "timed-out"

# Same cache settings as the pm/n sensitivity script
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Scenario ID (leading number of the model folder name) -> label
SCENARIO_IDS = {1: "constant", 3: "pm3_n5", 5: "pm5_n1"}
SCENARIO_ORDER = list(SCENARIO_IDS.values())
REFERENCE_SCENARIO = "pm3_n5"       # delta = COMPARISON - REFERENCE
COMPARISON_SCENARIO = "pm5_n1"
TARGET_DATE = np.datetime64("2031-01-01")

BEDLEVEL_VAR = "mesh2d_mor_bl"

# t=0 centreline reference used for detrending
REFERENCE_TIME_IDX = 0
CENTERLINE_XMIN_M = 20_000.0
CENTERLINE_XMAX_M = 45_000.0
CENTERLINE_Y_M = 7_500.0

# Cross-sections: central 5-km strip, 1 profile per km
CROSS_SECTION_X_KM = np.arange(25, 46)
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y_SAMPLES = 500

# Robust depth = mean of the N deepest samples per section
N_DEEPEST = 3

# Percentile metrics on the estuary envelope (see module docstring)
PERCENTILES = (5, 95)
WATER_LEVEL_KM_M = [(20, 2.1), (35, 2.7), (44, 4.3)]   # (x [km], water level [m]) at Q=500,
                                                       # linearly interpolated; refine if needed
WET_MARGIN_M = 1.0        # wet = bed < water level + margin (includes bars just above water)
MAX_BAR_WIDTH_M = 400.0   # dry stretches narrower than this, between wet samples, are bars (kept)
CHANNEL_INIT_THRESHOLD = 2.2   # 'frozen' / 'union': channel = bed below this [m]
MASK_MODE = {"p5": "union", "p95": "union"}   # per metric: 'frozen' | 'union' | 'envelope'

P_KEYS = [f"p{p}" for p in PERCENTILES]
MASK_TAG = "_".join(f"{k}-{MASK_MODE[k]}" for k in P_KEYS)   # e.g. p5-envelope_p95-union (in file names)


# %%
# =============================================================================
# OUTPUT / PLOT SETTINGS
# =============================================================================

OUTPUT_DIR = (
    MODEL_DIR / "output_plots" / "cross_sections_channel_depth"
)
DPI = 300

# Variant-specific labels / output names. The detrended names are unchanged.
VARIANTS = {
    "detrended": dict(
        ylabel="Detrended bed level [m]",
        profile_dir="detrended_bed_profiles",
        suffix="",
    ),
    "raw": dict(
        ylabel="Bed level [m]",
        profile_dir="raw_bed_profiles",
        suffix="_raw",
    ),
}

STYLE = {
    "constant": dict(color="#7f7f7f", linestyle="--", zorder=2,
                     label="constant (pm1_n0)"),
    "pm3_n5": dict(color=plt.cm.Blues(0.55), linestyle="-", zorder=3,
                   label="pm3_n5"),
    "pm5_n1": dict(color=plt.cm.Blues(0.92), linestyle="-", zorder=3,
                   label="pm5_n1"),
}

# Reference lines on the profile plots: (metric key, linestyle, legend label)
LEVEL_LINES = (
    ("p5", "--", f"p5 ({MASK_MODE['p5']} mask)"),
    ("p95", "-.", f"p95 ({MASK_MODE['p95']} mask)"),
    ("robust", ":", f"mean of {N_DEEPEST} lowest"),
)

apply_plot_style("AGU", font_size=10, title_delta=0, tick_delta=0,
                 **{"grid.linewidth": 0.5})


# %%
# =============================================================================
# FUNCTIONS
# =============================================================================

def estuary_envelope(init_profile, now_profile, x_coords_m, dy_m):
    """
    Boolean mask (n_sections, n_y) of wet samples plus bar-sized dry stretches between them.

    init_profile / now_profile: raw bed at t=0 and at the snapshot, sampled at
    the same points. Wet = bed < water level(x) + WET_MARGIN_M; dy_m = sample spacing.
    """
    wl_km, wl_m = zip(*WATER_LEVEL_KM_M)
    limit = (np.interp(np.asarray(x_coords_m) / 1000.0, wl_km, wl_m) + WET_MARGIN_M)[:, None]
    wet = (init_profile < limit) | (now_profile < limit)          # NaN -> False

    idx = np.broadcast_to(np.arange(wet.shape[1]), wet.shape)
    prev_wet = np.maximum.accumulate(np.where(wet, idx, -10**9), axis=1)
    next_wet = np.minimum.accumulate(np.where(wet, idx, 10**9)[:, ::-1], axis=1)[:, ::-1]
    bridged = (prev_wet >= 0) & (next_wet < 10**9) & ((next_wet - prev_wet) * dy_m <= MAX_BAR_WIDTH_M)
    return wet | bridged


def channel_masks(init_profile, now_profile, x_coords_m, dy_m):
    """All candidate sample populations, each a boolean (n_sections, n_y) mask."""
    low_init = init_profile < CHANNEL_INIT_THRESHOLD              # NaN -> False
    low_now = now_profile < CHANNEL_INIT_THRESHOLD
    return {
        "frozen": low_init,
        "union": low_init | low_now,
        "envelope": estuary_envelope(init_profile, now_profile, x_coords_m, dy_m),
    }


def sample_profiles(ds, snapshot_index, x_coords_m, y_samples_m):
    """
    Sample raw and detrended bed level on a regular y-grid at each target x
    (nearest face centre).

    Returns
    -------
    profiles : dict with keys "raw" and "detrended", each an array of shape
        (n_sections, n_y_samples).
    envelope : boolean array of the same shape (estuary envelope, see docstring).
    """
    face_x, face_y = _get_face_coords(ds)
    face_x = np.asarray(face_x, dtype=float)
    face_y = np.asarray(face_y, dtype=float)
    bed = ds[BEDLEVEL_VAR]
    raw_bed = np.asarray(bed.isel(time=snapshot_index).values, dtype=float)
    init_bed = np.asarray(bed.isel(time=REFERENCE_TIME_IDX).values, dtype=float)

    if not (face_x.shape == face_y.shape == raw_bed.shape and raw_bed.ndim == 1):
        raise ValueError(
            "Expected matching 1-D face x, face y and bed arrays; got "
            f"x={face_x.shape}, y={face_y.shape}, bed={raw_bed.shape}."
        )

    # Each scenario is detrended with its own initial bed.
    reference = build_centerline_reference(
        ds=ds,
        var_name=BEDLEVEL_VAR,
        reference_time_idx=REFERENCE_TIME_IDX,
        xmin=CENTERLINE_XMIN_M,
        xmax=CENTERLINE_XMAX_M,
        centerline_y=CENTERLINE_Y_M,
    )
    detrended_bed = raw_bed - reference

    # One KDTree query for all sections: points shape (n_x, n_y, 2)
    xx, yy = np.meshgrid(x_coords_m, y_samples_m, indexing="ij")
    _, face_idx = cKDTree(np.column_stack((face_x, face_y))).query(
        np.stack((xx, yy), axis=-1)
    )

    raw_profile = raw_bed[face_idx]
    masks = channel_masks(init_bed[face_idx], raw_profile, x_coords_m,
                          np.ptp(Y_RANGE_M) / (N_Y_SAMPLES - 1))

    return {"raw": raw_profile, "detrended": detrended_bed[face_idx]}, masks


def bed_metrics(profiles, masks, n_deepest):
    """
    Bed-level metrics per section (rows of `profiles`), in metres; lower = deeper.

    thalweg: lowest sample; robust: mean of the n_deepest lowest samples;
    pNN: percentile of the samples in the population chosen by MASK_MODE.
    """
    ordered = np.sort(profiles, axis=1)  # ascending, NaNs last
    metrics = {
        "thalweg": np.nanmin(profiles, axis=1),
        "robust": np.nanmean(ordered[:, :n_deepest], axis=1),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN sections
        for p, key in zip(PERCENTILES, P_KEYS):
            in_pop = np.where(masks[MASK_MODE[key]], profiles, np.nan)
            metrics[key] = np.nanpercentile(in_pop, p, axis=1)
    return metrics


def profile_label(scenario):
    """Legend label for the cross-section profile plots."""
    if scenario == "constant":
        return STYLE[scenario]["label"]
    pm, n = map(int, re.fullmatch(r"pm(\d+)_n(\d+)", scenario).groups())
    return rf"$R_{{peak}} = {pm}$, $n_{{peaks}} = {n}$"


def clean_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def plot_section_profiles(km, y_km, profiles, levels, variant, out_dir):
    """One figure per cross-section: one subplot per scenario, shared axes."""
    n = len(SCENARIO_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), sharex=True, sharey=True)

    for ax, name in zip(np.atleast_1d(axes), SCENARIO_ORDER):
        s = STYLE[name]
        ax.plot(y_km, profiles[name], color=s["color"], linewidth=1.4, zorder=3)
        for key, ls, _ in LEVEL_LINES:
            if np.isfinite(levels[name][key]):
                ax.axhline(levels[name][key], color=s["color"], linestyle=ls,
                           linewidth=0.9, alpha=0.85, zorder=1)
        ax.set_title(profile_label(name))
        ax.set_xlabel("Cross-estuary distance [km]")
        clean_axes(ax)

    np.atleast_1d(axes)[0].set_ylabel(VARIANTS[variant]["ylabel"])
    handles = [
        mlines.Line2D([], [], color="0.3", linestyle=ls, linewidth=0.9, label=label)
        for _, ls, label in LEVEL_LINES
    ]
    np.atleast_1d(axes)[-1].legend(handles=handles, loc="best", frameon=False)
    fig.suptitle(rf"$Q_{{mean}} = {DISCHARGE}$ m$^3$ s$^{{-1}}$, $x = {km}$ km")

    save_fig(fig, out_dir / f"{variant}_bed_profile_km{km}_{MASK_TAG}.png")


def plot_bed_comparison(x_km, metrics, path):
    """Along-estuary mean bed level of the deepest samples (top) and the pm5_n1 - pm3_n5 difference."""
    fig, (ax, ax_diff) = plt.subplots(
        2, 1, figsize=(7.0, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    for name in SCENARIO_ORDER:
        s = STYLE[name]
        ax.plot(x_km, metrics[name]["robust"], marker="o", markersize=3.5,
                linewidth=1.25, linestyle=s["linestyle"], color=s["color"],
                label=s["label"], zorder=s["zorder"])

    ax.set_ylabel(f"Bed level [m]\n(mean of {N_DEEPEST} lowest samples)")
    ax.set_title("Channel bed level at final morphological snapshot")
    ax.legend(loc="best", frameon=False)
    clean_axes(ax)

    delta = metrics[COMPARISON_SCENARIO]["robust"] - metrics[REFERENCE_SCENARIO]["robust"]
    ax_diff.axhline(0.0, color="0.25", linewidth=0.75, zorder=0)
    ax_diff.plot(x_km, delta, marker="o", markersize=3.5, linewidth=1.25, color="black")
    for where, scenario in ((delta <= 0, COMPARISON_SCENARIO),
                            (delta > 0, REFERENCE_SCENARIO)):
        ax_diff.fill_between(x_km, 0.0, delta, where=where, interpolate=True,
                             color=STYLE[scenario]["color"], alpha=0.35)

    ax_diff.set_xlabel("Along-estuary distance [km]")
    ax_diff.set_ylabel(f"Δ bed level [m]\n({COMPARISON_SCENARIO} − {REFERENCE_SCENARIO})")
    clean_axes(ax_diff)

    save_fig(fig, path)


def plot_metric(x_km, metrics, kind, title, ylabel, path):
    """Along-estuary single bed-level metric (thalweg, p5, p95) for all scenarios."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for name in SCENARIO_ORDER:
        s = STYLE[name]
        ax.plot(x_km, metrics[name][kind], marker="o", markersize=3.5,
                linewidth=1.25, linestyle=s["linestyle"], color=s["color"],
                label=s["label"], zorder=s["zorder"])

    ax.set_title(title)
    ax.set_xlabel("Along-estuary distance [km]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    clean_axes(ax)

    save_fig(fig, path)


def save_metrics_csv(x_km, metrics, path):
    """Thalweg, robust and percentile bed level per scenario, plus deltas."""
    cols, header = [x_km], ["x_km"]
    kinds = ("thalweg", "robust", *P_KEYS)
    for kind in kinds:
        for name in SCENARIO_ORDER:
            cols.append(metrics[name][kind])
            header.append(f"{name}_{kind}_bed_level_m")
    for kind in kinds:
        cols.append(metrics[COMPARISON_SCENARIO][kind] - metrics[REFERENCE_SCENARIO][kind])
        header.append(f"delta_{kind}_bed_level_{COMPARISON_SCENARIO}_minus_{REFERENCE_SCENARIO}_m")

    np.savetxt(path, np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.6f")
    print(f"Saved: {path.name}")


# %%
# =============================================================================
# LOAD CACHES AND SAMPLE PROFILES
# =============================================================================

x_coords_m = CROSS_SECTION_X_KM * 1_000.0
y_samples_m = np.linspace(*Y_RANGE_M, N_Y_SAMPLES)

profiles = {variant: {} for variant in VARIANTS}  # [variant][scenario] -> (n_x, n_y)
pop_masks = {}   # [scenario][mode] -> (n_x, n_y) boolean sample population
selected_times = {}

VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=MODEL_DIR,
    discharge=DISCHARGE,
    scenarios_to_process=None,
    analyze_noisy=False,
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def scenario_from_folder(folder_name):
    """Scenario label for a model folder such as '03_Qr500_pm3_n5.9600329', else None."""
    match = re.match(r"0*(\d+)_", folder_name)
    return SCENARIO_IDS.get(int(match.group(1))) if match else None


for folder in model_folders:
    name = scenario_from_folder(folder.name)
    if name is None:
        continue
    print(f"\nProcessing {name}: {folder.name}")

    run_paths = get_stitched_map_run_paths(
        base_path=MODEL_DIR,
        folder_name=folder.name,
        timed_out_dir=TIMED_OUT_DIR,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [MODEL_DIR / folder]

    ds = load_or_update_map_cache_multi(
        cache_dir=CACHE_DIR,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=[BEDLEVEL_VAR],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        print(f"  No cached data for {folder.name}, skipping.")
        continue

    try:
        matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
        if not matches:
            print(f"  No timestep found for {folder.name}, skipping.")
            continue
        _, idx, actual_dt = matches[0]
        selected_times[name] = actual_dt
        sampled, pop_masks[name] = sample_profiles(ds, idx, x_coords_m, y_samples_m)
    finally:
        ds.close()

    for variant in VARIANTS:
        profiles[variant][name] = sampled[variant]

    dy_km = np.ptp(Y_RANGE_M) / (N_Y_SAMPLES - 1) / 1_000.0
    print(f"  Selected time: {actual_dt}")
    for mode, mk in pop_masks[name].items():
        print(f"  Population width per section, '{mode}' [km]: {np.round(mk.sum(axis=1) * dy_km, 2)}")

missing = [n for n in SCENARIO_ORDER if n not in profiles["raw"]]
if missing:
    raise RuntimeError(f"No data loaded for scenario(s): {missing}")


# %%
# =============================================================================
# METRICS, PLOTS AND CSV FOR EACH VARIANT
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "csv").mkdir(exist_ok=True)

y_km = y_samples_m / 1_000.0

for variant, cfg in VARIANTS.items():
    metrics = {
        name: bed_metrics(profiles[variant][name], pop_masks[name], N_DEEPEST)
        for name in SCENARIO_ORDER
    }

    profile_dir = OUTPUT_DIR / cfg["profile_dir"]
    profile_dir.mkdir(exist_ok=True)

    for i, km in enumerate(CROSS_SECTION_X_KM):
        plot_section_profiles(
            km=int(km),
            y_km=y_km,
            profiles={n: profiles[variant][n][i] for n in SCENARIO_ORDER},
            levels={n: {k: metrics[n][k][i] for k, _, _ in LEVEL_LINES}
                    for n in SCENARIO_ORDER},
            variant=variant,
            out_dir=profile_dir,
        )

    if variant != "raw":      # summary plots and CSV use the raw bed level only
        continue

    plot_bed_comparison(
        CROSS_SECTION_X_KM, metrics,
        OUTPUT_DIR / f"bed_level_mean_{N_DEEPEST}_lowest_samples{cfg['suffix']}.png",
    )
    plot_metric(
        CROSS_SECTION_X_KM, metrics, "thalweg",
        "Thalweg bed level (lowest sample) at final morphological snapshot",
        "Thalweg bed level [m]",
        OUTPUT_DIR / f"bed_level_single_point_thalweg{cfg['suffix']}.png",
    )
    for key in P_KEYS:
        plot_metric(
            CROSS_SECTION_X_KM, metrics, key,
            f"{key} bed level ({MASK_MODE[key]} mask) at final morphological snapshot",
            f"{key} bed level [m]",
            OUTPUT_DIR / f"bed_level_{key}_{MASK_MODE[key]}{cfg['suffix']}.png",
        )
    save_metrics_csv(
        CROSS_SECTION_X_KM, metrics,
        OUTPUT_DIR / "csv" / f"bed_level_metrics{cfg['suffix']}_{MASK_TAG}.csv",
    )


# %%
print("\n" + "=" * 80)
print("CROSS-SECTION PROFILE AND CHANNEL-DEPTH ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory:\n{OUTPUT_DIR}")
print("\nSelected cached MAP times:")
for name in SCENARIO_ORDER:
    print(f"  {name}: {selected_times[name]}")
print("\nBed-level convention (summary plots, raw bed only):")
print("  Metrics are bed levels [m]; more negative = deeper. Delta < 0 = comparison lower.")
print(f"  Thalweg = lowest sample; robust = mean of the {N_DEEPEST} lowest samples.")
print(f"  Percentile populations: {', '.join(f'{k}={MASK_MODE[k]}' for k in P_KEYS)}.")
print(f"  Positive Δ bed level = {COMPARISON_SCENARIO} higher than {REFERENCE_SCENARIO}.")
print("=" * 80)