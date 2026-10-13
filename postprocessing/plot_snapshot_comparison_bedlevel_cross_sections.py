# %% Compare detrended cross-section bed profiles and channel-depth metrics
"""
Compare the final morphology of pm3_n5 and pm5_n1 using the existing cached
MAP files only.

This script:
- Opens existing cached mesh2d_mor_bl MAP output directly;
- Selects the timestep nearest to 2031-01-01;
- Builds the same t=0 centreline reference profile used for detrending in
  plot_map_variable.py;
- Samples each cross-section on a regular y-grid using a KDTree, rather than
  selecting one irregular mesh-face x-column;
- Plots detrended bed profiles for both scenarios;
- Calculates channel depth from each detrended cross-section profile.

No detailed-hydrodynamic run output is used.
No raw MAP output is loaded.
No cache is generated, updated, or stitched.
No timed-out directory is inspected.
"""

# %%
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


# %%
# =============================================================================
# PROJECT IMPORTS
# =============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style
from functions.F_morphological_activity import build_centerline_reference


# %%
# =============================================================================
# INPUT: EXISTING CACHED MAIN-SCENARIO MAP FILES
# =============================================================================

CACHE_FILES = {
    "pm3_n5": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\cached_data"
        r"\mapoutput_mesh2d_mor_bl_03_Qr500_pm3_n5.9600329_"
        r"bbox_1_1_45000_15000.nc"
    ),
    "pm5_n1": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\cached_data"
        r"\mapoutput_mesh2d_mor_bl_05_Qr500_pm5_n1.9517572_"
        r"bbox_1_1_45000_15000.nc"
    ),
}

SCENARIO_ORDER = ["pm3_n5", "pm5_n1"]
REFERENCE_SCENARIO = "pm3_n5"
COMPARISON_SCENARIO = "pm5_n1"
DISCHARGE = 500
TARGET_DATE = np.datetime64("2031-01-01", "ns")


# %%
# =============================================================================
# MAP / DETRENDING SETTINGS
# =============================================================================

BEDLEVEL_VAR = "mesh2d_mor_bl"
FACE_X_VAR = "mesh2d_face_x"
FACE_Y_VAR = "mesh2d_face_y"

REFERENCE_TIME_IDX = 0
DETREND_LAND_THRESHOLD_M = 6.0

CENTERLINE_XMIN_M = 20_000.0
CENTERLINE_XMAX_M = 45_000.0
CENTERLINE_Y_M = 7_500.0


# %%
# =============================================================================
# CROSS-SECTION AND CHANNEL-DEPTH SETTINGS
# =============================================================================

# Preserve your intended locations, but explicitly use integer-km values.
CROSS_SECTION_X_KM = np.arange(25.0, 46.0, 1.0)
X_COORDS_M = CROSS_SECTION_X_KM * 1_000.0

# This is the same central 5-km strip used by the successful uniform-profile
# script, rather than the full y=0–15 km model width.
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y_SAMPLES = 500

# Use selected representative sections for individual profile figures.
SELECTED_PROFILE_KM = CROSS_SECTION_X_KM.astype(int)

# A value of 1 is the single sampled-point thalweg.
# A value of 3 makes a more robust metric: mean of the three deepest sampled
# profile points at each cross-section.
N_DEEPEST = 3

# This is an optional profile-only mask. It removes elevated land if present.
# Set to None to keep every sampled elevation.
PROFILE_LAND_THRESHOLD_M = 8.0


# %%
# =============================================================================
# OUTPUT / PLOT SETTINGS
# =============================================================================

OUTPUT_DIR = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Output\Q500\output_plots"
    r"\comparison_amplitude_frequency_same_Vsed"
    r"\cross_sections_channel_depth"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_OUTPUT_DIR = OUTPUT_DIR / "detrended_bed_profiles"
PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_OUTPUT_DIR = OUTPUT_DIR / "csv"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

apply_plot_style(
    "AGU",
    font_size=10,
    title_delta=0,
    tick_delta=0,
    **{"grid.linewidth": 0.5},
)

DPI = 300

SCENARIO_COLORS = {
    "pm3_n5": plt.cm.Blues(0.55),
    "pm5_n1": plt.cm.Blues(0.92),
}


# %%
# =============================================================================
# FUNCTIONS
# =============================================================================

def get_face_dimension(ds, bedlevel_var):
    """Return the unique non-time dimension of the MAP bed-level variable."""
    if bedlevel_var not in ds:
        raise KeyError(
            f"'{bedlevel_var}' not found. Available variables:\n"
            f"{list(ds.variables)}"
        )

    dims = ds[bedlevel_var].dims

    if "time" not in dims:
        raise ValueError(
            f"'{bedlevel_var}' has no time dimension; dimensions are {dims}."
        )

    non_time_dims = [dim for dim in dims if dim != "time"]

    if len(non_time_dims) != 1:
        raise ValueError(
            f"Expected one face dimension in '{bedlevel_var}', "
            f"but found dimensions {dims}."
        )

    return non_time_dims[0]


def find_nearest_time_index(ds, target_date):
    """
    Find the cached MAP timestep nearest to target_date.

    Returns
    -------
    time_index : int
    selected_time : numpy.datetime64
    all_times : ndarray
    """
    if "time" not in ds.dims or ds.sizes["time"] == 0:
        raise ValueError("Dataset has no valid time dimension.")

    all_times = np.asarray(ds["time"].values).astype("datetime64[ns]")

    time_index = int(np.argmin(np.abs(all_times - target_date)))

    return time_index, all_times[time_index], all_times


def sample_detrended_profiles(
    ds,
    snapshot_index,
    x_coords_m,
    y_range_m,
    n_y_samples,
):
    """
    Build a t=0 centreline reference, detrend selected-time bed elevation,
    and sample regular y-profiles at each target x using nearest face centres.

    Returns
    -------
    y_samples_m : ndarray
        Uniform cross-section sampling coordinates [m].
    detrended_profiles : dict[str, ndarray]
        Detrended sampled profile for every cross-section.
    raw_profiles : dict[str, ndarray]
        Raw sampled profile for every cross-section.
    """
    required = [BEDLEVEL_VAR, FACE_X_VAR, FACE_Y_VAR]
    missing = [name for name in required if name not in ds]

    if missing:
        raise KeyError(
            f"Required variable(s) missing: {missing}\n"
            f"Available variables:\n{list(ds.variables)}"
        )

    face_dimension = get_face_dimension(ds, BEDLEVEL_VAR)

    # Same t=0 centreline-reference detrending method as your existing script.
    reference_per_face = build_centerline_reference(
        ds=ds,
        var_name=BEDLEVEL_VAR,
        reference_time_idx=REFERENCE_TIME_IDX,
        xmin=CENTERLINE_XMIN_M,
        xmax=CENTERLINE_XMAX_M,
        centerline_y=CENTERLINE_Y_M,
    )

    face_x = np.asarray(ds[FACE_X_VAR].values, dtype=float)
    face_y = np.asarray(ds[FACE_Y_VAR].values, dtype=float)

    raw_bed = np.asarray(
        ds[BEDLEVEL_VAR]
        .isel(time=snapshot_index)
        .transpose(face_dimension)
        .values,
        dtype=float,
    )

    if not (
        face_x.ndim == 1
        and face_y.ndim == 1
        and raw_bed.ndim == 1
    ):
        raise ValueError(
            "Expected 1-D face x, face y, and selected-time bed arrays. "
            f"Shapes are x={face_x.shape}, y={face_y.shape}, "
            f"bed={raw_bed.shape}."
        )

    if not (face_x.size == face_y.size == raw_bed.size):
        raise ValueError(
            "Face-coordinate and bed-level array lengths differ:\n"
            f"x = {face_x.size}, y = {face_y.size}, "
            f"bed = {raw_bed.size}."
        )

    # Apply the current-time land mask before subtraction, exactly as in the
    # map-plotting script and your old cross-section script.
    masked_bed = raw_bed.copy()
    masked_bed[raw_bed > DETREND_LAND_THRESHOLD_M] = np.nan

    detrended_bed = masked_bed - reference_per_face

    face_tree = cKDTree(np.column_stack((face_x, face_y)))

    y_samples_m = np.linspace(
        y_range_m[0],
        y_range_m[1],
        n_y_samples,
    )

    raw_profiles = {}
    detrended_profiles = {}

    for x_coord_m in x_coords_m:
        section_name = f"km{int(round(x_coord_m / 1000.0))}"

        sample_points = np.column_stack(
            (
                np.full(y_samples_m.size, x_coord_m, dtype=float),
                y_samples_m,
            )
        )

        _, face_indices = face_tree.query(sample_points)

        raw_profile = raw_bed[face_indices].astype(float)
        detrended_profile = detrended_bed[face_indices].astype(float)

        # Optional stricter profile mask. This is applied to both profile
        # representations purely for plotting and depth calculation.
        if PROFILE_LAND_THRESHOLD_M is not None:
            land_mask = raw_profile > PROFILE_LAND_THRESHOLD_M
            raw_profile[land_mask] = np.nan
            detrended_profile[land_mask] = np.nan

        raw_profiles[section_name] = raw_profile
        detrended_profiles[section_name] = detrended_profile

    return y_samples_m, detrended_profiles, raw_profiles


def compute_depth_metrics(detrended_profile, n_deepest=3):
    """
    Calculate channel-depth metrics from one detrended cross-section profile.

    Because the profile is detrended relative to the initial centreline bed,
    lower/negative values represent deeper bed elevations relative to that
    local initial longitudinal reference.

    Metrics
    -------
    thalweg_bedlevel:
        Single minimum detrended bed level. More negative = deeper.

    robust_deep_bedlevel:
        Mean of the N_DEEPEST lowest sampled detrended points.
        More negative = deeper.

    thalweg_depth:
        Positive depth equivalent of thalweg_bedlevel:
            -thalweg_bedlevel

    robust_depth:
        Positive depth equivalent of robust_deep_bedlevel:
            -robust_deep_bedlevel

    Notes
    -----
    These are bed-elevation-relative metrics, not instantaneous hydraulic
    water depths. A positive change in thalweg_depth means a lower/deeper bed
    relative to the chosen t=0 centreline reference.
    """
    valid_values = np.asarray(detrended_profile, dtype=float)
    valid_values = valid_values[np.isfinite(valid_values)]

    if valid_values.size == 0:
        return {
            "thalweg_bedlevel": np.nan,
            "robust_deep_bedlevel": np.nan,
            "thalweg_depth": np.nan,
            "robust_depth": np.nan,
            "n_valid_samples": 0,
        }

    n_available = min(int(n_deepest), valid_values.size)

    deepest_values = np.partition(valid_values, n_available - 1)[:n_available]

    thalweg_bedlevel = np.nanmin(valid_values)
    robust_deep_bedlevel = np.nanmean(deepest_values)

    return {
        "thalweg_bedlevel": thalweg_bedlevel,
        "robust_deep_bedlevel": robust_deep_bedlevel,
        "thalweg_depth": -thalweg_bedlevel,
        "robust_depth": -robust_deep_bedlevel,
        "n_valid_samples": int(valid_values.size),
    }


def scenario_pm_value(label):
    """Extract integer pm value from a label such as pm3_n5."""
    match = re.search(r"pm(\d+)", label)
    return int(match.group(1)) if match else None


def scenario_n_value(label):
    """Extract integer n value from a label such as pm3_n5."""
    match = re.search(r"n(\d+)", label)
    return int(match.group(1)) if match else None


def plot_profile_pair(
    section_name,
    y_samples_m,
    detrended_profiles_by_scenario,
    depth_metrics_by_scenario,
):
    """Plot one regular-grid detrended bed-profile comparison."""
    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    for scenario_name in SCENARIO_ORDER:
        profile = detrended_profiles_by_scenario[scenario_name][section_name]
        metrics = depth_metrics_by_scenario[scenario_name][section_name]

        pm_value = scenario_pm_value(scenario_name)
        n_value = scenario_n_value(scenario_name)

        label = (
            rf"$R_{{peak}} = {pm_value}$, "
            rf"$n_{{peaks}} = {n_value}$"
        )

        ax.plot(
            y_samples_m / 1000.0,
            profile,
            color=SCENARIO_COLORS[scenario_name],
            linewidth=1.4,
            label=label,
        )

        # Mark the robust mean-deepest-N bed level.
        if np.isfinite(metrics["robust_deep_bedlevel"]):
            ax.axhline(
                metrics["robust_deep_bedlevel"],
                color=SCENARIO_COLORS[scenario_name],
                linestyle="--",
                linewidth=0.8,
                alpha=0.85,
            )

    km = int(section_name.replace("km", ""))

    ax.set_title(
        rf"$Q_{{mean}} = {DISCHARGE}$ m$^3$ s$^{{-1}}$, "
        rf"$x = {km}$ km"
    )
    ax.set_xlabel("Cross-estuary distance [km]")
    ax.set_ylabel("Detrended bed level [m]")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()

    output_path = PROFILE_OUTPUT_DIR / (
        f"detrended_bed_profile_{section_name}.png"
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path.name}")


# %%
# =============================================================================
# LOAD BOTH CACHE FILES, EXTRACT UNIFORM PROFILES, CALCULATE DEPTHS
# =============================================================================

detrended_profiles = {}
raw_profiles = {}
depth_metrics = {}
selected_times = {}

y_samples_reference_m = None

for scenario_name in SCENARIO_ORDER:
    cache_file = CACHE_FILES[scenario_name]

    print("\n" + "=" * 80)
    print(f"Processing {scenario_name}")
    print(f"Cache: {cache_file}")
    print("=" * 80)

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file does not exist:\n{cache_file}")

    ds = xr.open_dataset(cache_file)

    try:
        snapshot_index, selected_time, all_times = find_nearest_time_index(
            ds=ds,
            target_date=TARGET_DATE,
        )

        print(
            f"Available time range: {all_times[0]} to {all_times[-1]}\n"
            f"Requested time:       {TARGET_DATE}\n"
            f"Selected time:        {selected_time}\n"
            f"Selected index:       {snapshot_index}"
        )

        y_samples_m, scenario_detrended_profiles, scenario_raw_profiles = (
            sample_detrended_profiles(
                ds=ds,
                snapshot_index=snapshot_index,
                x_coords_m=X_COORDS_M,
                y_range_m=Y_RANGE_M,
                n_y_samples=N_Y_SAMPLES,
            )
        )

        if y_samples_reference_m is None:
            y_samples_reference_m = y_samples_m
        elif not np.allclose(y_samples_reference_m, y_samples_m):
            raise ValueError(
                "The regular y-sampling positions differ between scenarios."
            )

        detrended_profiles[scenario_name] = scenario_detrended_profiles
        raw_profiles[scenario_name] = scenario_raw_profiles
        selected_times[scenario_name] = selected_time

        depth_metrics[scenario_name] = {}

        for x_km in CROSS_SECTION_X_KM:
            section_name = f"km{int(round(x_km))}"

            depth_metrics[scenario_name][section_name] = (
                compute_depth_metrics(
                    detrended_profile=(
                        scenario_detrended_profiles[section_name]
                    ),
                    n_deepest=N_DEEPEST,
                )
            )

    finally:
        ds.close()


# %%
# =============================================================================
# PLOT SELECTED DETRENDED BED-PROFILE COMPARISONS
# =============================================================================

for x_km in SELECTED_PROFILE_KM:
    section_name = f"km{int(x_km)}"

    plot_profile_pair(
        section_name=section_name,
        y_samples_m=y_samples_reference_m,
        detrended_profiles_by_scenario=detrended_profiles,
        depth_metrics_by_scenario=depth_metrics,
    )


# %%
# =============================================================================
# BUILD DEPTH ARRAYS
# =============================================================================

section_names = [f"km{int(round(x_km))}" for x_km in CROSS_SECTION_X_KM]

thalweg_depth = {}
robust_depth = {}
thalweg_bedlevel = {}
robust_deep_bedlevel = {}

for scenario_name in SCENARIO_ORDER:
    thalweg_depth[scenario_name] = np.array(
        [
            depth_metrics[scenario_name][section_name]["thalweg_depth"]
            for section_name in section_names
        ]
    )

    robust_depth[scenario_name] = np.array(
        [
            depth_metrics[scenario_name][section_name]["robust_depth"]
            for section_name in section_names
        ]
    )

    thalweg_bedlevel[scenario_name] = np.array(
        [
            depth_metrics[scenario_name][section_name]["thalweg_bedlevel"]
            for section_name in section_names
        ]
    )

    robust_deep_bedlevel[scenario_name] = np.array(
        [
            depth_metrics[scenario_name][section_name]["robust_deep_bedlevel"]
            for section_name in section_names
        ]
    )


delta_thalweg_depth = (
    thalweg_depth[COMPARISON_SCENARIO]
    - thalweg_depth[REFERENCE_SCENARIO]
)

delta_robust_depth = (
    robust_depth[COMPARISON_SCENARIO]
    - robust_depth[REFERENCE_SCENARIO]
)


# %%
# =============================================================================
# PLOT ALONG-ESTUARY CHANNEL-DEPTH COMPARISON
# =============================================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(7.0, 6.2),
    sharex=True,
    gridspec_kw={"height_ratios": [2.0, 1.0]},
)

ax_depth, ax_difference = axes

for scenario_name in SCENARIO_ORDER:
    ax_depth.plot(
        CROSS_SECTION_X_KM,
        robust_depth[scenario_name],
        marker="o",
        markersize=3.5,
        linewidth=1.25,
        color=SCENARIO_COLORS[scenario_name],
        label=scenario_name,
    )

ax_depth.set_ylabel(
    f"Channel depth [m]\n"
    f"(−mean of {N_DEEPEST} deepest samples)"
)
ax_depth.set_title(
    "Detrended channel-depth metric at final morphological snapshot"
)
ax_depth.grid(True, linestyle="--", alpha=0.55)
ax_depth.legend(loc="best", frameon=False)
ax_depth.spines["top"].set_visible(False)
ax_depth.spines["right"].set_visible(False)

ax_difference.axhline(
    0.0,
    color="0.25",
    linewidth=0.75,
    zorder=0,
)

ax_difference.plot(
    CROSS_SECTION_X_KM,
    delta_robust_depth,
    marker="o",
    markersize=3.5,
    linewidth=1.25,
    color="black",
)

ax_difference.fill_between(
    CROSS_SECTION_X_KM,
    0.0,
    delta_robust_depth,
    where=(delta_robust_depth >= 0.0),
    color=SCENARIO_COLORS[COMPARISON_SCENARIO],
    alpha=0.35,
    interpolate=True,
)

ax_difference.fill_between(
    CROSS_SECTION_X_KM,
    0.0,
    delta_robust_depth,
    where=(delta_robust_depth < 0.0),
    color=SCENARIO_COLORS[REFERENCE_SCENARIO],
    alpha=0.35,
    interpolate=True,
)

ax_difference.set_xlabel("Along-estuary distance [km]")
ax_difference.set_ylabel(
    f"Δ depth [m]\n"
    f"({COMPARISON_SCENARIO} − {REFERENCE_SCENARIO})"
)
ax_difference.grid(True, linestyle="--", alpha=0.55)
ax_difference.spines["top"].set_visible(False)
ax_difference.spines["right"].set_visible(False)

fig.tight_layout()

depth_plot_path = OUTPUT_DIR / (
    f"channel_depth_mean_{N_DEEPEST}_deepest_samples.png"
)
fig.savefig(depth_plot_path, dpi=DPI, bbox_inches="tight")
fig.savefig(depth_plot_path.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

print(f"Saved: {depth_plot_path.name}")


# %%
# =============================================================================
# OPTIONAL: STRICT SINGLE-POINT THALWEG DEPTH PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(7.0, 3.8))

for scenario_name in SCENARIO_ORDER:
    ax.plot(
        CROSS_SECTION_X_KM,
        thalweg_depth[scenario_name],
        marker="o",
        markersize=3.5,
        linewidth=1.25,
        color=SCENARIO_COLORS[scenario_name],
        label=scenario_name,
    )

ax.set_title("Single-point thalweg depth at final morphological snapshot")
ax.set_xlabel("Along-estuary distance [km]")
ax.set_ylabel("Thalweg depth [m]\n(−minimum detrended bed level)")
ax.grid(True, linestyle="--", alpha=0.55)
ax.legend(loc="best", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

thalweg_plot_path = OUTPUT_DIR / "channel_depth_single_point_thalweg.png"
fig.savefig(thalweg_plot_path, dpi=DPI, bbox_inches="tight")
fig.savefig(thalweg_plot_path.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

print(f"Saved: {thalweg_plot_path.name}")


# %%
# =============================================================================
# SAVE DEPTH METRICS AS CSV
# =============================================================================

csv_path = DATA_OUTPUT_DIR / "channel_depth_metrics.csv"

header = (
    "x_km,"
    "pm3_n5_thalweg_bedlevel_m,"
    "pm5_n1_thalweg_bedlevel_m,"
    "pm3_n5_robust_deep_bedlevel_m,"
    "pm5_n1_robust_deep_bedlevel_m,"
    "pm3_n5_thalweg_depth_m,"
    "pm5_n1_thalweg_depth_m,"
    "pm3_n5_robust_depth_m,"
    "pm5_n1_robust_depth_m,"
    "delta_thalweg_depth_pm5_n1_minus_pm3_n5_m,"
    "delta_robust_depth_pm5_n1_minus_pm3_n5_m"
)

table = np.column_stack(
    (
        CROSS_SECTION_X_KM,
        thalweg_bedlevel[REFERENCE_SCENARIO],
        thalweg_bedlevel[COMPARISON_SCENARIO],
        robust_deep_bedlevel[REFERENCE_SCENARIO],
        robust_deep_bedlevel[COMPARISON_SCENARIO],
        thalweg_depth[REFERENCE_SCENARIO],
        thalweg_depth[COMPARISON_SCENARIO],
        robust_depth[REFERENCE_SCENARIO],
        robust_depth[COMPARISON_SCENARIO],
        delta_thalweg_depth,
        delta_robust_depth,
    )
)

np.savetxt(
    csv_path,
    table,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.6f",
)

print(f"Saved: {csv_path.name}")


# %%
# =============================================================================
# FINISH
# =============================================================================

print("\n" + "=" * 80)
print("CROSS-SECTION PROFILE AND CHANNEL-DEPTH ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory:\n{OUTPUT_DIR}")
print("\nSelected cached MAP times:")
for scenario_name in SCENARIO_ORDER:
    print(f"  {scenario_name}: {selected_times[scenario_name]}")
print("\nDepth convention:")
print(
    "  Positive channel depth = negative detrended bed elevation multiplied by −1."
)
print(
    f"  Robust depth = −mean of the {N_DEEPEST} deepest sampled "
    "bed-level values."
)
print(
    f"  Positive Δ depth = {COMPARISON_SCENARIO} is deeper than "
    f"{REFERENCE_SCENARIO}."
)
print("=" * 80)