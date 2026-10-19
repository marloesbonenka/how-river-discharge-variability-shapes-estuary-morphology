# %% Compare final bed profiles, final-12-hour water-level envelopes, wet flow area, channel width/count and depth

"""

Compare detailed-hydrodynamic-run cross-sections for mean-flow and peak-flow simulations.
The constant-discharge scenario (pm1_n0) is included as a grey reference.

For every cross-section, this script plots:
- Bed level at the final available output timestep.
- Maximum water level during the final twelve hours.
- Minimum water level during the final twelve hours.

The wet flow area of every cross-section is computed for both the maximum and
the minimum water level (integral of water level minus final bed level over y,
dry points counted as zero depth).

A separate figure per flow condition shows the wet flow area (for the maximum
and minimum water level) against the along-estuary distance.

A second separate figure per flow condition shows the channel width against the
along-estuary distance. A channel is a contiguous wet stretch of a
cross-section (water level above the final bed level), so a cross-section can
hold several channels. For both the maximum and the minimum water level the
figure shows:
- the width of the widest single channel;
- the cumulative width of all channels in the cross-section.

A third figure per flow condition shows the number of separate channels.

Channel definition: a sample is wet when water level - final bed level exceeds
WET_DEPTH_THRESHOLD_M (default 0.1 m), and wet stretches narrower than
MIN_CHANNEL_WIDTH_M are ignored. A threshold above zero stops sills that are
only a few cm under water from merging neighbouring channels.
channel_count_sensitivity() shows how the metrics depend on the threshold.

Relative versions (SUMMARY_PLOT_MODE): every scenario is divided by the
reference (constant) scenario at the same cross-section, which removes the
shared tide- to river-dominated trend so differences between scenarios stand
out (1 = same as constant). The bed-level row is shown as a difference in
metres instead, because bed levels can be negative.

All along-estuary figures use a two-column layout: left column = minimum
water level, right column = maximum water level (shared y-axis per row).

A fourth figure per flow condition shows channel depth against the
along-estuary distance:
- the deepest bed level of the cross-section (thalweg bed level);
- the maximum water depth anywhere in the cross-section, i.e.
  max(water level - bed level) over y, for the maximum and minimum water level;
- the water depth at the thalweg, i.e. water level - bed level at the y-sample
  where the bed is lowest, for the maximum and minimum water level.

Plot output can show both scenarios on one set of axes, one scenario per
subplot, or both output styles. Subplot figures stack one scenario per row
(shared axes), draw the bed as a filled body and the water level as a shaded
min-max band, and are saved beneath each flow condition in a dedicated
``subplots`` directory.

The script follows the same cross-section sampling approach as the successful
cached-MAP cross-section script:

- Use the cached grid's 1-D face-centre x/y coordinates.
- Build a KDTree from all face centres.
- Sample a regular y-grid at every requested x location.
- Extract the nearest face-centre value at every sample point.

Map output is loaded through the project-wide stitched-run and MAP-cache
workflow, which handles partitioned output consistently with the other
post-processing scripts.
"""
# %%

from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# %%
# =============================================================================
# PROJECT IMPORTS
# =============================================================================
# If this script is in postprocessing/, keep this import block.
# Otherwise, remove or adjust it.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_general import apply_plot_style
# %%
# =============================================================================
# INPUT: DETAILED-HYDRODYNAMIC RUN DIRECTORIES
# =============================================================================

SCENARIOS = {
    # Constant-discharge reference (pm1_n0). There is a single detailed-hydro
    # run, used as the reference in both the mean-flow and peak-flow
    # comparisons (the discharge is constant, so mean = peak).
    "constant": Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    "pm3_n5_mean_flow": Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_mean.964903"),
    "pm5_n1_mean_flow": Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_mean.964908"),
    "pm3_n5_peak_flow": Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_peak.964907"),
    "pm5_n1_peak_flow": Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_peak.965221"),
}

COMPARISON_PAIRS = {
    "mean_flow": ["constant", "pm3_n5_mean_flow", "pm5_n1_mean_flow"],
    "peak_flow": ["constant", "pm3_n5_peak_flow", "pm5_n1_peak_flow"],
}

DISCHARGE = 500

# Fail early, with a clear message, if a run directory does not exist
# (e.g. the JOBID placeholder of the constant scenario is not filled in yet).
_missing_runs = [
    f"  {name}: {path}"
    for name, path in SCENARIOS.items()
    if not Path(path).exists()
]
if _missing_runs:
    raise FileNotFoundError(
        "These detailed-hydro run directories do not exist:\n"
        + "\n".join(_missing_runs)
    )

# %%
# =============================================================================
# MAP VARIABLES
# =============================================================================

BEDLEVEL_VAR = "mesh2d_mor_bl"
WATERLEVEL_VAR = "mesh2d_s1"
LOAD_VARS = [WATERLEVEL_VAR, BEDLEVEL_VAR]

# Same spatial extent and cache-update behaviour as the established cached-MAP script.
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# %%
# =============================================================================
# CROSS-SECTION SETTINGS
# =============================================================================
# Same cross-section locations as the successful cached-MAP script, extended
# upstream to 20 km for the along-estuary plots.

CROSS_SECTION_X_KM = np.arange(20.0, 46.0, 1.0)
X_COORDS_M = CROSS_SECTION_X_KM * 1_000.0

# Same central 5-km strip as the successful uniform-profile script.
# The wet flow area, channel widths and depths are computed over this strip only.
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y_SAMPLES = 500

# Optional profile-only mask. Set to None to retain every sampled value.
PROFILE_LAND_THRESHOLD_M = None

# A sample counts as wet (part of a channel) when
# water level - final bed level > this threshold [m].
# With 0.0, a sill that is only a few cm under water still connects two
# channels, which biases the channel count low and the widest channel high.
# Use channel_count_sensitivity() (below) to choose a value on the plateau
# where the channel metrics stop changing. Affects channel width/count only
# (wet flow area and water depths do not use it).
WET_DEPTH_THRESHOLD_M = 0.1

# Wet stretches narrower than this [m] are ignored (tiny puddles).
# 0.0 keeps every channel. With 10 m sample spacing, e.g. 50.0 drops
# channels of fewer than 5 samples.
MIN_CHANNEL_WIDTH_M = 0.0

# Upper y-limit [m] of the stacked cross-section subplots. The ~8 m bank
# elevation otherwise squeezes the channel detail into the lower half of each
# panel. Set to None for automatic limits (shows the full bank).
SUBPLOT_YLIM_TOP_M = None   # e.g. 3.0 to zoom in on the channel

# %%
# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
    r"\detailed-hydro-run\output_plots"
    r"\cross_sections_bed_water_level_envelope\test3_normalized"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
# Plot output options:
#   "combined"  : all scenarios plotted on the same axes (original behaviour)
#   "subplots"  : one scenario per row, saved in <flow_condition>/subplots
#   "both"      : save both styles
# (The along-estuary figures are always saved, whatever the mode.)
PLOT_MODE = "subplots"

# Along-estuary summary figures:
#   "absolute" : metrics in their own units
#   "relative" : every scenario normalised against REFERENCE_SCENARIO at the
#                same cross-section (ratio; difference in m for the bed level),
#                which removes the shared tide- to river-dominated trend
#   "both"     : save both versions (relative ones get a "_relative" suffix)
SUMMARY_PLOT_MODE = "both"
REFERENCE_SCENARIO = "constant"
if SUMMARY_PLOT_MODE not in {"absolute", "relative", "both"}:
    raise ValueError(
        "SUMMARY_PLOT_MODE must be 'absolute', 'relative', or 'both'; "
        f"received {SUMMARY_PLOT_MODE!r}."
    )
SUMMARY_RELATIVE_FLAGS = {
    "absolute": [False], "relative": [True], "both": [False, True],
}[SUMMARY_PLOT_MODE]

if PLOT_MODE not in {"combined", "subplots", "both"}:
    raise ValueError(
        "PLOT_MODE must be 'combined', 'subplots', or 'both'; "
        f"received {PLOT_MODE!r}."
    )


apply_plot_style(
    "AGU",
    font_size=10,
    title_delta=0,
    tick_delta=0,
    **{"grid.linewidth": 0.5},
)

# %%
# =============================================================================
# SCENARIO COLOURS
# =============================================================================

all_pm_vals = sorted([0, 2, 3, 4, 5])
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {
    pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm)
    for i, pm in enumerate(all_pm_vals)
}

all_n_vals = sorted([1, 3, 5])
_n_n = max(len(all_n_vals) - 1, 1)
N_COLOR = {
    n: plt.cm.Greens(0.35 + 0.55 * i / _n_n)
    for i, n in enumerate(all_n_vals)

}

# Constant scenario colour
GREY_CONST = "#7f7f7f"


def is_constant_scenario(label):
    """True for the constant-discharge reference scenario (pm1_n0)."""
    return label.startswith("constant")


def get_scenario_color(label, color_by="pm"):
    """Return a scenario colour based on its pm or n value."""
    if is_constant_scenario(label):
        return GREY_CONST

    pm_match = re.search(r"pm(\d+)", label)
    n_match = re.search(r"n(\d+)", label)

    if color_by == "pm" and pm_match:
        pm_val = int(pm_match.group(1))
        if pm_val in PM_COLOR:
            return PM_COLOR[pm_val]

    elif color_by == "n" and n_match:
        n_val = int(n_match.group(1))
        if n_val in N_COLOR:
            return N_COLOR[n_val]

    return "tab:blue"


def scenario_pm_value(label):
    """Extract integer pm value from a label such as pm3_n5_mean_flow."""
    match = re.search(r"pm(\d+)", label)
    return int(match.group(1)) if match else None


def scenario_n_value(label):
    """Extract integer n value from a label such as pm3_n5_mean_flow."""
    match = re.search(r"n(\d+)", label)
    return int(match.group(1)) if match else None


def scenario_legend_label(label):
    """Create a concise scenario label for plotting."""
    if is_constant_scenario(label):
        return "constant (pm1_n0)"

    pm_value = scenario_pm_value(label)
    n_value = scenario_n_value(label)

    return (
        rf"$R_{{peak}} = {pm_value}$, "
        rf"$n_{{peaks}} = {n_value}$"
    )

# =============================================================================
# FUNCTIONS: CACHED DATASET DIMENSIONS
# =============================================================================


def get_face_dimension(ds, variable_name):
    """Return the unique non-time (face) dimension of a cached MAP variable."""
    if variable_name not in ds:
        raise KeyError(
            f"'{variable_name}' not found. Available variables:\n"
            f"{list(ds.variables)}"
        )

    dims = ds[variable_name].dims
    non_time_dims = [dim for dim in dims if dim != "time"]
    if len(non_time_dims) != 1:
        raise ValueError(
            f"Expected one face dimension in '{variable_name}', "
            f"but found dimensions {dims}."
        )
    return non_time_dims[0]


# %%
# =============================================================================
# FUNCTIONS: FINAL-TWELVE-HOUR TIME SELECTION
# =============================================================================

def get_final_twelve_hour_indices(ds, waterlevel_var):

    """
    Return indices of map-output times in the final twelve hours of the run.
    """
    if "time" not in ds[waterlevel_var].dims:
        raise ValueError(
            f"'{waterlevel_var}' has no time dimension; "
            f"dimensions are {ds[waterlevel_var].dims}."
        )

    times = np.asarray(ds["time"].values).astype("datetime64[ns]")

    if times.size == 0:
        raise ValueError("The map output contains no time steps.")

    final_time = times[-1]
    cutoff_time = final_time - np.timedelta64(12, "h")
    final_indices = np.where(times >= cutoff_time)[0]
    if final_indices.size == 0:
        raise ValueError(
            "No output times were found within the final twelve hours.\n"
            f"Final output time: {final_time}\n"
            f"Cutoff time: {cutoff_time}"
        )

    return final_indices, times[final_indices], final_time



# %%
# =============================================================================
# FUNCTIONS: SAMPLE CROSS-SECTIONS
# ============================================================================

def sample_profiles_at_x(
    ds,
    variable_name,
    x_coords_m,
    y_range_m,
    n_y_samples,
    time_indices=None,
    reduction=None,

):
    """
    Sample a face-based variable on regular y-lines at each target x.
    This uses the same nearest-face-centre approach as the successful
    cached-MAP cross-section script.

    Parameters
    ----------
    ds : xr.Dataset
        D-Flow FM map-output dataset.
    variable_name : str
        Variable to sample, e.g. bed level or water level.
    x_coords_m : ndarray
        Cross-section x coordinates in metres.
    y_range_m : tuple[float, float]
        Minimum and maximum y coordinates of the sampling strip.
    n_y_samples : int
        Number of equally spaced y samples per cross-section.
    time_indices : array-like or None
        Time indices to use. If None, use all available times.
    reduction : str or None
        One of 'last', 'max', 'min', or None.
        - 'last': final selected time.
        - 'max': maximum over selected times.
        - 'min': minimum over selected times.
        - None: no time reduction.


    Returns
    -------
    y_samples_m : ndarray
    profiles : dict[str, ndarray]
        Cross-section profiles, keyed by section name.
    """
    required = [variable_name]
    missing = [name for name in required if name not in ds]

    if missing:
        raise KeyError(
            f"Required variable(s) missing: {missing}\n"
            f"Available variables:\n{list(ds.variables)}"
        )
    face_dimension = get_face_dimension(ds, variable_name)
    face_coordinates = np.asarray(ds.grid.face_coordinates, dtype=float)
    if face_coordinates.ndim != 2 or face_coordinates.shape[1] < 2:
        raise ValueError(
            "Expected ds.grid.face_coordinates with shape (n_faces, 2); "
            f"received {face_coordinates.shape}."
        )
    face_x = face_coordinates[:, 0]
    face_y = face_coordinates[:, 1]
    da = ds[variable_name]
    if "time" in da.dims:
        if time_indices is None:
            da_selected = da
        else:
            da_selected = da.isel(time=time_indices)
    else:
        da_selected = da

    if reduction == "last":
        if "time" in da_selected.dims:
            da_selected = da_selected.isel(time=-1)
    elif reduction == "max":
        if "time" in da_selected.dims:
            da_selected = da_selected.max(dim="time", skipna=True)
    elif reduction == "min":
        if "time" in da_selected.dims:
            da_selected = da_selected.min(dim="time", skipna=True)
    elif reduction is not None:
        raise ValueError("reduction must be one of 'last', 'max', 'min', or None.")

    values = np.asarray(
        da_selected.transpose(face_dimension).values,
        dtype=float,

    )

    if not (
        face_x.ndim == 1
        and face_y.ndim == 1
        and values.ndim == 1
    ):
        raise ValueError(
            "Expected 1-D face x, face y, and reduced variable arrays. "
            f"Shapes are x={face_x.shape}, y={face_y.shape}, "
            f"values={values.shape}."
        )

    if not (face_x.size == face_y.size == values.size):
        raise ValueError(
            "Face-coordinate and variable array lengths differ:\n"
            f"x = {face_x.size}, y = {face_y.size}, "
            f"values = {values.size}."

        )

    face_tree = cKDTree(np.column_stack((face_x, face_y)))

    y_samples_m = np.linspace(
        y_range_m[0],
        y_range_m[1],
        n_y_samples,

    )

    profiles = {}

    for x_coord_m in x_coords_m:
        section_name = f"km{int(round(x_coord_m / 1000.0))}"
        sample_points = np.column_stack(
            (
                np.full(y_samples_m.size, x_coord_m, dtype=float),
                y_samples_m,
            )
        )

        _, face_indices = face_tree.query(sample_points)

        profile = values[face_indices].astype(float)

        if PROFILE_LAND_THRESHOLD_M is not None:
            land_mask = profile > PROFILE_LAND_THRESHOLD_M
            profile[land_mask] = np.nan
        profiles[section_name] = profile
    return y_samples_m, profiles


# %%
# =============================================================================
# FUNCTIONS: WET FLOW AREA
# =============================================================================

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def compute_wetted_area(y_samples_m, bed_level, water_level):
    """
    Wet flow area [m^2] of one cross-section.

    Integrates the water depth (water level minus bed level) over y with the
    trapezoidal rule. Depth is clipped at zero, so dry points (water level at
    or below the bed) and NaN-masked points contribute nothing. Only the
    sampled y-strip is integrated.

    Parameters
    ----------
    y_samples_m : ndarray
        Regular y-sample positions [m].
    bed_level : ndarray
        Bed level at the y samples [m].
    water_level : ndarray
        Water level at the y samples [m].
    """
    depth = (
        np.asarray(water_level, dtype=float)
        - np.asarray(bed_level, dtype=float)
    )
    depth = np.where(np.isfinite(depth), depth, 0.0)
    depth = np.clip(depth, 0.0, None)
    return float(_trapezoid(depth, y_samples_m))


# %%
# =============================================================================
# FUNCTIONS: CHANNEL WIDTH
# =============================================================================

def compute_channel_widths(
    y_samples_m, bed_level, water_level, threshold_m=None, min_width_m=None
):
    """
    Channel widths [m] of one cross-section.

    A sample is wet when water level - bed level > threshold_m
    (NaN samples are dry). A channel is a run of consecutive wet samples, so a
    cross-section can contain several channels. Each wet sample represents one
    y-sample spacing, so a channel of n consecutive wet samples has a width of
    n * spacing. Channels narrower than min_width_m are ignored. Only the
    sampled y-strip is considered.

    Parameters
    ----------
    y_samples_m : ndarray
        Regular y-sample positions [m].
    bed_level : ndarray
        Bed level at the y samples [m].
    water_level : ndarray
        Water level at the y samples [m].
    threshold_m : float or None
        Wet-depth threshold [m]; defaults to WET_DEPTH_THRESHOLD_M.
    min_width_m : float or None
        Minimum channel width [m]; defaults to MIN_CHANNEL_WIDTH_M.

    Returns
    -------
    max_single_width_m : float
        Width of the widest single channel (0 if the section is dry).
    cumulative_width_m : float
        Summed width of all channels (0 if the section is dry).
    n_channels : int
        Number of separate channels.
    """
    if threshold_m is None:
        threshold_m = WET_DEPTH_THRESHOLD_M
    if min_width_m is None:
        min_width_m = MIN_CHANNEL_WIDTH_M

    depth = (
        np.asarray(water_level, dtype=float)
        - np.asarray(bed_level, dtype=float)
    )
    wet = np.isfinite(depth) & (depth > threshold_m)

    if not wet.any():
        return 0.0, 0.0, 0

    dy = float(np.mean(np.diff(y_samples_m)))

    # Start (inclusive) and end (exclusive) index of every run of wet samples.
    padded = np.concatenate(([False], wet, [False])).astype(int)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    widths = (ends - starts) * dy
    widths = widths[widths >= min_width_m]

    if widths.size == 0:
        return 0.0, 0.0, 0

    return float(widths.max()), float(widths.sum()), int(widths.size)


# %%
# =============================================================================
# FUNCTIONS: CHANNEL DEPTH
# =============================================================================

def compute_thalweg_bed_level(bed_level):
    """Deepest (lowest) bed level [m] in the cross-section strip."""
    bed = np.asarray(bed_level, dtype=float)
    if not np.isfinite(bed).any():
        return float("nan")
    return float(np.nanmin(bed))


def compute_max_water_depth(bed_level, water_level):
    """
    Largest water depth [m] anywhere in the cross-section (0 if dry).

    This is max(water level - bed level) over all y samples. It usually
    coincides with the thalweg, but not necessarily if the water level varies
    across the section.
    """
    depth = (
        np.asarray(water_level, dtype=float)
        - np.asarray(bed_level, dtype=float)
    )
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return float(np.clip(depth, 0.0, None).max())


def compute_thalweg_water_depth(bed_level, water_level):
    """
    Water depth [m] at the deepest bed point (thalweg) of the cross-section.

    Uses the water level at the y sample where the bed is lowest
    (0 if that point is dry or the section has no valid bed samples).
    """
    bed = np.asarray(bed_level, dtype=float)
    wl = np.asarray(water_level, dtype=float)

    if not np.isfinite(bed).any():
        return 0.0

    i = int(np.nanargmin(bed))
    if not np.isfinite(wl[i]):
        return 0.0
    return float(max(wl[i] - bed[i], 0.0))


# %%
# =============================================================================
# FUNCTIONS: PLOTTING
# =============================================================================

def _style_axes(ax):
    """Common grid and spine styling."""
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, output_path, description):
    """Save a figure as PNG and PDF, close it, and print a message."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(str(output_path)) >= 259:
        plt.close(fig)
        raise OSError(
            f"Output path is {len(str(output_path))} characters, over the "
            f"Windows limit of 260. Shorten OUTPUT_DIR:\n{output_path}"
        )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {description}: {output_path}")


def _title_text(flow_condition):
    return (
        rf"$Q_{{mean}} = {DISCHARGE}$ m$^3$ s$^{{-1}}$, "
        rf"{flow_condition.replace('_', ' ')}"
    )


def plot_cross_section_comparison(
    flow_condition,
    scenario_names,
    section_name,
    y_samples_m,
    final_bed_profiles,
    maximum_water_levels,
    minimum_water_levels,
):

    """
    Plot final bed level and the final-twelve-hour water-level envelope for one
    cross-section, all scenarios on the same axes.
    """
    km = int(section_name.replace("km", ""))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    for scenario_name in scenario_names:
        color = get_scenario_color(scenario_name, color_by="pm")
        label = scenario_legend_label(scenario_name)

        ax.plot(
            y_samples_m / 1000.0,
            final_bed_profiles[scenario_name][section_name],
            color=color,
            linewidth=1.4,
            linestyle="-",
            label=f"{label} — final bed",
        )

        ax.plot(
            y_samples_m / 1000.0,
            maximum_water_levels[scenario_name][section_name],
            color=color,
            linewidth=1.1,
            linestyle="--",
            label=f"{label} — max water level",
        )

        ax.plot(
            y_samples_m / 1000.0,
            minimum_water_levels[scenario_name][section_name],
            color=color,
            linewidth=1.1,
            linestyle=":",
            label=f"{label} — min water level",
        )

    ax.set_title(_title_text(flow_condition) + rf", $x = {km}$ km")

    ax.set_xlabel("Cross-estuary distance [km]")
    ax.set_ylabel("Elevation [m]")
    _style_axes(ax)
    ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / flow_condition
        / f"cross_section_{section_name}_{flow_condition}.png"
    )
    _save_figure(fig, output_path, "cross-section figure")


def plot_cross_section_subplots(
    flow_condition,
    scenario_names,
    section_name,
    y_samples_m,
    final_bed_profiles,
    maximum_water_levels,
    minimum_water_levels,
):
    """
    One scenario per row (shared x and y axes).

    The bed is drawn as a filled body. Water is drawn only where it is wet
    (depth > WET_DEPTH_THRESHOLD_M): light cyan up to the maximum water level,
    darker cyan up to the minimum water level of the final 12 hours.
    """
    km = int(section_name.replace("km", ""))
    n = len(scenario_names)
    y_km = y_samples_m / 1000.0

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(6.8, 2.3 * n + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    # Common floor so the filled bed bodies line up between panels.
    floor = min(
        np.nanmin(final_bed_profiles[s][section_name]) for s in scenario_names
    ) - 0.5

    for i, (ax, scenario_name) in enumerate(zip(axes, scenario_names)):
        color = get_scenario_color(scenario_name, color_by="pm")
        bed = final_bed_profiles[scenario_name][section_name]
        wl_max = maximum_water_levels[scenario_name][section_name]
        wl_min = minimum_water_levels[scenario_name][section_name]

        ax.fill_between(
            y_km, bed, floor,
            color=color, alpha=0.30, linewidth=0, label="Bed",
        )
        ax.plot(y_km, bed, color=color, linewidth=1.4)

        # Draw water only where it is deep enough to count as wet, so the
        # plot shows exactly what the channel metrics see (no water line
        # running over dry sills and banks).
        wet_max = (wl_max - bed) > WET_DEPTH_THRESHOLD_M
        wet_min = (wl_min - bed) > WET_DEPTH_THRESHOLD_M

        ax.fill_between(
            y_km, bed, wl_max, where=wet_max,
            color="tab:cyan", alpha=0.30, linewidth=0,
            label=f"Wet at max water level (depth > {WET_DEPTH_THRESHOLD_M:g} m)",
        )
        ax.fill_between(
            y_km, bed, wl_min, where=wet_min,
            color="tab:cyan", alpha=0.65, linewidth=0,
            label="Wet at min water level",
        )
        ax.plot(y_km, np.where(wet_max, wl_max, np.nan),
                color="tab:cyan", linewidth=0.8)
        ax.plot(y_km, np.where(wet_min, wl_min, np.nan),
                color="tab:cyan", linewidth=0.8)

        ax.set_ylim(bottom=floor, top=SUBPLOT_YLIM_TOP_M)
        ax.text(
            0.01, 0.95, scenario_legend_label(scenario_name),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="black",
        )
        ax.set_ylabel("Elevation [m]")
        _style_axes(ax)
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    axes[-1].set_xlabel("Cross-estuary distance [km]")
    fig.suptitle(_title_text(flow_condition) + rf", $x = {km}$ km")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = (
        OUTPUT_DIR
        / flow_condition
        / "subplots"
        / f"cross_section_{section_name}_{flow_condition}_subplots.png"
    )
    _save_figure(fig, output_path, "subplot figure")


LEVEL_COLUMNS = (
    ("min", "Minimum water level"),
    ("max", "Maximum water level"),
)


def _series(data, scenario_name, section_names):
    """Metric of one scenario as an array along the estuary."""
    return np.array(
        [data[scenario_name][name] for name in section_names], dtype=float
    )


def _relative(values, reference, mode):
    """
    Normalise a series against the reference scenario at the same x.

    mode "ratio": values / reference (NaN where the reference is <= 0).
    mode "difference": values - reference.
    """
    if mode == "ratio":
        safe = np.where(reference > 0, reference, np.nan)
        return values / safe
    return values - reference


def plot_metric_grid(
    flow_condition,
    scenario_names,
    section_names,
    x_km,
    rows,
    filename,
    description,
    bed_row=None,
    zero_based=True,
    relative=False,
):
    """
    Generic along-estuary summary figure.

    Left column = minimum water level, right column = maximum water level.
    Each row is one metric; the two panels of a row share the y-axis so they
    can be compared directly. All scenarios are drawn as solid lines (colour =
    scenario) in every panel.

    Parameters
    ----------
    rows : list[tuple[str, dict, str]]
        (y-label, data, y-label when relative) per row, with
        data[level][scenario][section] and level in {"min", "max"}.
    bed_row : tuple[str, dict, str] or None
        Optional extra top row spanning both columns for a metric that does
        not depend on the water level: (y-label, data[scenario][section],
        y-label when relative).
    zero_based : bool
        Start the y-axis of the water-level rows at zero (absolute mode only).
    relative : bool
        If True, every scenario is normalised against REFERENCE_SCENARIO at
        the same cross-section, which removes the shared along-estuary trend
        (tide- to river-dominated) so differences between scenarios stand
        out. Water-level rows are shown as a ratio (1 = same as the
        reference); the bed-level row, which can be negative, is shown as a
        difference in metres (0 = same as the reference). The reference
        scenario itself becomes the horizontal baseline.
    """
    if relative and REFERENCE_SCENARIO not in scenario_names:
        raise ValueError(
            f"Reference scenario '{REFERENCE_SCENARIO}' is not in {scenario_names}."
        )
    plotted = [
        s for s in scenario_names
        if not (relative and s == REFERENCE_SCENARIO)
    ]

    def add_baseline(ax, value):
        ax.axhline(
            value, color=GREY_CONST, linewidth=1.2, linestyle="-",
            label=f"{scenario_legend_label(REFERENCE_SCENARIO)} (reference)",
            zorder=1,
        )

    n_rows = len(rows) + (1 if bed_row is not None else 0)
    fig = plt.figure(figsize=(9.0, 2.5 * n_rows + 0.9))
    gs = fig.add_gridspec(n_rows, 2)

    x_ref = None
    legend_ax = None
    last_row_axes = []
    r0 = 0

    if bed_row is not None:
        ylabel, data, ylabel_rel = bed_row
        ax = fig.add_subplot(gs[0, :])
        x_ref = ax
        legend_ax = ax
        if relative:
            add_baseline(ax, 0.0)
        for scenario_name in plotted:
            values = _series(data, scenario_name, section_names)
            if relative:
                values = _relative(
                    values, _series(data, REFERENCE_SCENARIO, section_names),
                    "difference",
                )
            ax.plot(
                x_km, values,
                color=get_scenario_color(scenario_name, color_by="pm"),
                linewidth=1.4, marker="o", markersize=3,
                label=scenario_legend_label(scenario_name),
            )
        ax.set_ylabel(ylabel_rel if relative else ylabel)
        ax.set_title("Independent of water level", fontsize=9)
        _style_axes(ax)
        last_row_axes = [ax]
        r0 = 1

    for i, (ylabel, data_by_level, ylabel_rel) in enumerate(rows):
        r = r0 + i
        ax_left = fig.add_subplot(gs[r, 0], sharex=x_ref)
        if x_ref is None:
            x_ref = ax_left
        ax_right = fig.add_subplot(gs[r, 1], sharex=x_ref, sharey=ax_left)
        if legend_ax is None:
            legend_ax = ax_left

        for ax, (level, level_title) in zip((ax_left, ax_right), LEVEL_COLUMNS):
            if relative:
                add_baseline(ax, 1.0)
            for scenario_name in plotted:
                values = _series(data_by_level[level], scenario_name, section_names)
                if relative:
                    reference = _series(
                        data_by_level[level], REFERENCE_SCENARIO, section_names
                    )
                    values = _relative(values, reference, "ratio")
                ax.plot(
                    x_km, values,
                    color=get_scenario_color(scenario_name, color_by="pm"),
                    linewidth=1.4, marker="o", markersize=3,
                    label=scenario_legend_label(scenario_name),
                )
            if i == 0:
                ax.set_title(level_title, fontsize=9)
            _style_axes(ax)

        ax_left.set_ylabel(ylabel_rel if relative else ylabel)
        ax_right.tick_params(labelleft=False)
        if zero_based and not relative:
            ax_left.set_ylim(bottom=0.0)
        last_row_axes = [ax_left, ax_right]

    # x-axis labels only on the bottom row; shared x-limits.
    x_ref.set_xlim(float(np.min(x_km)), float(np.max(x_km)))
    for ax in fig.axes:
        ax.tick_params(labelbottom=False)
    for ax in last_row_axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Along-estuary distance [km]")

    legend_ax.legend(loc="best", frameon=False, fontsize=8)
    suffix = (
        f" (relative to {scenario_legend_label(REFERENCE_SCENARIO)})"
        if relative else ""
    )
    fig.suptitle(_title_text(flow_condition) + suffix)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    name = f"{filename}_relative" if relative else filename
    output_path = OUTPUT_DIR / flow_condition / f"{name}_{flow_condition}.png"
    _save_figure(fig, output_path, description)


def plot_wet_flow_area_along_estuary(
    flow_condition,
    scenario_names,
    section_names,
    x_km,
    wetted_area_max,
    wetted_area_min,
    relative=False,
):
    """Wet flow area vs along-estuary distance (left: min WL, right: max WL)."""
    plot_metric_grid(
        flow_condition, scenario_names, section_names, x_km,
        rows=[(
            "Wet flow area [m$^2$]",
            {"max": wetted_area_max, "min": wetted_area_min},
            "Wet flow area /\nconstant [-]",
        )],
        filename="wet_flow_area_along_estuary",
        description="wet-flow-area figure",
        relative=relative,
    )


def plot_channel_width_along_estuary(
    flow_condition,
    scenario_names,
    section_names,
    x_km,
    channel_width_single,
    channel_width_total,
    relative=False,
):
    """
    Channel widths vs along-estuary distance (left: min WL, right: max WL).

    Top row: widest single channel. Bottom row: cumulative width of all channels.
    """
    plot_metric_grid(
        flow_condition, scenario_names, section_names, x_km,
        rows=[
            ("Widest single\nchannel [m]", channel_width_single,
             "Widest single channel /\nconstant [-]"),
            ("Cumulative channel\nwidth [m]", channel_width_total,
             "Cumulative channel width /\nconstant [-]"),
        ],
        filename="channel_width_along_estuary",
        description="channel-width figure",
        relative=relative,
    )


def plot_channel_count_along_estuary(
    flow_condition,
    scenario_names,
    section_names,
    x_km,
    channel_count,
    relative=False,
):
    """Number of separate channels (left: min WL, right: max WL)."""
    plot_metric_grid(
        flow_condition, scenario_names, section_names, x_km,
        rows=[(
            "Number of separate\nchannels [-]", channel_count,
            "Number of channels /\nconstant [-]",
        )],
        filename="channel_count_along_estuary",
        description="channel-count figure",
        relative=relative,
    )


def plot_depth_along_estuary(
    flow_condition,
    scenario_names,
    section_names,
    x_km,
    thalweg_bed_level,
    max_water_depth,
    thalweg_water_depth,
    relative=False,
):
    """
    Channel depth vs along-estuary distance.

    Top row (spans both columns): deepest bed level, independent of water level.
    Middle row: maximum water depth anywhere in the section.
    Bottom row: water depth at the thalweg (lowest bed point).
    Left column = minimum water level, right column = maximum water level.
    """
    plot_metric_grid(
        flow_condition, scenario_names, section_names, x_km,
        rows=[
            ("Max water depth anywhere\nin section [m]", max_water_depth,
             "Max water depth /\nconstant [-]"),
            ("Water depth at\nthalweg [m]", thalweg_water_depth,
             "Thalweg water depth /\nconstant [-]"),
        ],
        bed_row=("Deepest bed level [m]", thalweg_bed_level,
                 "Deepest bed level minus\nconstant [m]"),
        filename="channel_depth_along_estuary",
        description="channel-depth figure",
        relative=relative,
    )


# %%
# =============================================================================
# LOAD MAP OUTPUT AND EXTRACT CROSS-SECTION PROFILES
# ============================================================================


final_bed_profiles = {}
maximum_water_levels = {}
minimum_water_levels = {}
wetted_area_max = {}
wetted_area_min = {}
# channel_*[water-level case ("max" / "min")][scenario][section]
channel_width_single = {"max": {}, "min": {}}
channel_width_total = {"max": {}, "min": {}}
channel_count = {"max": {}, "min": {}}
# Depth metrics.
thalweg_bed_level = {}                      # [scenario][section]: deepest bed level [m]
max_water_depth = {"max": {}, "min": {}}    # [level][scenario][section]: max(WL - bed) [m]
thalweg_water_depth = {"max": {}, "min": {}}  # [level][scenario][section]: WL - bed at thalweg [m]
y_samples_reference_m = None
run_path_info = {}
selected_time_info = {}


for scenario_name, run_dir in SCENARIOS.items():
    print("\n" + "=" * 80)
    print(f"Processing {scenario_name}")
    print(f"Run directory: {run_dir}")
    print("=" * 80)

    # Match the normal project workflow: discover stitched runs and let the
    # shared map-cache loader handle partitioned output, caching and updates.
    base_path = run_dir.parent
    folder_name = run_dir.name
    assessment_dir = base_path / "cached_data"
    assessment_dir.mkdir(parents=True, exist_ok=True)

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=None,
        variability_map=None,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [run_dir]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=LOAD_VARS,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        raise FileNotFoundError(
            f"The shared MAP cache loader returned no dataset for scenario "
            f"'{scenario_name}' (folder '{folder_name}')."
        )

    print(f"Stitched run paths supplied to cache: {len(run_paths)}")
    for run_path in run_paths:
        print(f"  {run_path}")

    try:
        missing_vars = [name for name in LOAD_VARS if name not in ds]
        if missing_vars:
            raise KeyError(
                f"Cached dataset for '{scenario_name}' is missing {missing_vars}. "
                f"Available variables: {list(ds.variables)}"
            )

        face_coordinates = np.asarray(ds.grid.face_coordinates, dtype=float)
        if face_coordinates.ndim != 2 or face_coordinates.shape[1] < 2:
            raise ValueError(
                "Expected ds.grid.face_coordinates with shape (n_faces, 2); "
                f"received {face_coordinates.shape}."
            )
        face_x = face_coordinates[:, 0]
        face_y = face_coordinates[:, 1]

        print(
            f"Cached face x range: {np.nanmin(face_x):.1f} to "
            f"{np.nanmax(face_x):.1f} m"
        )
        print(
            f"Cached face y range: {np.nanmin(face_y):.1f} to "
            f"{np.nanmax(face_y):.1f} m"
        )

        if (
            np.nanmax(face_x) < X_COORDS_M.min()
            or np.nanmin(face_x) > X_COORDS_M.max()
        ):
            raise ValueError(
                "The cached MAP dataset does not cover the requested "
                "cross-section x range."
            )

        if (
            np.nanmax(face_y) < Y_RANGE_M[0]
            or np.nanmin(face_y) > Y_RANGE_M[1]
        ):
            raise ValueError(
                "The cached MAP dataset does not cover the requested "
                "cross-section y range."
            )

        final_indices, final_times, final_time = get_final_twelve_hour_indices(
            ds=ds,
            waterlevel_var=WATERLEVEL_VAR,
        )

        print(
            f"Final output time: {final_time}\n"
            f"Final-twelve-hour output times: {final_times[0]} to "
            f"{final_times[-1]}\n"
            f"Number of final-twelve-hour map times: {final_indices.size}"
        )

        # a) Bed level at the final available output timestep.
        y_samples_m, scenario_final_bed = sample_profiles_at_x(
            ds=ds,
            variable_name=BEDLEVEL_VAR,
            x_coords_m=X_COORDS_M,
            y_range_m=Y_RANGE_M,
            n_y_samples=N_Y_SAMPLES,
            time_indices=None,
            reduction="last",
        )

        # b) Maximum water level during the final twelve hours.
        _, scenario_max_water_level = sample_profiles_at_x(
            ds=ds,
            variable_name=WATERLEVEL_VAR,
            x_coords_m=X_COORDS_M,
            y_range_m=Y_RANGE_M,
            n_y_samples=N_Y_SAMPLES,
            time_indices=final_indices,
            reduction="max",
        )

        # c) Minimum water level during the final twelve hours.
        _, scenario_min_water_level = sample_profiles_at_x(
            ds=ds,
            variable_name=WATERLEVEL_VAR,
            x_coords_m=X_COORDS_M,
            y_range_m=Y_RANGE_M,
            n_y_samples=N_Y_SAMPLES,
            time_indices=final_indices,
            reduction="min",
        )

        if y_samples_reference_m is None:
            y_samples_reference_m = y_samples_m
        elif not np.allclose(y_samples_reference_m, y_samples_m):
            raise ValueError(
                "The regular y-sampling positions differ between scenarios."
            )

        final_bed_profiles[scenario_name] = scenario_final_bed
        maximum_water_levels[scenario_name] = scenario_max_water_level
        minimum_water_levels[scenario_name] = scenario_min_water_level

        # d) Wet flow area per cross-section at the maximum and minimum water
        #    level, relative to the final bed level.
        wetted_area_max[scenario_name] = {
            section_name: compute_wetted_area(
                y_samples_m,
                scenario_final_bed[section_name],
                scenario_max_water_level[section_name],
            )
            for section_name in scenario_final_bed
        }
        wetted_area_min[scenario_name] = {
            section_name: compute_wetted_area(
                y_samples_m,
                scenario_final_bed[section_name],
                scenario_min_water_level[section_name],
            )
            for section_name in scenario_final_bed
        }

        # e) Channel widths, channel count and depth metrics per cross-section
        #    at the maximum and minimum water level.
        for level, water_levels in (
            ("max", scenario_max_water_level),
            ("min", scenario_min_water_level),
        ):
            widths_by_section = {
                section_name: compute_channel_widths(
                    y_samples_m,
                    scenario_final_bed[section_name],
                    water_levels[section_name],
                )
                for section_name in scenario_final_bed
            }
            channel_width_single[level][scenario_name] = {
                section_name: widths[0]
                for section_name, widths in widths_by_section.items()
            }
            channel_width_total[level][scenario_name] = {
                section_name: widths[1]
                for section_name, widths in widths_by_section.items()
            }
            channel_count[level][scenario_name] = {
                section_name: widths[2]
                for section_name, widths in widths_by_section.items()
            }

            # Maximum water depth anywhere in the section.
            max_water_depth[level][scenario_name] = {
                section_name: compute_max_water_depth(
                    scenario_final_bed[section_name],
                    water_levels[section_name],
                )
                for section_name in scenario_final_bed
            }
            # Water depth at the thalweg (lowest bed point).
            thalweg_water_depth[level][scenario_name] = {
                section_name: compute_thalweg_water_depth(
                    scenario_final_bed[section_name],
                    water_levels[section_name],
                )
                for section_name in scenario_final_bed
            }

        # f) Deepest bed level per cross-section (independent of water level).
        thalweg_bed_level[scenario_name] = {
            section_name: compute_thalweg_bed_level(
                scenario_final_bed[section_name]
            )
            for section_name in scenario_final_bed
        }

        run_path_info[scenario_name] = list(run_paths)
        selected_time_info[scenario_name] = {
            "final_time": final_time,
            "first_final_twelve_hour_time": final_times[0],
            "last_final_twelve_hour_time": final_times[-1],
            "n_final_twelve_hour_times": int(final_indices.size),
        }
    finally:
        ds.close()


# %%
# =============================================================================
# DIAGNOSTICS: MARGINAL SILLS AND CHANNEL-COUNT SENSITIVITY
# =============================================================================

def report_marginal_samples(
    scenario_name, section_name, level="max", max_depth_m=0.25
):
    """List samples that are wet but only by a few cm (sills that merge channels)."""
    wl = (maximum_water_levels if level == "max" else minimum_water_levels)[
        scenario_name][section_name]
    bed = final_bed_profiles[scenario_name][section_name]
    depth = wl - bed
    idx = np.where(np.isfinite(depth) & (depth > 0) & (depth <= max_depth_m))[0]
    print(
        f"{scenario_name} {section_name} ({level} WL): {idx.size} samples "
        f"with 0 < depth <= {max_depth_m} m"
    )
    for i in idx:
        print(
            f"  y = {y_samples_reference_m[i] / 1000:.3f} km, "
            f"depth = {depth[i] * 100:.1f} cm"
        )


def channel_count_sensitivity(
    scenario_name, section_name, level="max",
    thresholds=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
):
    """Channel metrics as a function of the wet-depth threshold."""
    wl = (maximum_water_levels if level == "max" else minimum_water_levels)[
        scenario_name][section_name]
    bed = final_bed_profiles[scenario_name][section_name]
    print(f"{scenario_name} {section_name} ({level} WL)")
    for t in thresholds:
        w1, wt, n = compute_channel_widths(
            y_samples_reference_m, bed, wl, threshold_m=t, min_width_m=0.0
        )
        print(
            f"  threshold {t:5.2f} m -> {n} channels, "
            f"widest {w1:6.0f} m, cumulative {wt:6.0f} m"
        )


# Set to True to print the diagnostics for the cases below (edit as needed).
RUN_THRESHOLD_DIAGNOSTICS = True
DIAGNOSTIC_CASES = [("pm3_n5_mean_flow", "km36")]

if RUN_THRESHOLD_DIAGNOSTICS:
    for _scenario, _section in DIAGNOSTIC_CASES:
        for _level in ("max", "min"):
            print()
            report_marginal_samples(_scenario, _section, _level)
            channel_count_sensitivity(_scenario, _section, _level)

# %%
# =============================================================================
# PLOT COMPARISON PAIRS
# =============================================================================

section_names = [f"km{int(round(x_km))}" for x_km in CROSS_SECTION_X_KM]

for flow_condition, scenario_names in COMPARISON_PAIRS.items():
    print(f"\nPlotting {flow_condition} comparison:")
    print("  " + " vs ".join(scenario_names))

    if PLOT_MODE in {"combined", "both"}:
        print("  Creating combined plots (all scenarios on the same axes).")
        for section_name in section_names:
            plot_cross_section_comparison(
                flow_condition=flow_condition,
                scenario_names=scenario_names,
                section_name=section_name,
                y_samples_m=y_samples_reference_m,
                final_bed_profiles=final_bed_profiles,
                maximum_water_levels=maximum_water_levels,
                minimum_water_levels=minimum_water_levels,
            )

    if PLOT_MODE in {"subplots", "both"}:
        print("  Creating subplot figures (one scenario per row).")
        for section_name in section_names:
            plot_cross_section_subplots(
                flow_condition=flow_condition,
                scenario_names=scenario_names,
                section_name=section_name,
                y_samples_m=y_samples_reference_m,
                final_bed_profiles=final_bed_profiles,
                maximum_water_levels=maximum_water_levels,
                minimum_water_levels=minimum_water_levels,
            )

    for relative in SUMMARY_RELATIVE_FLAGS:
        mode_text = "relative to reference" if relative else "absolute"
        print(f"  Creating along-estuary summary plots ({mode_text}).")

        plot_wet_flow_area_along_estuary(
            flow_condition=flow_condition,
            scenario_names=scenario_names,
            section_names=section_names,
            x_km=CROSS_SECTION_X_KM,
            wetted_area_max=wetted_area_max,
            wetted_area_min=wetted_area_min,
            relative=relative,
        )

        plot_channel_width_along_estuary(
            flow_condition=flow_condition,
            scenario_names=scenario_names,
            section_names=section_names,
            x_km=CROSS_SECTION_X_KM,
            channel_width_single=channel_width_single,
            channel_width_total=channel_width_total,
            relative=relative,
        )

        plot_channel_count_along_estuary(
            flow_condition=flow_condition,
            scenario_names=scenario_names,
            section_names=section_names,
            x_km=CROSS_SECTION_X_KM,
            channel_count=channel_count,
            relative=relative,
        )

        plot_depth_along_estuary(
            flow_condition=flow_condition,
            scenario_names=scenario_names,
            section_names=section_names,
            x_km=CROSS_SECTION_X_KM,
            thalweg_bed_level=thalweg_bed_level,
            max_water_depth=max_water_depth,
            thalweg_water_depth=thalweg_water_depth,
            relative=relative,
        )

# %%
# =============================================================================
# FINISH
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED-HYDRO CROSS-SECTION ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory:\n{OUTPUT_DIR}")
print(f"Plot mode: {PLOT_MODE}")
if PLOT_MODE in {"subplots", "both"}:
    print("Subplot figures are saved under:")
    print(f"  {OUTPUT_DIR / 'mean_flow' / 'subplots'}")
    print(f"  {OUTPUT_DIR / 'peak_flow' / 'subplots'}")
print("\nStitched run paths supplied to cache and final-twelve-hour windows:")

for scenario_name in SCENARIOS:
    info = selected_time_info[scenario_name]
    print(f"\n{scenario_name}")
    print(f"  Number of stitched run paths: {len(run_path_info[scenario_name])}")
    for run_path in run_path_info[scenario_name]:
        print(f"    {run_path}")

    print(f"  Final time: {info['final_time']}")
    print(
        f"  Final-twelve-hour window: "
        f"{info['first_final_twelve_hour_time']} to "
        f"{info['last_final_twelve_hour_time']}"
    )
    print(
        f"  Number of map times used: "
        f"{info['n_final_twelve_hour_times']}"
    )


print("\nPlot conventions:")

print(
    "  Cross-section subplots: filled body = bed level at the final available "
    "output timestep; shaded band = min-max water level during the final "
    "twelve hours of the detailed-hydro run."
)

print(
    "  Combined cross-sections: solid = final bed, dashed = maximum water "
    "level, dotted = minimum water level."
)

print(
    "  Wet flow area: integral of (water level - final bed level, clipped at "
    f"zero) over y = {Y_RANGE_M[0] / 1000:.0f}-{Y_RANGE_M[1] / 1000:.0f} km."
)

print(
    "  Channel width: a channel is a contiguous stretch of samples with "
    f"water level - final bed level > {WET_DEPTH_THRESHOLD_M} m, within "
    f"y = {Y_RANGE_M[0] / 1000:.0f}-{Y_RANGE_M[1] / 1000:.0f} km. "
    f"Wet stretches narrower than {MIN_CHANNEL_WIDTH_M:g} m are ignored. "
    "Widest single channel, cumulative width of all channels and number of "
    "channels are shown in separate panels: left column = minimum water "
    "level, right column = maximum water level."
)

print(
    "  Channel depth: deepest bed level = minimum final bed level in the "
    "strip (thalweg). Max water depth = max(water level - final bed level) "
    "over the strip. Thalweg water depth = water level - final bed level at "
    "the y-sample where the bed is lowest. Both depths are shown in separate "
    "panels: left column = minimum water level, right column = maximum "
    "water level."
)

print("=" * 80)