# %% Compare final bed profiles and final-2-hour water-level envelopes

"""

Compare detailed-hydrodynamic-run cross-sections for mean-flow and peak-flow simulations.

For every cross-section, this script plots:
- Bed level at the final available output timestep.
- Maximum water level during the final two hours.
- Minimum water level during the final two hours.

Plot output can show both scenarios on one set of axes, one scenario per
subplot, or both output styles. Subplot figures are saved beneath each flow
condition in a dedicated ``subplots`` directory.

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
    "pm3_n5_mean_flow": Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_mean.964903"),
    "pm5_n1_mean_flow": Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_mean.964908"),
    "pm3_n5_peak_flow": Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_peak.964907"),
    "pm5_n1_peak_flow": Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_peak.965221"),
}

COMPARISON_PAIRS = {
    "mean_flow": ["pm3_n5_mean_flow", "pm5_n1_mean_flow"],
    "peak_flow": ["pm3_n5_peak_flow", "pm5_n1_peak_flow"],
}

DISCHARGE = 500

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
# Same cross-section locations as the successful cached-MAP script.

CROSS_SECTION_X_KM = np.arange(25.0, 46.0, 1.0)
X_COORDS_M = CROSS_SECTION_X_KM * 1_000.0

# Same central 5-km strip as the successful uniform-profile script.
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y_SAMPLES = 500

# Optional profile-only mask. Set to None to retain every sampled value.
PROFILE_LAND_THRESHOLD_M = None

# %%
# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
    r"\detailed-hydro-run\output_plots"
    r"\cross_sections_final_bed_water_level_envelope"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
# Plot output options:
#   "combined"  : both scenarios plotted on the same axes (original behaviour)
#   "subplots"  : one scenario per subplot, saved in <flow_condition>/subplots
#   "both"      : save both styles
PLOT_MODE = "both"
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

def get_scenario_color(label, color_by="pm"):
    """Return a scenario colour based on its pm or n value."""
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
# FUNCTIONS: PLOTTING
# =============================================================================

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

    Plot final bed level and the final-two-hour water-level envelope for one

    cross-section.

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



    ax.set_title(
        rf"$Q_{{mean}} = {DISCHARGE}$ m$^3$ s$^{{-1}}$, "
        rf"{flow_condition.replace('_', ' ')}, "
        rf"$x = {km}$ km"
    )

    ax.set_xlabel("Cross-estuary distance [km]")
    ax.set_ylabel("Elevation [m]")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / flow_condition
        / f"cross_section_{section_name}_{flow_condition}.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path.name}")

def plot_cross_section_subplots(
    flow_condition,
    scenario_names,
    section_name,
    y_samples_m,
    final_bed_profiles,
    maximum_water_levels,
    minimum_water_levels,
):
    """Plot one scenario per panel for a single cross-section."""
    km = int(section_name.replace("km", ""))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(scenario_names),
        figsize=(6.4 * len(scenario_names), 4.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, scenario_name in zip(axes, scenario_names):
        color = get_scenario_color(scenario_name, color_by="pm")

        ax.plot(
            y_samples_m / 1000.0,
            final_bed_profiles[scenario_name][section_name],
            color=color,
            linewidth=1.5,
            linestyle="-",
            label="Final bed",
        )
        ax.plot(
            y_samples_m / 1000.0,
            maximum_water_levels[scenario_name][section_name],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label="Maximum water level (final 2 h)",
        )
        ax.plot(
            y_samples_m / 1000.0,
            minimum_water_levels[scenario_name][section_name],
            color=color,
            linewidth=1.2,
            linestyle=":",
            label="Minimum water level (final 2 h)",
        )

        ax.set_title(scenario_legend_label(scenario_name))
        ax.set_xlabel("Cross-estuary distance [km]")
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=8)

    axes[0].set_ylabel("Elevation [m]")
    fig.suptitle(
        rf"$Q_{{mean}} = {DISCHARGE}$ m$^3$ s$^{{-1}}$, "
        rf"{flow_condition.replace('_', ' ')}, $x = {km}$ km"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = (
        OUTPUT_DIR
        / flow_condition
        / "subplots"
        / f"cross_section_{section_name}_{flow_condition}_subplots.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved subplot figure: {output_path}")

# %%
# =============================================================================
# LOAD MAP OUTPUT AND EXTRACT CROSS-SECTION PROFILES
# ============================================================================


final_bed_profiles = {}
maximum_water_levels = {}
minimum_water_levels = {}
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
# PLOT COMPARISON PAIRS
# =============================================================================

section_names = [f"km{int(round(x_km))}" for x_km in CROSS_SECTION_X_KM]

for flow_condition, scenario_names in COMPARISON_PAIRS.items():
    print(f"\nPlotting {flow_condition} comparison:")
    print(f"  {scenario_names[0]} vs {scenario_names[1]}")

    if PLOT_MODE in {"combined", "both"}:
        print("  Creating combined plots (both scenarios on the same axes).")
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
        print("  Creating subplot figures (one scenario per panel).")
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
print("\nStitched run paths supplied to cache and final-two-hour windows:")

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
    "  Solid line: bed level at the final available output timestep."
)

print(
    "  Dashed line: maximum water level during the final twelve hours "
    "of the detailed-hydro run."

)

print(
    "  Dotted line: minimum water level during the final twelve hours "
    "of the detailed-hydro run.")

print("=" * 80)
