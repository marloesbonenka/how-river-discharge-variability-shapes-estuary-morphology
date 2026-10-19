# %% Detailed-hydro sediment routing and short-period zone budgets
"""
Compute 2D transport-routing diagnostics and cross-section sediment budgets
from detailed-hydro MAP caches using repository loading/caching functions.

File loading follows the pm/n sensitivity script:
- base_path = base_directory / Model_Output/Q{DISCHARGE}
- cache_dir = base_path / 'cached_data'
- run_paths from get_stitched_map_run_paths(...), falling back to
  [base_path / folder] when empty
- load_or_update_map_cache_multi(...)
- face coordinates from _get_face_coords(ds)

For every scenario:
- Load SX_VAR / SY_VAR (default mesh2d_sxtot / mesh2d_sytot) and water depth.
- Sum over the sediment-fraction dimension (nSedTot).
- Convert transport to bed-volume-equivalent units (units read from the file).
- Compute time-mean and cumulative transport.
- Compute a 2D potential bed-volume-change diagnostic:
      potential change = -div(cumulative q) / (1 - porosity)
- Integrate the x-component of transport through fixed x-boundaries.
- Derive cumulative net volume in each 5-km zone.

Sign conventions
----------------
- Zone budget: net rate = F(x_left) - F(x_right), where F is the flux in
  the +x direction of the model. Positive zone volume = convergence =
  potential deposition. This is independent of which way +x points.
- Displayed cross-section flux is made 'seaward-positive' using
  POSITIVE_X_IS_SEAWARD (this only affects the flux plots / CSV).

Important limitation
--------------------
These are short-period routing budgets. If the detailed-hydro runs have a
fixed bed, the potential erosion/deposition map is not realised morphology.
"""

# %% IMPORTS
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# Add repo root to path (one level up from script location)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions.F_general import get_variability_map
from functions.F_map_cache import (
    cache_tag_from_bbox,
    load_or_update_map_cache_multi,
    _get_face_coords,
)
from functions.F_loaddata import get_stitched_map_run_paths

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz


# %% BASE PATHS (same construction as the sensitivity script)
DISCHARGE = 500

base_directory = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
)
config = f"Model_Output/Q{DISCHARGE}"

BASE_PATH = base_directory / config
VARIABILITY_MAP = get_variability_map(DISCHARGE)

ASSESSMENT_DIR = BASE_PATH / "cached_data"
ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
TIMED_OUT_DIR = BASE_PATH / "timed-out"


# %% SCENARIOS
SCENARIOS = {
    "constant": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_01_Qr500_pm1_n0.9724783"
    ),
    "pm3_n5_mean_flow": Path(
        r"u:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_03_Qr500_pm3_n5_mean.964903"
    ),
    "pm5_n1_mean_flow": Path(
        r"u:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_05_Qr500_pm5_n1_mean.964908"
    ),
    "pm3_n5_peak_flow": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_03_Qr500_pm3_n5_peak.964907"
    ),
    "pm5_n1_peak_flow": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_05_Qr500_pm5_n1_peak.965221"
    ),
}

COMPARISON_PAIRS = {
    "mean_flow": [
        "constant",
        "pm3_n5_mean_flow",
        "pm5_n1_mean_flow",
    ],
    "peak_flow": [
        "constant",
        "pm3_n5_peak_flow",
        "pm5_n1_peak_flow",
    ],
}


# %% SPATIAL DOMAIN & CACHE SETTINGS
# Detailed-hydro zoom domain.
X_LIMITS_M = (19_000.0, 45_000.0)
Y_LIMITS_M = (5_000.0, 10_000.0)

# Cache settings matching the sensitivity script
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Cross-section boundaries used for the along-estuary budget.
BOUNDARY_X_KM = np.arange(20.0, 46.0, 5.0)
BOUNDARY_X_M = BOUNDARY_X_KM * 1000.0

# Zones are formed between consecutive boundaries.
ZONE_EDGES_KM = BOUNDARY_X_KM

# Regular y grid used to integrate q_x through each x boundary.
N_Y_SAMPLES = 1000

# Optional override of the y-range of the cross-section budget
# (None -> Y_LIMITS_M). Must cover the full width over which sediment moves.
BUDGET_Y_RANGE_M = None

# Maximum distance allowed when sampling a face onto a boundary line.
# (grid cells are 85 m x 50 m)
MAX_BOUNDARY_SAMPLE_DISTANCE_M = 250.0

# Minimum mean water depth for including a face in map diagnostics.
MIN_WATER_DEPTH_M = 0.05

# Regular diagnostic grid used for the 2D transport-convergence map.
DIAGNOSTIC_DX_M = 250.0
DIAGNOSTIC_DY_M = 250.0


# %% TRANSPORT SETTINGS
# NOTE: in D-Flow FM, sxtot/sytot are TOTAL load (bed + suspended).
# For bedload only, use e.g. SX_VAR = "mesh2d_sbcx", SY_VAR = "mesh2d_sbcy"
# (plus the wave-related sbwx/sbwy if you want those included).
SX_VAR = "mesh2d_sxtot"
SY_VAR = "mesh2d_sytot"
WATER_DEPTH_VAR = "mesh2d_waterdepth"

LOAD_VARS = [SX_VAR, SY_VAR, WATER_DEPTH_VAR]

# Fallback transport units, used only if the 'units' attribute of the
# variable is unrecognised. Options: "volume_bed", "volume_solid", "mass".
# The attribute is checked first: kg-based -> mass, m3-based -> volume_solid
# (unless this is set to "volume_bed").
TRANSPORT_UNITS = "volume_solid"

# Solid sediment density, used only for mass-based transport.
SEDIMENT_DENSITY_KG_M3 = 2650.0

# Porosity converts solid transport volume to bed-volume equivalent.
POROSITY = 0.4
BED_VOLUME_FACTOR = 1.0 / (1.0 - POROSITY)

# Morphological acceleration factor of the detailed-hydro runs
# (1.0 for a fixed bed / no acceleration). Multiplies all time integrals.
MORFAC = 1.0

# Direction of the model x-axis. In the HIS buffer script the river is at
# ~44 km and distance is counted from the sea, so +x points LANDWARD.
# Only affects the sign of the displayed cross-section flux, not the budgets.
POSITIVE_X_IS_SEAWARD = False
SEAWARD_SIGN = 1.0 if POSITIVE_X_IS_SEAWARD else -1.0


# %% TIME SETTINGS
TIME_START_INDEX = 0
TIME_END_INDEX = None
TIME_CHUNK = 25  # timesteps read per block (limits memory use)


# %% OUTPUT AND PLOTTING
OUTPUT_BASE = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Output\Q500"
    r"\detailed-hydro-run\output_plots\sediment_budgets"
)

DPI = 300
SHOW_FIGURES = False

SCENARIO_COLORS = {
    "constant": "#7f7f7f",
    "pm3_n5_mean_flow": plt.cm.Blues(0.55),
    "pm5_n1_mean_flow": plt.cm.Blues(0.92),
    "pm3_n5_peak_flow": plt.cm.Oranges(0.55),
    "pm5_n1_peak_flow": plt.cm.Oranges(0.92),
}

QUIVER_STRIDE = 15


# %% Helper functions
def get_face_dimension(ds, variable_name, face_size):
    """Return the face dimension of a MAP variable.

    MAP transport variables can have extra non-time dimensions (e.g. the
    sediment fraction 'nSedTot'), so pick the face dimension by name and,
    as a fallback, by matching the number of faces.
    """
    if variable_name not in ds:
        raise KeyError(
            f"'{variable_name}' not found. Available variables:\n"
            f"{list(ds.variables)}"
        )

    dims = ds[variable_name].dims
    if "time" not in dims:
        raise ValueError(f"{variable_name} has no time dimension: {dims}")

    candidates = [d for d in dims if d != "time"]

    by_name = [d for d in candidates if "nface" in d.lower()]
    if len(by_name) == 1:
        return by_name[0]

    by_size = [d for d in candidates if ds.sizes[d] == face_size]
    if len(by_size) == 1:
        return by_size[0]

    raise ValueError(
        f"Cannot identify the face dimension of {variable_name}; "
        f"dims={dims}, number of faces={face_size}."
    )


def read_summed_transport(da, face_dimension, start, end, chunk=TIME_CHUNK):
    """Read da[time, (fraction), face] in blocks and sum over fractions.

    Returns an array of shape (end - start, n_faces).
    """
    extra_dims = [d for d in da.dims if d not in ("time", face_dimension)]
    out = np.empty((end - start, da.sizes[face_dimension]), dtype=float)

    for s in range(start, end, chunk):
        e = min(s + chunk, end)
        block = da.isel(time=slice(s, e))
        if extra_dims:
            block = block.sum(dim=extra_dims, skipna=True)
        out[s - start:e - start] = np.asarray(
            block.transpose("time", face_dimension).values, dtype=float
        )

    return out


def resolve_transport_units(ds, variable_name):
    """Decide the unit interpretation from the variable's 'units' attribute."""
    raw = str(ds[variable_name].attrs.get("units", ""))
    units = raw.lower().replace(" ", "").replace("^", "")

    if units.startswith("kg"):
        resolved = "mass"
    elif units.startswith("m3"):
        resolved = (
            TRANSPORT_UNITS
            if TRANSPORT_UNITS in ("volume_bed", "volume_solid")
            else "volume_solid"
        )
    else:
        resolved = TRANSPORT_UNITS
        print(f"  [WARNING] Unrecognised units '{raw}' for {variable_name}; "
              f"using TRANSPORT_UNITS = '{resolved}'.")

    print(f"  Units attribute of {variable_name}: '{raw}' -> {resolved}")
    return resolved


def bed_volume_factor(units_mode):
    """Factor converting the raw transport to bed-volume-equivalent rate."""
    if units_mode == "volume_bed":
        return 1.0
    if units_mode == "volume_solid":
        return BED_VOLUME_FACTOR
    if units_mode == "mass":
        return BED_VOLUME_FACTOR / SEDIMENT_DENSITY_KG_M3
    raise ValueError(
        "units mode must be 'volume_bed', 'volume_solid' or 'mass'."
    )


def load_scenario_transport(folder_path):
    """Load detailed-hydro MAP data for one scenario.

    Returns None if no cached data could be loaded.
    """
    folder = Path(folder_path)

    # --- run paths (identical call to the sensitivity script) ---
    run_paths = get_stitched_map_run_paths(
        base_path=BASE_PATH,
        folder_name=folder.name,
        timed_out_dir=TIMED_OUT_DIR,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [BASE_PATH / folder]

    # --- cache load (identical call to the sensitivity script) ---
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=ASSESSMENT_DIR,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=LOAD_VARS,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )
    if ds is None:
        print(f"  No cached data for {folder.name}, skipping.")
        return None

    missing = [name for name in (SX_VAR, SY_VAR) if name not in ds]
    if missing:
        ds.close()
        raise KeyError(f"Missing variables in dataset: {missing}")

    face_x, face_y = _get_face_coords(ds)
    face_x = np.asarray(face_x, dtype=float)
    face_y = np.asarray(face_y, dtype=float)

    face_dimension = get_face_dimension(ds, SX_VAR, face_x.size)
    if ds.sizes[face_dimension] != face_x.size:
        ds.close()
        raise ValueError(
            f"Face coordinates ({face_x.size}) do not match dimension "
            f"{face_dimension} ({ds.sizes[face_dimension]})."
        )

    fraction_dims = [d for d in ds[SX_VAR].dims
                     if d not in ("time", face_dimension)]
    print(f"  Face dimension: {face_dimension}; "
          f"summing over: {fraction_dims or 'nothing'}")

    time = np.asarray(ds["time"].values).astype("datetime64[ns]")
    n_times = time.size

    start = TIME_START_INDEX
    end = n_times if TIME_END_INDEX is None else TIME_END_INDEX

    if not (0 <= start < end <= n_times):
        ds.close()
        raise ValueError(
            f"Invalid time selection: start={start}, end={end}, "
            f"available={n_times}."
        )

    units_mode = resolve_transport_units(ds, SX_VAR)

    print(f"  Reading {end - start} timesteps ...")
    sx = read_summed_transport(ds[SX_VAR], face_dimension, start, end)
    sy = read_summed_transport(ds[SY_VAR], face_dimension, start, end)

    water_depth = None
    if WATER_DEPTH_VAR in ds:
        water_depth = read_summed_transport(
            ds[WATER_DEPTH_VAR], face_dimension, start, end
        )

    return {
        "cache_dataset": ds,
        "time": time[start:end],
        "face_x": face_x,
        "face_y": face_y,
        "sx": sx,
        "sy": sy,
        "water_depth": water_depth,
        "units_mode": units_mode,
    }


def build_boundary_samplers(face_x, face_y, x_boundaries, y_samples,
                            max_distance_m):
    """Pre-compute, per boundary, which face each y-sample maps to.

    The KD-tree is built once; the sampling is then a simple fancy-index
    per timestep (much faster than querying inside the time loop).
    """
    valid = np.isfinite(face_x) & np.isfinite(face_y)
    valid_indices = np.flatnonzero(valid)
    tree = cKDTree(np.column_stack((face_x[valid], face_y[valid])))

    samplers = []
    for x_boundary in x_boundaries:
        points = np.column_stack(
            (np.full(y_samples.size, x_boundary, dtype=float), y_samples)
        )
        distances, tree_indices = tree.query(points)
        samplers.append(
            (valid_indices[tree_indices], distances <= max_distance_m)
        )
    return samplers


def compute_boundary_flux(qx, samplers, dy):
    """Integrate q_x [m3/s/m] over y at every boundary -> [m3/s].

    Returns (n_times, n_boundaries) in the model +x direction.
    """
    flux = np.empty((qx.shape[0], len(samplers)), dtype=float)
    for b, (face_idx, ok) in enumerate(samplers):
        q = np.where(ok[None, :], qx[:, face_idx], np.nan)
        flux[:, b] = np.nansum(q, axis=1) * dy
    return flux


def compute_zone_budgets(boundary_flux, time_seconds):
    """Zone convergence rate and cumulative volume (positive = deposition).

    boundary_flux is in the model +x direction, boundaries increase in x:
    net rate of a zone = inflow at the left edge - outflow at the right edge.
    """
    if boundary_flux.shape[1] < 2:
        raise ValueError("At least two boundaries are required.")

    net_rate = boundary_flux[:, :-1] - boundary_flux[:, 1:]

    cumulative = np.zeros_like(net_rate)
    cumulative[1:] = np.cumsum(
        0.5 * (net_rate[1:] + net_rate[:-1])
        * np.diff(time_seconds)[:, None],
        axis=0,
    )

    return net_rate, cumulative


def grid_mean(values, ix, iy, nx, ny, valid):
    """Average face values into a regular (ny, nx) grid."""
    ok = valid & np.isfinite(values)
    flat = iy[ok] * nx + ix[ok]
    total = np.bincount(flat, weights=values[ok], minlength=nx * ny)
    count = np.bincount(flat, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / count, np.nan)
    return out.reshape(ny, nx)


def save_figure(fig, path):
    path.with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path.name}.png / .pdf")


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved: {path.name}")


# %% Process each scenario
scenario_results = {}

y_budget = np.linspace(
    *(BUDGET_Y_RANGE_M if BUDGET_Y_RANGE_M is not None else Y_LIMITS_M),
    N_Y_SAMPLES,
)
dy_budget = y_budget[1] - y_budget[0]

for scenario_name, folder_path in SCENARIOS.items():
    print("\n" + "=" * 80)
    print(f"Processing: {scenario_name}")
    print(f"Folder: {folder_path}")
    print("=" * 80)

    data = load_scenario_transport(folder_path)
    if data is None:
        continue

    time = data["time"]
    face_x = data["face_x"]
    face_y = data["face_y"]
    water_depth = data["water_depth"]

    # Convert to bed-volume-equivalent rate [m3 bed / s / m]
    factor = bed_volume_factor(data["units_mode"])
    sx = data.pop("sx") * factor
    sy = data.pop("sy") * factor
    data.pop("water_depth")

    # Morphological time (MORFAC = 1 for a fixed-bed detailed-hydro run)
    time_seconds = (
        (time - time[0]) / np.timedelta64(1, "s")
    ).astype(float)
    time_morph = time_seconds * MORFAC
    duration_morph = time_morph[-1]

    output_dir = OUTPUT_BASE / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if water_depth is not None:
        wet_mask = np.nanmean(water_depth, axis=0) >= MIN_WATER_DEPTH_M
    else:
        wet_mask = np.ones(face_x.size, dtype=bool)

    # Cumulative transport per unit width [m3 bed / m] and time-mean [m3/s/m]
    cumulative_sx = _trapz(np.nan_to_num(sx), time_morph, axis=0)
    cumulative_sy = _trapz(np.nan_to_num(sy), time_morph, axis=0)
    mean_sx = cumulative_sx / duration_morph
    mean_sy = cumulative_sy / duration_morph

    # Cross-section fluxes (model +x direction) and zone budgets
    samplers = build_boundary_samplers(
        face_x, face_y, BOUNDARY_X_M, y_budget,
        MAX_BOUNDARY_SAMPLE_DISTANCE_M,
    )
    boundary_flux = compute_boundary_flux(sx, samplers, dy_budget)

    net_zone_rate, cumulative_zone_volume = compute_zone_budgets(
        boundary_flux, time_morph,
    )

    del sx, sy  # free memory before the next scenario

    scenario_results[scenario_name] = {
        "data": data,
        "output_dir": output_dir,
        "mean_sx": mean_sx,
        "mean_sy": mean_sy,
        "cumulative_sx": cumulative_sx,
        "cumulative_sy": cumulative_sy,
        "boundary_flux": boundary_flux,                      # +x direction
        "boundary_flux_seaward": SEAWARD_SIGN * boundary_flux,
        "net_zone_rate": net_zone_rate,
        "cumulative_zone_volume": cumulative_zone_volume,
        "wet_mask": wet_mask,
        "duration_morph": duration_morph,
    }

    print(f"  Final zone volumes [m3]: "
          f"{np.round(cumulative_zone_volume[-1]).astype(int)}")


# %% Time-mean transport-vector maps
for scenario_name, result in scenario_results.items():
    face_x = result["data"]["face_x"]
    face_y = result["data"]["face_y"]
    wet = result["wet_mask"]

    plot_indices = np.flatnonzero(wet)[::QUIVER_STRIDE]

    magnitude = np.hypot(
        result["mean_sx"][plot_indices],
        result["mean_sy"][plot_indices],
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    q = ax.quiver(
        face_x[plot_indices],
        face_y[plot_indices],
        result["mean_sx"][plot_indices],
        result["mean_sy"][plot_indices],
        magnitude,
        cmap="viridis",
        width=0.002,
        headwidth=4,
    )

    ax.set_xlim(X_LIMITS_M)
    ax.set_ylim(Y_LIMITS_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Time-mean sediment transport\n{scenario_name}")

    cbar = plt.colorbar(q, ax=ax, pad=0.02)
    cbar.set_label(
        r"Mean transport magnitude [m$^3$ bed material s$^{-1}$ m$^{-1}$]"
    )

    fig.tight_layout()
    save_figure(
        fig,
        result["output_dir"] / "mean_transport_vectors",
    )


# %% Cumulative displacement maps
for scenario_name, result in scenario_results.items():
    face_x = result["data"]["face_x"]
    face_y = result["data"]["face_y"]
    wet = result["wet_mask"]

    displacement = np.hypot(
        result["cumulative_sx"],
        result["cumulative_sy"],
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    points = ax.scatter(
        face_x[wet],
        face_y[wet],
        c=displacement[wet],
        s=2,
        cmap="magma",
        vmin=0,
    )

    ax.quiver(
        face_x[wet][::QUIVER_STRIDE],
        face_y[wet][::QUIVER_STRIDE],
        result["cumulative_sx"][wet][::QUIVER_STRIDE],
        result["cumulative_sy"][wet][::QUIVER_STRIDE],
        color="white",
        width=0.0015,
    )

    ax.set_xlim(X_LIMITS_M)
    ax.set_ylim(Y_LIMITS_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Cumulative sediment displacement\n{scenario_name}")

    cbar = plt.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label(
        r"Cumulative displacement [m$^3$ bed material m$^{-1}$]"
    )

    fig.tight_layout()
    save_figure(
        fig,
        result["output_dir"] / "cumulative_displacement",
    )


# %% Potential erosion-deposition maps
x_grid = np.arange(
    X_LIMITS_M[0], X_LIMITS_M[1] + DIAGNOSTIC_DX_M, DIAGNOSTIC_DX_M
)
y_grid = np.arange(
    Y_LIMITS_M[0], Y_LIMITS_M[1] + DIAGNOSTIC_DY_M, DIAGNOSTIC_DY_M
)
x_centres = 0.5 * (x_grid[:-1] + x_grid[1:])
y_centres = 0.5 * (y_grid[:-1] + y_grid[1:])
xx, yy = np.meshgrid(x_centres, y_centres)

for scenario_name, result in scenario_results.items():
    face_x = result["data"]["face_x"]
    face_y = result["data"]["face_y"]
    wet = result["wet_mask"]

    x_bin = np.digitize(face_x, x_grid) - 1
    y_bin = np.digitize(face_y, y_grid) - 1

    valid = (
        wet
        & (x_bin >= 0) & (x_bin < x_centres.size)
        & (y_bin >= 0) & (y_bin < y_centres.size)
    )

    cum_sx_grid = grid_mean(
        result["cumulative_sx"], x_bin, y_bin,
        x_centres.size, y_centres.size, valid,
    )
    cum_sy_grid = grid_mean(
        result["cumulative_sy"], x_bin, y_bin,
        x_centres.size, y_centres.size, valid,
    )

    divergence = (
        np.gradient(cum_sx_grid, DIAGNOSTIC_DX_M, axis=1)
        + np.gradient(cum_sy_grid, DIAGNOSTIC_DY_M, axis=0)
    )

    # cumulative transport [m3/m] -> divergence [m]: potential bed change
    potential_change = -divergence

    fig, ax = plt.subplots(figsize=(10, 4))

    vmax = np.nanpercentile(np.abs(potential_change), 98)

    mesh = ax.pcolormesh(
        x_grid,
        y_grid,
        potential_change,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
    )

    ax.quiver(
        xx[::4, ::4],
        yy[::4, ::4],
        cum_sx_grid[::4, ::4],
        cum_sy_grid[::4, ::4],
        color="black",
        width=0.0015,
    )

    ax.set_xlim(X_LIMITS_M)
    ax.set_ylim(Y_LIMITS_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Potential bed-volume change from transport convergence\n"
        f"{scenario_name}"
    )

    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Potential bed-level change [m] (red = deposition)")

    fig.tight_layout()
    save_figure(
        fig,
        result["output_dir"] / "potential_erosion_deposition",
    )


# %% Cross-section flux time series
for scenario_name, result in scenario_results.items():
    fig, ax = plt.subplots(figsize=(10, 4))

    for boundary_index, x_km in enumerate(BOUNDARY_X_KM):
        ax.plot(
            result["data"]["time"],
            result["boundary_flux_seaward"][:, boundary_index],
            linewidth=0.9,
            label=f"{x_km:.0f} km",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(
        r"Cross-section flux, seaward positive "
        r"[m$^3$ bed material s$^{-1}$]"
    )
    ax.set_title(
        f"Instantaneous sediment flux through x boundaries\n"
        f"{scenario_name}"
    )
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(
        fig,
        result["output_dir"] / "cross_section_flux_timeseries",
    )


# %% Zone cumulative-volume plots
zone_labels = [
    f"{ZONE_EDGES_KM[i]:.0f}-{ZONE_EDGES_KM[i + 1]:.0f} km"
    for i in range(ZONE_EDGES_KM.size - 1)
]

for scenario_name, result in scenario_results.items():
    fig, ax = plt.subplots(figsize=(10, 4))

    for zone_index, zone_label in enumerate(zone_labels):
        ax.plot(
            result["data"]["time"],
            result["cumulative_zone_volume"][:, zone_index],
            linewidth=1.2,
            label=zone_label,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(
        r"Cumulative net sediment volume "
        r"[m$^3$ bed material] (+ = deposition)"
    )
    ax.set_title(f"Short-period zone sediment budget\n{scenario_name}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(
        fig,
        result["output_dir"] / "zone_cumulative_volume",
    )


# %% Scenario comparison: final zone budget
final_volumes = {
    scenario: result["cumulative_zone_volume"][-1]
    for scenario, result in scenario_results.items()
}

zone_centres = 0.5 * (ZONE_EDGES_KM[:-1] + ZONE_EDGES_KM[1:])

for flow_condition, scenarios in COMPARISON_PAIRS.items():
    fig, ax = plt.subplots(figsize=(8, 4))

    for scenario_name in scenarios:
        if scenario_name not in final_volumes:
            print(f"   Skipping {scenario_name} in {flow_condition}: "
                  f"not loaded.")
            continue

        ax.plot(
            zone_centres,
            final_volumes[scenario_name],
            marker="o",
            linewidth=1.5,
            color=SCENARIO_COLORS[scenario_name],
            label=scenario_name,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Zone centre [km]")
    ax.set_ylabel(
        r"Final cumulative net volume "
        r"[m$^3$ bed material] (+ = deposition)"
    )
    ax.set_title(
        f"{flow_condition.replace('_', ' ').title()} scenarios:\n"
        "final zone sediment budget"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_figure(
        fig,
        OUTPUT_BASE / f"comparison_{flow_condition}_final_zone_budget",
    )


# %% Export CSV files
for scenario_name, result in scenario_results.items():
    csv_dir = result["output_dir"] / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    times = result["data"]["time"]

    write_csv(
        csv_dir / "boundary_flux_seaward_positive.csv",
        [
            "time",
            *[f"flux_x_{x:.0f}km_m3_bed_per_s" for x in BOUNDARY_X_KM],
        ],
        [
            [str(t), *result["boundary_flux_seaward"][i]]
            for i, t in enumerate(times)
        ],
    )

    write_csv(
        csv_dir / "zone_cumulative_volume.csv",
        [
            "time",
            *[f"zone_{label}_m3_bed_material" for label in zone_labels],
        ],
        [
            [str(t), *result["cumulative_zone_volume"][i]]
            for i, t in enumerate(times)
        ],
    )

    write_csv(
        csv_dir / "final_zone_budget.csv",
        [
            "zone",
            "zone_centre_km",
            "final_cumulative_net_volume_m3_bed_material",
            "final_instantaneous_net_rate_m3_bed_per_s",
        ],
        [
            [
                zone_label,
                zone_centres[zone_index],
                result["cumulative_zone_volume"][-1, zone_index],
                result["net_zone_rate"][-1, zone_index],
            ]
            for zone_index, zone_label in enumerate(zone_labels)
        ],
    )


# %% Cleanup / close open datasets
for scenario_name, result in scenario_results.items():
    if "cache_dataset" in result["data"]:
        result["data"]["cache_dataset"].close()


# %% Finish
print("\n" + "=" * 80)
print("DETAILED-HYDRO SEDIMENT ROUTING AND BUDGET ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory:\n{OUTPUT_BASE}")
print(f"\nTransport variables: {SX_VAR}, {SY_VAR}")
print(f"POROSITY = {POROSITY}, MORFAC = {MORFAC}")
print("\nSign convention:")
print(f"   Model +x points {'seaward' if POSITIVE_X_IS_SEAWARD else 'landward'}"
      f" (POSITIVE_X_IS_SEAWARD = {POSITIVE_X_IS_SEAWARD}).")
print("   Displayed cross-section flux: positive = seaward.")
print("   Zone cumulative volume: positive = convergence = potential deposition.")
print("\nInterpretation:")
print("   Short-period routing budgets over the detailed-hydro period.")
print("   If the runs have a fixed bed, potential bed change is not realised "
      "morphology.")
print("=" * 80)