# %% Detailed-hydro sediment routing and short-period zone budgets
"""
Transport-routing diagnostics and cross-section sediment budgets from
detailed-hydro MAP caches (uses the repository loading/caching functions).

Workflow per scenario
---------------------
1. Load SX_VAR / SY_VAR (default mesh2d_sxtot / mesh2d_sytot) and water depth
   through load_or_update_map_cache_multi (same calls as the pm/n script).
2. Sum over the sediment-fraction dimension (nSedTot), convert to a
   bed-volume-equivalent rate q [m3 bed / s / m] (units read from the file).
3. STREAM through time in blocks (nothing of size n_times x n_faces is kept):
   - time integrals per face (net, gross) via trapezoid weights,
   - flux through the fixed x-boundaries (a sparse sampling-weight matrix),
   - time-mean water depth (wet mask).
4. Zone budgets between consecutive boundaries, maps, CSV files.

How the cumulative transport maps are computed
----------------------------------------------
At every face, with q = (qx, qy) the transport per unit width [m3 bed/s/m]
and t the morphological time (model time * MORFAC):

    net vector   N = integral q dt                  [m3 bed / m]
    net magnitude |N| = hypot(Nx, Ny)               (colour of the "net" map)
    gross        G = integral |q| dt                [m3 bed / m]
    directionality = |N| / G   in [0, 1]

* N is a signed vector integral: ebb/flood and flow reversals CANCEL. It is
  the residual routing. It is a transport per metre width (m3/m), NOT a
  distance in metres (that would need division by an active-layer thickness).
* G does not cancel; it measures total sediment activity.
* |N|/G ~ 1: sediment moves one way; ~ 0: back-and-forth with little net
  movement.
* The integrals are trapezoidal, i.e. exactly np.trapz, implemented as a
  weight vector w so that  integral f dt = w . f  can be accumulated block
  by block. NaN (dry) values count as zero transport.
* Arrows show N averaged over regular coarse cells (spatially uniform, with
  a reference arrow), not every n-th face of the unstructured mesh.

Potential bed-level change
--------------------------
    potential change [m] = -div(N)
where N is already a BED-volume transport (the 1/(1-porosity) factor is
applied when q is converted), so it is NOT divided by (1 - porosity) again.
Faces are averaged onto a regular grid (dry faces count as zero transport,
cells without faces are NaN); cells that are mostly dry are masked.

Channel infilling diagnostic (peak vs mean flow)
------------------------------------------------
Channel footprint (CHANNEL_DEFINITION) inside the zoom domain: "wet" = faces that
are still wet at low water (min water level of the last 12 h minus bed level
> 0.1 m, as in the cross-section script), or "bed_level" = bed level <
CHANNEL_BED_LEVEL_MAX_M. Only the largest connected component is kept. On the (rectilinear) model grid the time-mean transport is put
through an exact finite-volume budget: the net inflow of every channel cell
is the sum of the edge fluxes (edge flux = mean of the two neighbouring
cell-centre transports). Per 1-km bin the net inflow of the channel cells is
split into
    along-channel convergence : channel cells exchanging with channel cells
                                in the neighbouring bins
    lateral supply            : exchange with non-channel cells (bars/banks)
and divided by the channel area -> potential channel bed-level change rate
[mm/day] (+ = infilling). This is the quantity that shows WHERE and HOW the
channel is refilled at peak vs mean flow.

Validation and explanation of the channel infilling
---------------------------------------------------
- HIS check (HIS_VALIDATION): the model's own cumulative bed-load through the
  observation cross-sections gives the whole-width zone budget as a
  difference of two stored numbers. It is printed against the MAP-based zone
  budget. NOTE: the HIS sections are not at the zone edges (up to ~0.4 km
  off), so HIS zones are shifted. The MAP reconstruction does not close
  either. The whole-width zone budget that is plotted and exported therefore
  comes from ZONE_BUDGET_SOURCE (default: measured bed-level change on the
  exact zone edges); MAP is drawn dotted for reference.
  HIS cannot separate channels from bars (the sections span the whole width).
- Measured channel bed change: mesh2d_mor_bl (last minus first output) over
  the channel cells, compared with the change predicted from the flux
  divergence (print_channel_summary, channel_local_supply_vs_capacity row 4).
- Local supply versus local capacity (channel_local_supply_vs_capacity):
  the modelled flux and the MPM capacity through the CHANNEL part and through
  the BARS/BANKS part of each section, the along-channel deposition implied by
  their change in the flow direction, and the lateral supply from bars/banks
  into the channel next to the measured and predicted bed change. The capacity is the net (direction of the model transport)
  capacity, so tides cancel as in the flux.

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
import warnings

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
from functions.F_loaddata import get_stitched_map_run_paths, load_and_cache_scenario


# %% BASE PATHS (same construction as the sensitivity script)
DISCHARGE = 500

MODEL_ROOT = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
)
BASE_PATH = MODEL_ROOT / "Model_Output" / f"Q{DISCHARGE}"
RUN_DIR = BASE_PATH / "detailed-hydro-run"

VARIABILITY_MAP = get_variability_map(DISCHARGE)

ASSESSMENT_DIR = BASE_PATH / "cached_data"
ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
TIMED_OUT_DIR = BASE_PATH / "timed-out"

OUTPUT_BASE = RUN_DIR / "output_plots" / "sediment_budgets"


# %% SCENARIOS
SCENARIO_FOLDERS = {
    "constant": "dhr_01_Qr500_pm1_n0.9724783",
    "pm3_n5_mean_flow": "dhr_03_Qr500_pm3_n5_mean.964903",
    "pm5_n1_mean_flow": "dhr_05_Qr500_pm5_n1_mean.964908",
    "pm3_n5_peak_flow": "dhr_03_Qr500_pm3_n5_peak.964907",
    "pm5_n1_peak_flow": "dhr_05_Qr500_pm5_n1_peak.965221",
}
SCENARIOS = {name: RUN_DIR / folder for name, folder in SCENARIO_FOLDERS.items()}

COMPARISON_PAIRS = {
    "mean_flow": ["constant", "pm3_n5_mean_flow", "pm5_n1_mean_flow"],
    "peak_flow": ["constant", "pm3_n5_peak_flow", "pm5_n1_peak_flow"],
}


# %% SPATIAL DOMAIN & CACHE SETTINGS
# Detailed-hydro zoom domain (also the extent of all maps).
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
ZONE_EDGES_KM = BOUNDARY_X_KM  # zones lie between consecutive boundaries

# Number of y samples used to integrate q_x through each x boundary.
N_Y_SAMPLES = 1000

# y-range of the cross-section budget (None -> Y_LIMITS_M). Must cover the
# full width over which sediment moves.
BUDGET_Y_RANGE_M = None
BUDGET_Y_M = tuple(BUDGET_Y_RANGE_M) if BUDGET_Y_RANGE_M else Y_LIMITS_M

# Maximum distance allowed when sampling a face onto a boundary line.
# (grid cells are 85 m x 50 m)
MAX_BOUNDARY_SAMPLE_DISTANCE_M = 250.0

# Only faces inside the analysis domain (+ this margin) are read from disk.
FACE_MARGIN_M = 500.0

# Minimum mean water depth for a face to count as wet.
MIN_WATER_DEPTH_M = 0.05

# Regular grid for the potential-bed-change map.
DIAGNOSTIC_DX_M = 250.0
DIAGNOSTIC_DY_M = 250.0
MIN_WET_FRACTION = 0.5  # mask grid cells that are less wet than this


# %% TRANSPORT SETTINGS
# NOTE: sxtot/sytot are the TOTAL load (bed + suspended) at cell centres,
# reconstructed from the edge transports (mesh2d_sbt / sbn = bed load,
# mesh2d_sst / ssn = suspended load). With Meyer-Peter & Muller there is no
# suspended load (the formula is imposed as bed load, manual 8.5.5), so
# sxtot/sytot are bed load. Quick check on a map file:
#   abs(ds["mesh2d_ssxcum"]).max() should be ~0.
# (mesh2d_sbcx/sbcy do not exist in these files; mesh2d_sbxcum/sbycum are
# time-averaged without a time axis and cannot be used for time series.)
SX_VAR = "mesh2d_sxtot"
SY_VAR = "mesh2d_sytot"
WATER_DEPTH_VAR = "mesh2d_waterdepth"

# Fallback transport units, used only if the 'units' attribute of the
# variable is unrecognised. Options: "volume_bed", "volume_solid", "mass".
# The attribute is checked first: kg-based -> mass, m3-based -> volume_solid
# (unless this is set to "volume_bed").
TRANSPORT_UNITS = "volume_solid"

# Solid sediment density, used only for mass-based transport.
SEDIMENT_DENSITY_KG_M3 = 2650.0

# Porosity converts solid transport volume to bed-volume equivalent.
# From the sediment file: CDryB = 1600, RhoSol = 2650 -> porosity 0.396.
DRY_BED_DENSITY_KG_M3 = 1600.0
POROSITY = 1.0 - DRY_BED_DENSITY_KG_M3 / SEDIMENT_DENSITY_KG_M3
BED_VOLUME_FACTOR = 1.0 / (1.0 - POROSITY)     # = RhoSol / CDryB

# Morphological acceleration factor of the detailed-hydro runs
# (morphology file: MorFac = 1). Multiplies all time integrals.
MORFAC = 1.0

# Direction of the model x-axis. In the HIS buffer script the river is at
# ~44 km and distance is counted from the sea, so +x points LANDWARD.
# Only affects the sign of the displayed cross-section flux, not the budgets.
POSITIVE_X_IS_SEAWARD = False
SEAWARD_SIGN = 1.0 if POSITIVE_X_IS_SEAWARD else -1.0


# %% TIME SETTINGS
TIME_START_INDEX = 0
TIME_END_INDEX = None
TIME_CHUNK = 50  # timesteps per block; memory ~ TIME_CHUNK x n_faces x 8 B


# %% OUTPUT AND PLOTTING
DPI = 300
SHOW_FIGURES = False

SCENARIO_COLORS = {
    "constant": "#7f7f7f",
    "pm3_n5_mean_flow": plt.cm.Blues(0.55),
    "pm5_n1_mean_flow": plt.cm.Blues(0.92),
    "pm3_n5_peak_flow": plt.cm.Oranges(0.55),
    "pm5_n1_peak_flow": plt.cm.Oranges(0.92),
}

# Quiver settings: arrows are the transport vector averaged over regular
# cells of QUIVER_CELL_M; a reference value of the key is drawn
# ARROW_LENGTH_M long (data units).
QUIVER_CELL_M = 1000.0
ARROW_LENGTH_M = 800.0

# Directionality (net/gross) is masked where gross transport is below this
# fraction of its 99th percentile (the ratio is meaningless there).
DIRECTIONALITY_MIN_GROSS_FRACTION = 0.01


# %% CHANNEL INFILLING DIAGNOSTIC SETTINGS
# mesh2d_mor_bl is the time-varying bed level; mesh2d_flowelem_bl has NO time
# axis in the map files (it is a static bed level), so BED_LEVEL_AT would be
# ignored with it. It is only used as a fallback if mor_bl is not available.
BED_LEVEL_VAR = "mesh2d_mor_bl"
BED_LEVEL_FALLBACK = "mesh2d_flowelem_bl"
BED_LEVEL_AT = "first"           # bed level for the footprint: "first" or "last" step.
# "first" = the channel as it is at the start of the run, i.e. a footprint chosen
# without knowing the outcome. "last" selects cells by where they ENDED, which biases
# the measured bed change of the footprint in a reach that changes a lot.
# Channel footprint: bed level below this value (same as the p5/p95 profiles),
# inside X_LIMITS_M / Y_LIMITS_M.
CHANNEL_BED_LEVEL_MAX_M = 2.2
# How the channel footprint is defined:
#   "wet"       = the area that is actually a channel in the model: faces that are
#                 still wet at LOW water (water level - bed level > WET_DEPTH_THRESHOLD_M).
#                 Same idea as the cross-section script (compute_channel_widths).
#   "bed_level" = bed level < CHANNEL_BED_LEVEL_MAX_M (fixed elevation cut).
# "wet" needs mesh2d_s1; if it is not in the data the code falls back to "bed_level".
CHANNEL_DEFINITION = "wet"
WATER_LEVEL_VAR = "mesh2d_s1"
WET_DEPTH_THRESHOLD_M = 0.1      # wet = water level - bed level > this [m]
WET_WINDOW_H = 12.0              # hours of output that define the water level
WET_WINDOW_AT = "last"           # "last" (as in the cross-section script) | "first"
WET_LEVEL_STAT = "min"           # "min" = wet at low water (channel only);
                                 # "max" = wet at high water (channel + flooded bars)
# Keep only the largest connected channel component. Isolated pits / holes with a
# bed level below the threshold are excluded from BOTH the channel and the bars:
# a small depression fills quickly, which is not channel infilling.
CHANNEL_MAIN_ONLY = True
# Footprint per run: a peak-flow run uses the footprint of the mean-flow run of
# its scenario (both start from the same bed). With its own footprint the
# "channel" of a peak run grows by ~50% (high water levels flood low bars), so
# channel/bar budgets of mean and peak runs would not be comparable. Runs not
# listed use their own footprint. The source run must be processed first.
CHANNEL_FOOTPRINT_FROM = {
    "pm3_n5_peak_flow": "pm3_n5_mean_flow",
    "pm5_n1_peak_flow": "pm5_n1_mean_flow",
}
FOOTPRINTS = {}   # filled while processing: name -> (face_x, face_y, channel, text)
# Zone bar chart (channel_infilling_by_zone): "measured" = bed-level change of
# the channel cells (mesh2d_mor_bl); "predicted" = flux divergence of the
# time-mean transport (at the noise level of the reconstruction, ~0.1 mm/day).
ZONE_RATE_SOURCE = "measured"
CHANNEL_BIN_M = 1000.0           # along-estuary bin length of the profiles
CHANNEL_OUTLINE_COLOR = "limegreen"   # outline of the footprint in the maps

# Transport-versus-elevation figure
BED_BIN_M = 0.5                  # bed-level bin width
SLOPE_MIN = 1e-3                 # bed slope below which "down-slope" is undefined
MIN_FACES_PER_BIN = 20           # bins with fewer faces are left blank

# Time-weighted budget figure: share of the hydrograph spent in the "peak-flow"
# state (the rest is treated as the "mean-flow" state).
# Per-timestep channel infilling (figure + GIF). Footprint = bed level at
# BED_LEVEL_AT, kept fixed so frames are comparable.
TIMESTEP_PROFILES = True
GIF_STEP = 1             # use every n-th output step as a GIF frame
GIF_FPS = 6
GIF_DPI = 110
FLUX_X_KM = (20.0, 45.0)   # cross-sections shown in the GIF flux panel
DEPOSITION_INTERVAL_KM = 2.0   # length of the stretches used for the along-flow change

# %% TRANSPORT CAPACITY (Meyer-Peter & Muller, D-Morphology manual eq. 8.213-8.216)
#   S = 8 a D50 sqrt(Delta g D50) (mu*theta - xi*theta_cr)^1.5   [m3 solid /s/m]
#   theta = tau / (rho_w g Delta D50)   (= (u/C)^2 / (Delta D50))
#   mu    = min((C / C_g90)^1.5, 1),  C_g90 = 18 log10(12 h / D90)
# Follows the manual literally: theta from the flow-velocity magnitude and the
# Chezy coefficient, eq. 8.214  theta = (u/C)^2 / (Delta D50), i.e.
# tau = rho_w g (u/C)^2. THETA_SOURCE = "taus" / "tausmax" uses the output bed
# shear stress instead (a cross-check; "tausmax" = stress used for morphology).
# The result is the capacity BEFORE bed-slope correction (Sb'' in the manual);
# the model's transport differs from it by bed-slope and fixed-layer effects.
CAPACITY_ANALYSIS = True
THETA_SOURCE = "velocity"        # "velocity" (manual) | "taus" | "tausmax"
UCMAG_VAR = "mesh2d_ucmag"       # flow velocity magnitude [m/s]
TAU_VAR = "mesh2d_taus"          # total bed shear stress [N/m2]
TAUMAX_VAR = "mesh2d_tausmax"    # bed shear stress for morphology [N/m2]
CHEZY_VAR = "mesh2d_czs"         # Chezy coefficient [m^0.5/s]
THETA_VARS = {"velocity": UCMAG_VAR, "taus": TAU_VAR, "tausmax": TAUMAX_VAR}
# Fixed-layer reduction (manual 8.4.4): transport is multiplied by
# min(DPSED / THRESH, 1). Morphology file: Thresh = 0.05 m, but the initial
# sediment layer is 15 m thick, so the factor is 1 unless the layer is eroded
# to < 5 cm. None = ignore; set 0.05 to include it (needs mesh2d_dpsed).
DPSED_VAR = "mesh2d_dpsed"
MPM_FIXED_LAYER_THRESH_M = None
# Not included in the recomputed capacity (they change the model transport
# relative to it): bed-slope correction (ISlope = 3, Koch & Flokstra:
# magnitude x (1 - AlfaBs dz/ds), plus a direction rotation), spiral flow
# (Espir = 1, direction only). Expect model/capacity to scatter around 1.
# From the sediment (.sed) file: one "sand" fraction, TraFrm = 2 (MPM),
# SedDia = 2e-4 m, ACal = 1, RhoSol = 2650, IniSedThick = 15 m.
MPM_D50_M = 2.0e-4           # SedDia
MPM_D90_M = 2.0e-4           # single fraction: D90 of the bed mixture = SedDia
MPM_ALPHA = 1.0              # ACal
MPM_THETA_CR = 0.047         # critical Shields parameter
MPM_XI = 1.0                 # hiding/exposure factor (1 for one fraction)
RHO_WATER = 1000.0           # [kg/m3] (use 1025 for saline water if set so)
GRAVITY = 9.81
CAPACITY_X_KM = np.arange(20.0, 45.0 + 0.5, 1.0)   # sections for the profile
SUPPLY_X_KM = 45.0           # section taken as the sediment inflow

# OPTIONAL: leave None to skip the weighted figure. To enable, give the real
# fractions of your hydrographs, e.g. {"pm3_n5": 0.15, "pm5_n1": 0.10}.
PEAK_TIME_FRACTION = None


# %% HIS CROSS-SECTION CHECK (measured fluxes through the model's own sections)
# The HIS file holds, per observation cross-section, the CUMULATIVE bed-load
# transport the model itself accumulated on the flow links (the same fluxes
# that update the bed). The budget of a stretch is therefore a difference of
# two stored numbers (no sampling, no integration error). It is compared
# with the MAP-based zone budget: this validates the reconstruction of the
# cross-section flux from the cell-centre transports and the y-range
# BUDGET_Y_M. The HIS sections span the whole width, so this check cannot
# separate channels from bars; for the channel footprint the measured
# bed-level change (mesh2d_mor_bl, first vs last output) is the reference.
HIS_VALIDATION = True
HIS_FILE_NAME = "FlowFM_0000_his.nc"            # in <scenario folder>/output/
HIS_BEDLOAD_VAR = "cross_section_bedload_sediment_transport"       # cumulative [kg]
HIS_SUSPENDED_VAR = "cross_section_suspended_sediment_transport"   # cumulative [kg]
HIS_DISCHARGE_VAR = "cross_section_discharge"   # [m3/s], gives the seaward sign
HIS_INCLUDE_SUSPENDED = True                    # MPM: should add ~0
HIS_KM_TOL = 0.5                                # section must lie this close to a zone edge

# Warn if a matched HIS section lies further than this from its zone edge. The
# sections named "km35", "km40", ... are NOT at those x (e.g. km35 at 35.31 km,
# km40 at 40.41 km), so the HIS zones are shifted by up to ~0.4 km. That alone
# reverses zones: pm5_n1 peak, 35-40 km: bed change -1702 m3 on the nominal
# edges, +948 m3 on the HIS edges (= the HIS value, so HIS itself is right).
HIS_EDGE_WARN_KM = 0.1

# Source of the whole-width zone budget in the scenario comparison figure,
# zone_cumulative_volume and final_zone_budget.csv:
#   "bed" = measured bed-level change x cell area per zone on the exact zone
#           edges (mesh2d_mor_bl, full width): the model's realised morphology
#   "map" = fluxes reconstructed from the cell-centre MAP transports; does not
#           close (errors of several hundred m3 per zone edge)
# HIS is not offered: its sections are not at the zone edges (see above).
# "bed" falls back to "map" (with a warning) if the bed level has no time axis.
ZONE_BUDGET_SOURCE = "bed"


# %% Helper functions: data access
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


def read_block(da, face_dimension, start, end, fill_nan=True):
    """Read da[start:end] as a (time, face) float array.

    Extra dimensions (sediment fractions) are summed. With fill_nan, NaN
    (dry / no data) becomes 0, i.e. no transport.
    """
    block = da.isel(time=slice(start, end))
    extra_dims = [d for d in block.dims if d not in ("time", face_dimension)]
    if extra_dims:
        block = block.sum(dim=extra_dims, skipna=True)
    out = np.asarray(block.transpose("time", face_dimension).values, dtype=float)
    return np.nan_to_num(out, copy=False) if fill_nan else out


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


# %% Helper functions: numerics
def trapezoid_weights(t):
    """Weights w such that  integral f dt  (trapezoid rule) = w @ f.

    Identical to np.trapz(f, t) but lets the integral be accumulated block
    by block while streaming through the file.
    """
    w = np.zeros(t.size)
    half = 0.5 * np.diff(t)
    w[:-1] += half
    w[1:] += half
    return w


def domain_face_indices(face_x, face_y):
    """Indices of faces inside the analysis domain (+ margin)."""
    x_lo = min(X_LIMITS_M[0], BOUNDARY_X_M.min()) - FACE_MARGIN_M
    x_hi = max(X_LIMITS_M[1], BOUNDARY_X_M.max()) + FACE_MARGIN_M
    y_lo = min(Y_LIMITS_M[0], BUDGET_Y_M[0]) - FACE_MARGIN_M
    y_hi = max(Y_LIMITS_M[1], BUDGET_Y_M[1]) + FACE_MARGIN_M
    keep = (
        (face_x >= x_lo) & (face_x <= x_hi)
        & (face_y >= y_lo) & (face_y <= y_hi)
    )  # NaN coordinates compare False and are dropped
    return np.flatnonzero(keep)


def build_boundary_weights(face_x, face_y, x_positions=None):
    """Sampling-weight matrix W (n_faces x n_boundaries).

    The flux through boundary b is  flux_b(t) = sum_f qx_f(t) * W[f, b]:
    N_Y_SAMPLES points (bin centres, spacing dy) along the boundary are
    mapped to their nearest face; W[f, b] = (number of samples mapped to f)
    * dy is the effective width that face represents. One matrix product per
    time block then replaces the per-boundary, per-timestep sampling.
    x_positions [m] defaults to BOUNDARY_X_M. The same matrix integrates any
    face field over y (scalar fields such as the transport capacity too).
    """
    if x_positions is None:
        x_positions = BOUNDARY_X_M
    y_lo, y_hi = BUDGET_Y_M
    dy = (y_hi - y_lo) / N_Y_SAMPLES
    y_samples = y_lo + (np.arange(N_Y_SAMPLES) + 0.5) * dy

    tree = cKDTree(np.column_stack((face_x, face_y)))
    weights = np.zeros((face_x.size, np.size(x_positions)))

    for b, x_boundary in enumerate(x_positions):
        points = np.column_stack(
            (np.full(N_Y_SAMPLES, x_boundary), y_samples)
        )
        distances, face_idx = tree.query(points)
        ok = distances <= MAX_BOUNDARY_SAMPLE_DISTANCE_M

        coverage = ok.mean()
        if coverage < 0.9:
            print(f"  [WARNING] Boundary x = {x_boundary / 1000:.1f} km: only "
                  f"{coverage:.0%} of the y samples hit a face within "
                  f"{MAX_BOUNDARY_SAMPLE_DISTANCE_M:.0f} m "
                  f"(is BUDGET_Y_RANGE_M inside the domain?).")

        weights[:, b] = np.bincount(
            face_idx[ok], minlength=face_x.size
        ) * dy

    return weights


def mpm_capacity(tau, chezy, depth, layer_factor=1.0):
    """Meyer-Peter & Muller transport capacity per face [m3 bed /s/m].

    Same formula as D-Morphology (manual 8.5.5), evaluated from the local
    bed shear stress tau [N/m2], Chezy coefficient [m^0.5/s] and water depth
    [m]. Returns the MAGNITUDE (no direction, no bed-slope correction),
    converted from solid to bed-volume equivalent like the model transport.
    """
    delta = (SEDIMENT_DENSITY_KG_M3 - RHO_WATER) / RHO_WATER
    ok = (depth >= MIN_WATER_DEPTH_M) & (chezy > 0) & (tau > 0)
    theta = np.where(ok, tau, 0.0) / (RHO_WATER * GRAVITY * delta * MPM_D50_M)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_g90 = 18.0 * np.log10(12.0 * np.maximum(depth, 1e-6) / MPM_D90_M)
        mu = np.where(c_g90 > 0, np.minimum((chezy / c_g90) ** 1.5, 1.0), 1.0)
    excess = np.maximum(mu * theta - MPM_XI * MPM_THETA_CR, 0.0)
    solid = (8.0 * MPM_ALPHA * MPM_D50_M
             * np.sqrt(delta * GRAVITY * MPM_D50_M) * excess ** 1.5)
    return np.where(ok, solid, 0.0) * layer_factor * BED_VOLUME_FACTOR


def compute_zone_budgets(boundary_flux, time_seconds):
    """Zone convergence rate and cumulative volume (positive = deposition).

    boundary_flux is in the model +x direction and boundaries increase in x:
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


def derive_fields(net_x, net_y, gross, duration):
    """Derived per-face fields from the time integrals."""
    net_mag = np.hypot(net_x, net_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        directionality = np.where(
            gross > 0, np.minimum(net_mag / gross, 1.0), np.nan
        )
    return {
        "net_x": net_x,
        "net_y": net_y,
        "net_mag": net_mag,
        "gross": gross,
        "mean_abs": gross / duration,        # time-mean |q| [m3 bed/s/m]
        "directionality": directionality,
        "mean_x": net_x / duration,
        "mean_y": net_y / duration,
    }


def regular_edges(limits, step):
    return np.arange(limits[0], limits[1] + 0.5 * step, step)


def bin_faces(face_x, face_y, x_edges, y_edges, mask):
    """Assign faces to regular grid cells. Returns (ix, iy, valid)."""
    nx, ny = x_edges.size - 1, y_edges.size - 1
    ix = np.digitize(face_x, x_edges) - 1
    iy = np.digitize(face_y, y_edges) - 1
    valid = mask & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return ix, iy, valid


def grid_mean(values, ix, iy, nx, ny, valid):
    """Average face values into a regular (ny, nx) grid (NaN if empty)."""
    ok = valid & np.isfinite(values)
    flat = iy[ok] * nx + ix[ok]
    total = np.bincount(flat, weights=values[ok], minlength=nx * ny)
    count = np.bincount(flat, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / count, np.nan)
    return out.reshape(ny, nx)


def coarse_vectors(res, vec_x, vec_y, cell_m=QUIVER_CELL_M):
    """Average a face vector field over regular cells (wet faces only).

    Returns cell-centre coordinates and vector components of the cells that
    contain data, so arrows are spatially uniform.
    """
    xe = regular_edges(X_LIMITS_M, cell_m)
    ye = regular_edges(Y_LIMITS_M, cell_m)
    nx, ny = xe.size - 1, ye.size - 1
    ix, iy, valid = bin_faces(res["face_x"], res["face_y"], xe, ye, res["wet"])
    u = grid_mean(vec_x, ix, iy, nx, ny, valid)
    v = grid_mean(vec_y, ix, iy, nx, ny, valid)
    xc, yc = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
    ok = np.isfinite(u) & np.isfinite(v)
    return xc[ok], yc[ok], u[ok], v[ok]


def potential_bed_change(res):
    """Potential bed-level change [m] = -div(net transport), on a grid.

    Net transport is already a bed-volume transport, so no further porosity
    factor. Dry faces carry zero transport; cells with no faces are NaN;
    cells that are mostly dry are masked.
    """
    xe = regular_edges(X_LIMITS_M, DIAGNOSTIC_DX_M)
    ye = regular_edges(Y_LIMITS_M, DIAGNOSTIC_DY_M)
    nx, ny = xe.size - 1, ye.size - 1

    everywhere = np.ones(res["face_x"].size, dtype=bool)
    ix, iy, valid = bin_faces(res["face_x"], res["face_y"], xe, ye, everywhere)

    qx = grid_mean(res["net_x"], ix, iy, nx, ny, valid)
    qy = grid_mean(res["net_y"], ix, iy, nx, ny, valid)
    wet_fraction = grid_mean(res["wet"].astype(float), ix, iy, nx, ny, valid)

    divergence = (
        np.gradient(qx, DIAGNOSTIC_DX_M, axis=1)
        + np.gradient(qy, DIAGNOSTIC_DY_M, axis=0)
    )
    change = -divergence
    change[~(wet_fraction >= MIN_WET_FRACTION)] = np.nan
    return xe, ye, change


# %% Helper functions: channel infilling budget
def native_grid(face_x, face_y):
    """Map faces onto the cells of a rectilinear grid.

    Returns (ix, iy, ux, uy, dx, dy): column/row index per face, the unique
    column/row coordinates and per-column / per-row cell sizes. Returns None
    if the faces do not form a rectilinear grid (e.g. a curvilinear mesh).
    """
    ux, ix = np.unique(np.round(face_x, 1), return_inverse=True)
    uy, iy = np.unique(np.round(face_y, 1), return_inverse=True)
    rectilinear = (
        ux.size * uy.size <= 1.5 * face_x.size
        and np.unique(iy * ux.size + ix).size == face_x.size
    )
    if not rectilinear:
        return None
    return ix, iy, ux, uy, np.gradient(ux), np.gradient(uy)


def channel_infilling_profiles(res, qx_face=None, qy_face=None, grid=None):
    """Along-estuary budget of the transport in the channel footprint.

    By default the time-mean transport is used; pass qx_face / qy_face
    (per face, bed-volume units) for a single timestep, and a precomputed
    `grid` (from native_grid) to avoid recomputing it every time.

    Finite-volume budget on the native grid: the flux through a cell edge is
    the mean of the two neighbouring cell-centre transports (outer edges use
    the edge cell's own value, i.e. open boundaries), the net inflow of a cell
    is the sum of its four edge fluxes. Interior edges between cells of the
    same set cancel exactly, so for a bin
        net inflow of channel cells = along-channel + lateral,
    where lateral = flux across edges to non-channel neighbours.

    Returns a dict of per-bin arrays (None if no bed level / not rectilinear).
    Units: flux [m3 bed/s], rates [mm/day of bed level over the channel area].
    """
    if res["bed_level"] is None:
        return None
    if grid is None:
        grid = native_grid(res["face_x"], res["face_y"])
    if grid is None:
        print("  [WARNING] Faces do not form a rectilinear grid; "
              "channel budget skipped.")
        return None
    ix, iy, ux, uy, dx, dy = grid
    ny, nx = uy.size, ux.size

    def to_grid(values, fill):
        out = np.full((ny, nx), fill, dtype=float)
        out[iy, ix] = values
        return out

    qx = to_grid(res["mean_x"] if qx_face is None else qx_face, 0.0)  # [m3 bed/s/m]
    qy = to_grid(res["mean_y"] if qy_face is None else qy_face, 0.0)  # missing cells = 0
    bed = to_grid(res["bed_level"], np.nan)

    # --- bins along x and the channel footprint ---
    bin_edges = regular_edges(X_LIMITS_M, CHANNEL_BIN_M)
    nb = bin_edges.size - 1
    col_bin = np.digitize(ux, bin_edges) - 1
    col_bin[(col_bin < 0) | (col_bin >= nb)] = -1
    row_in = (uy >= Y_LIMITS_M[0]) & (uy <= Y_LIMITS_M[1])
    # Footprint used to classify edges: the whole channel network, NOT clipped
    # to the zoom window (clipping would turn along-channel flux across the
    # window edge into fake "lateral" supply). Sums are restricted to the
    # window afterwards.
    if res.get("channel_raw") is not None:
        channel_all = to_grid(res["channel_raw"].astype(float), 0.0) > 0.5
    else:
        with np.errstate(invalid="ignore"):
            channel_all = bed < CHANNEL_BED_LEVEL_MAX_M
    excluded = np.zeros_like(channel_all)
    if CHANNEL_MAIN_ONLY:
        from scipy import ndimage
        labels, n_comp = ndimage.label(channel_all, structure=np.ones((3, 3)))
        if n_comp > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            main = labels == int(np.argmax(sizes))
            excluded = channel_all & ~main
            channel_all = main
    window = (col_bin >= 0)[None, :] & row_in[:, None]
    channel = channel_all & window

    # --- edge fluxes (open outer boundaries) and cell convergence ---
    qxp = np.pad(qx, 1, mode="edge")
    qyp = np.pad(qy, 1, mode="edge")
    fx = 0.5 * (qxp[1:-1, :-1] + qxp[1:-1, 1:]) * dy[:, None]   # (ny, nx+1)
    fy = 0.5 * (qyp[:-1, 1:-1] + qyp[1:, 1:-1]) * dx[None, :]   # (ny+1, nx)
    conv = fx[:, :-1] - fx[:, 1:] + fy[:-1, :] - fy[1:, :]      # + = deposition

    # --- inflow from non-channel neighbours (lateral supply) ---
    chp = np.pad(channel_all, 1, mode="edge")
    left, right = chp[1:-1, :-1], chp[1:-1, 1:]
    down, up = chp[:-1, 1:-1], chp[1:, 1:-1]
    lateral = (
        np.where(right & ~left, fx, 0.0)[:, :-1]
        + np.where(left & ~right, -fx, 0.0)[:, 1:]
        + np.where(up & ~down, fy, 0.0)[:-1, :]
        + np.where(down & ~up, -fy, 0.0)[1:, :]
    )

    # --- per-bin sums over channel cells ---
    bins = np.broadcast_to(col_bin[None, :], (ny, nx))[channel]
    area = np.bincount(bins, weights=(dy[:, None] * dx[None, :])[channel],
                       minlength=nb)
    total = np.bincount(bins, weights=conv[channel], minlength=nb)
    lat = np.bincount(bins, weights=lateral[channel], minlength=nb)
    n_cells = np.bincount(bins, minlength=nb)
    # same budget for ALL cells of the window (channel + non-channel), used to
    # check the channel budget against the cross-section budget
    wbins = np.broadcast_to(col_bin[None, :], (ny, nx))[window]
    conv_window = np.bincount(wbins, weights=conv[window], minlength=nb)
    bed_mean = np.bincount(bins, weights=bed[channel], minlength=nb)

    # net flux through the channel / non-channel part of every column,
    # averaged over the columns of a bin
    flux_channel = (np.where(channel, qx, 0.0) * dy[:, None]).sum(axis=0)
    flux_other = (np.where(~channel_all & ~excluded & window, qx, 0.0) * dy[:, None]).sum(axis=0)
    cols = col_bin >= 0
    n_cols = np.bincount(col_bin[cols], minlength=nb)
    flux_channel = np.bincount(col_bin[cols], weights=flux_channel[cols],
                               minlength=nb) / np.maximum(n_cols, 1)
    flux_other = np.bincount(col_bin[cols], weights=flux_other[cols],
                             minlength=nb) / np.maximum(n_cols, 1)

    # --- time-mean quantities only (not for single timesteps) ---
    # net transport CAPACITY through the channel part of every column (the
    # capacity vector points the way the model transports, so ebb/flood
    # cancel like in the flux), and the MEASURED bed-level change of the
    # channel cells (last minus first output of mesh2d_mor_bl).
    cap_channel = None
    cap_other = None
    measured = None
    if qx_face is None and res.get("cap_net_x_mean") is not None:
        capx = to_grid(res["cap_net_x_mean"], 0.0)
        cap_channel = (np.where(channel, capx, 0.0) * dy[:, None]).sum(axis=0)
        cap_channel = np.bincount(col_bin[cols], weights=cap_channel[cols],
                                  minlength=nb) / np.maximum(n_cols, 1)
        cap_other = (np.where(~channel_all & ~excluded & window, capx, 0.0)
                     * dy[:, None]).sum(axis=0)
        cap_other = np.bincount(col_bin[cols], weights=cap_other[cols],
                                minlength=nb) / np.maximum(n_cols, 1)
    if (qx_face is None and res.get("bed_change") is not None
            and res.get("duration_model_s", 0) > 0):
        dz = to_grid(res["bed_change"], np.nan)
        ok_dz = channel & np.isfinite(dz)
        dz_bins = np.broadcast_to(col_bin[None, :], (ny, nx))[ok_dz]
        cell_area = (dy[:, None] * dx[None, :])[ok_dz]
        dz_volume = np.bincount(dz_bins, weights=dz[ok_dz] * cell_area,
                                minlength=nb)                       # [m3]
        dz_area = np.bincount(dz_bins, weights=cell_area, minlength=nb)
        # same for the bars/banks (non-channel cells of the strip)
        bar_ok = ~channel_all & ~excluded & window & np.isfinite(dz)
        bar_bins = np.broadcast_to(col_bin[None, :], (ny, nx))[bar_ok]
        bar_cell_area = (dy[:, None] * dx[None, :])[bar_ok]
        bar_volume = np.bincount(bar_bins, weights=dz[bar_ok] * bar_cell_area,
                                 minlength=nb)
        bar_area = np.bincount(bar_bins, weights=bar_cell_area, minlength=nb)
        measured = (dz_volume, dz_area, bar_volume, bar_area)

    to_mm_day = MORFAC * 86400.0 * 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = lambda v: np.where(area > 0, v / area * to_mm_day, np.nan)
        mean_bed = np.where(n_cells > 0, bed_mean / n_cells, np.nan)

    out = {
        "x_km": 0.5 * (bin_edges[:-1] + bin_edges[1:]) / 1000.0,
        "flux_channel_seaward": SEAWARD_SIGN * flux_channel,
        "flux_other_seaward": SEAWARD_SIGN * flux_other,
        "rate_total_mm_day": rate(total),
        "rate_lateral_mm_day": rate(lat),
        "rate_along_mm_day": rate(total - lat),
        "conv_total_m3_s": total,
        "conv_window_m3_s": conv_window,
        "conv_lateral_m3_s": lat,
        "area_m2": area,
        "channel_width_m": area / CHANNEL_BIN_M,
        "definition": res.get("channel_text"),
        "bed_level_mean_m": mean_bed,
        "area_pits_m2": np.bincount(
            np.broadcast_to(col_bin[None, :], (ny, nx))[excluded & window],
            weights=(dy[:, None] * dx[None, :])[excluded & window],
            minlength=nb),
    }

    # Along-channel deposition implied by the change of the channel flux (or
    # of the channel capacity) in the flow direction. F_s is seaward-positive
    # and x counts from the sea (+x landward when POSITIVE_X_IS_SEAWARD is
    # False), so more flux entering from upstream than leaving = deposition.
    x_m = out["x_km"] * 1000.0
    width = out["channel_width_m"]

    def along_rate(f_seaward):
        with np.errstate(invalid="ignore", divide="ignore"):
            grad = np.gradient(f_seaward, x_m)
            return np.where(width > 0, -SEAWARD_SIGN * grad / width * to_mm_day,
                            np.nan)

    out["dep_flux_channel_mm_day"] = along_rate(out["flux_channel_seaward"])
    if cap_channel is not None:
        out["cap_channel_seaward"] = SEAWARD_SIGN * cap_channel
        out["cap_other_seaward"] = SEAWARD_SIGN * cap_other
        out["dep_cap_channel_mm_day"] = along_rate(out["cap_channel_seaward"])
        with np.errstate(invalid="ignore", divide="ignore"):
            out["flux_over_cap_channel"] = np.where(
                out["cap_channel_seaward"] > 0,
                out["flux_channel_seaward"] / out["cap_channel_seaward"], np.nan)
    if measured is not None:
        dz_volume, dz_area, bar_volume, bar_area = measured
        days = res["duration_model_s"] / 86400.0
        with np.errstate(invalid="ignore", divide="ignore"):
            out["measured_mm_day"] = np.where(
                dz_area > 0, dz_volume / dz_area * 1000.0 / days, np.nan)
        out["measured_m3_s"] = dz_volume / res["duration_model_s"]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["measured_bar_mm_day"] = np.where(
                bar_area > 0, bar_volume / bar_area * 1000.0 / days, np.nan)
        out["measured_bar_m3_s"] = bar_volume / res["duration_model_s"]
        out["area_other_m2"] = bar_area
    return out


def transport_by_elevation(res):
    """Net transport averaged per bed-level bin, per along-estuary zone.

    net      : magnitude of the time-mean transport VECTOR [m3 bed/s/m], so ebb
               and flood cancel. Dry faces carry zero transport, so high ground
               that is only flooded at peak flow is included.
    downslope: time-mean net transport component along the local down-slope
               direction (-grad z), [m3 bed/s/m]; positive = net transport
               downhill (e.g. from a bar or bank towards the channel).
    Both are (n_zones, n_bins) arrays for the zones of ZONE_EDGES_KM; bins use
    the bed level at BED_LEVEL_AT. Returns None without bed level.
    """
    if res["bed_level"] is None:
        return None
    fx, fy, z = res["face_x"], res["face_y"], res["bed_level"]
    window = ((fx >= X_LIMITS_M[0]) & (fx <= X_LIMITS_M[1])
              & (fy >= Y_LIMITS_M[0]) & (fy <= Y_LIMITS_M[1]) & np.isfinite(z))
    if not window.any():
        return None

    lo = np.floor(np.percentile(z[window], 1) / BED_BIN_M) * BED_BIN_M
    hi = np.ceil(np.percentile(z[window], 99) / BED_BIN_M) * BED_BIN_M
    edges = np.arange(lo, hi + 0.5 * BED_BIN_M, BED_BIN_M)
    nb = edges.size - 1
    idx = np.digitize(z, edges) - 1
    ok = window & (idx >= 0) & (idx < nb)

    def bin_mean(values, mask):
        count = np.bincount(idx[mask], minlength=nb)
        total = np.bincount(idx[mask], weights=values[mask], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(count >= MIN_FACES_PER_BIN, total / count, np.nan)
        return out

    net = np.hypot(res["mean_x"], res["mean_y"])
    comp = np.full(fx.size, np.nan)
    grid = native_grid(fx, fy)
    if grid is not None:
        ix, iy, ux, uy, _, _ = grid
        z2d = np.full((uy.size, ux.size), np.nan)
        z2d[iy, ix] = z
        gy, gx = np.gradient(z2d, uy, ux)
        gxf, gyf = gx[iy, ix], gy[iy, ix]
        slope = np.hypot(gxf, gyf)
        with np.errstate(invalid="ignore", divide="ignore"):
            comp = -(res["mean_x"] * gxf + res["mean_y"] * gyf) / slope
        comp[~(slope >= SLOPE_MIN)] = np.nan

    zones = list(zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:]))
    net_zone = np.full((len(zones), nb), np.nan)
    down_zone = np.full((len(zones), nb), np.nan)
    for k, (z_lo, z_hi) in enumerate(zones):
        in_zone = ok & (fx >= z_lo * 1000.0) & (fx < z_hi * 1000.0)
        net_zone[k] = bin_mean(net, in_zone)
        down_zone[k] = bin_mean(np.nan_to_num(comp), in_zone & np.isfinite(comp))
    net_zone[net_zone <= 0] = np.nan                   # for the log axis

    return {"z": 0.5 * (edges[:-1] + edges[1:]), "zones": zones,
            "net": net_zone, "downslope": down_zone}


# %% Helper functions: plotting and output
def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {path.name}.png / .pdf")


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Saved: {path.name}")


def nice_reference(values, percentile=90):
    """A round reference magnitude (1 significant digit) for the arrow key."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    ref = np.percentile(values, percentile)
    return float(f"{ref:.1g}") if ref > 0 else 1.0


def draw_vectors(ax, xc, yc, u, v, key_unit, color="white", cmap=None):
    """Quiver with a physically meaningful, labelled arrow scale.

    Arrow length is proportional to magnitude (not autoscaled); the key shows
    the reference value. With cmap, arrows are coloured by magnitude.
    """
    magnitude = np.hypot(u, v)
    ref = nice_reference(magnitude)
    kwargs = dict(angles="xy", scale_units="xy",
                  scale=ref / ARROW_LENGTH_M, width=0.002)
    if cmap is not None:
        q = ax.quiver(xc, yc, u, v, magnitude, cmap=cmap, **kwargs)
    else:
        q = ax.quiver(xc, yc, u, v, color=color, **kwargs)
    ax.quiverkey(q, 0.88, 1.06, ref, f"{ref:g} {key_unit}", labelpos="E",
                 coordinates="axes", labelcolor="black")
    return q


def draw_channel_outline(ax, res):
    """Outline the channel footprint: contour of bed level at the threshold.

    Returns True if drawn (False if the bed level is not available).
    """
    if res["bed_level"] is None:
        return False

    use_mask = res.get("channel_raw") is not None
    field = res["channel_raw"].astype(float) if use_mask else res["bed_level"]
    kwargs = dict(levels=[0.5 if use_mask else CHANNEL_BED_LEVEL_MAX_M],
                  colors=CHANNEL_OUTLINE_COLOR, linewidths=0.8)
    grid = native_grid(res["face_x"], res["face_y"])
    if grid is not None:
        ix, iy, ux, uy, _, _ = grid
        arr = np.full((uy.size, ux.size), np.nan)
        arr[iy, ix] = field
        ax.contour(ux, uy, np.ma.masked_invalid(arr), **kwargs)
    else:
        ok = np.isfinite(field)
        ax.tricontour(res["face_x"][ok], res["face_y"][ok], field[ok], **kwargs)
    return True


def style_map_axes(ax, title):
    ax.set_xlim(X_LIMITS_M)
    ax.set_ylim(Y_LIMITS_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)


# %% Per-scenario plots
CUMULATIVE_FIELDS = {
    "net_mag": (r"Net cumulative transport $|\int \mathbf{q}\,dt|$",
                r" [m$^3$ bed m$^{-1}$]", "magma"),
    "gross": (r"Gross cumulative transport $\int |\mathbf{q}|\,dt$",
              r" [m$^3$ bed m$^{-1}$]", "magma"),
    "directionality": ("Directionality (net / gross)", "", "viridis"),
}


def plot_mean_transport_vectors(res):
    fig, ax = plt.subplots(figsize=(10, 4))
    xc, yc, u, v = coarse_vectors(res, res["mean_x"], res["mean_y"])
    q = draw_vectors(ax, xc, yc, u, v, r"m$^3$ s$^{-1}$ m$^{-1}$",
                     cmap="viridis")
    style_map_axes(ax, f"Time-mean sediment transport\n{res['name']}")
    fig.colorbar(q, ax=ax, pad=0.02).set_label(
        r"Mean transport magnitude [m$^3$ bed s$^{-1}$ m$^{-1}$]"
    )
    fig.tight_layout()
    save_figure(fig, res["output_dir"] / "mean_transport_vectors")


def plot_cumulative_transport(res, field):
    title, unit, cmap = CUMULATIVE_FIELDS[field]
    wet = res["wet"]
    values = res[field].copy()

    if field == "directionality":
        gross_ref = np.percentile(res["gross"][wet], 99)
        values[res["gross"] < DIRECTIONALITY_MIN_GROSS_FRACTION * gross_ref] = np.nan
        vmax, extend = 1.0, "neither"
    else:
        vmax, extend = np.percentile(values[wet], 99), "max"
        if not vmax > 0:
            vmax = None

    fig, ax = plt.subplots(figsize=(10, 4))
    points = ax.scatter(
        res["face_x"][wet], res["face_y"][wet], c=values[wet],
        s=2, cmap=cmap, vmin=0, vmax=vmax, rasterized=True,
    )

    xc, yc, u, v = coarse_vectors(res, res["net_x"], res["net_y"])
    draw_vectors(ax, xc, yc, u, v, r"m$^3$ m$^{-1}$", color="white")

    style_map_axes(ax, f"{title}\n{res['name']}")
    fig.colorbar(points, ax=ax, pad=0.02, extend=extend).set_label(title + unit)
    fig.tight_layout()
    save_figure(fig, res["output_dir"] / f"cumulative_transport_{field}")


def plot_potential_change(res):
    xe, ye, change = potential_bed_change(res)

    vmax = np.nanpercentile(np.abs(change), 98) if np.isfinite(change).any() else np.nan
    if not vmax > 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(xe, ye, change, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         shading="flat", rasterized=True)

    xc, yc, u, v = coarse_vectors(res, res["net_x"], res["net_y"])
    draw_vectors(ax, xc, yc, u, v, r"m$^3$ m$^{-1}$", color="black")

    if draw_channel_outline(ax, res):
        ax.plot([], [], color=CHANNEL_OUTLINE_COLOR, linewidth=0.8,
                label=f"Channel ({res.get('channel_text') or 'footprint'})")
        ax.legend(loc="lower left", fontsize=7, frameon=True)

    style_map_axes(
        ax,
        "Potential bed-level change from transport convergence\n"
        f"{res['name']}",
    )
    fig.colorbar(mesh, ax=ax, pad=0.02, extend="both").set_label(
        "Potential bed-level change [m] (red = deposition)"
    )
    fig.tight_layout()
    save_figure(fig, res["output_dir"] / "potential_erosion_deposition")


def plot_measured_bed_change(res):
    """Measured bed-level change (mesh2d_mor_bl last minus first output), map.

    The potential-change map above is a divergence of reconstructed cell-centre
    transports and shows cell-scale noise; this one is the model's own bed.
    """
    if res.get("bed_change") is None:
        return
    xe = regular_edges(X_LIMITS_M, DIAGNOSTIC_DX_M)
    ye = regular_edges(Y_LIMITS_M, DIAGNOSTIC_DY_M)
    nx, ny = xe.size - 1, ye.size - 1
    everywhere = np.ones(res["face_x"].size, dtype=bool)
    ix, iy, valid = bin_faces(res["face_x"], res["face_y"], xe, ye, everywhere)
    dz = grid_mean(res["bed_change"], ix, iy, nx, ny, valid)
    wet_fraction = grid_mean(res["wet"].astype(float), ix, iy, nx, ny, valid)
    dz[~(wet_fraction >= MIN_WET_FRACTION)] = np.nan
    vmax = np.nanpercentile(np.abs(dz), 98) if np.isfinite(dz).any() else np.nan
    if not vmax > 0:
        vmax = 1e-3

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(xe, ye, dz, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         shading="flat", rasterized=True)
    if draw_channel_outline(ax, res):
        ax.plot([], [], color=CHANNEL_OUTLINE_COLOR, linewidth=0.8,
                label=f"Channel ({res.get('channel_text') or 'footprint'})")
        ax.legend(loc="lower left", fontsize=7, frameon=True)
    style_map_axes(ax, "Measured bed-level change (last - first output)\n"
                       f"{res['name']}")
    fig.colorbar(mesh, ax=ax, pad=0.02, extend="both").set_label(
        "Measured bed-level change [m] (red = deposition)")
    fig.tight_layout()
    save_figure(fig, res["output_dir"] / "measured_bed_change")


def plot_flux_timeseries(res):
    fig, ax = plt.subplots(figsize=(10, 4))
    for b, x_km in enumerate(BOUNDARY_X_KM):
        ax.plot(res["time"], res["boundary_flux_seaward"][:, b],
                linewidth=0.9, label=f"{x_km:.0f} km")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Cross-section flux, seaward positive "
                  r"[m$^3$ bed material s$^{-1}$]")
    ax.set_title(f"Instantaneous sediment flux through x boundaries\n{res['name']}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, res["output_dir"] / "cross_section_flux_timeseries")


def plot_zone_volumes(res, zone_labels):
    """Cumulative zone volume from ZONE_BUDGET_SOURCE (solid), MAP dotted."""
    fig, ax = plt.subplots(figsize=(10, 4))
    main, source = zone_volume_series(res)
    show_map = main is not res["zone_volume"]
    for z, label in enumerate(zone_labels):
        line, = ax.plot(res["time"], main[:, z], linewidth=1.2, label=label)
        if show_map:
            ax.plot(res["time"], res["zone_volume"][:, z], linestyle=":",
                    linewidth=1.2, color=line.get_color())
    if show_map:
        ax.plot([], [], color="0.3", linestyle=":", linewidth=1.2,
                label="MAP reconstruction (does not close)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Cumulative net sediment volume "
                  r"[m$^3$ bed material] (+ = deposition)")
    ax.set_title(f"Short-period zone sediment budget ({source})\n{res['name']}")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, res["output_dir"] / "zone_cumulative_volume")


def export_csv(res, zone_labels, zone_centres):
    csv_dir = res["output_dir"] / "csv"
    times = res["time"].astype(str)

    write_csv(
        csv_dir / "boundary_flux_seaward_positive.csv",
        ["time", *[f"flux_x_{x:.0f}km_m3_bed_per_s" for x in BOUNDARY_X_KM]],
        ([t, *row] for t, row in zip(times, res["boundary_flux_seaward"])),
    )
    # main columns = ZONE_BUDGET_SOURCE (what the figures show); the other
    # sources follow for reference (NaN if unavailable; HIS on shifted edges)
    series, source = zone_volume_series(res)
    write_csv(
        csv_dir / "zone_cumulative_volume.csv",
        ["time", *[f"zone_{label}_m3_bed_material" for label in zone_labels],
         *[f"zone_{label}_map_sections_m3_bed" for label in zone_labels]],
        ([t, *row, *row_map]
         for t, row, row_map in zip(times, series, res["zone_volume"])),
    )
    nan = np.full(len(zone_labels), np.nan)
    his = res.get("his_zone_volume")
    his_end = his[-1] if his is not None else nan
    write_csv(
        csv_dir / "final_zone_budget.csv",
        ["zone", "zone_centre_km",
         "final_cumulative_net_volume_m3_bed_material", "source",
         "map_sections_m3_bed", "his_sections_shifted_edges_m3_bed",
         "map_final_instantaneous_net_rate_m3_bed_per_s"],
        (
            [label, zone_centres[z], series[-1, z], source,
             res["zone_volume"][-1, z], his_end[z], res["net_zone_rate"][-1, z]]
            for z, label in enumerate(zone_labels)
        ),
    )


def plot_channel_infilling(profiles):
    """Peak vs mean flow: how is the channel footprint refilled? (AGU style)"""
    if not profiles:
        print("  No channel profiles available, figure skipped.")
        return

    rows = [
        ("flux_channel_seaward", "Channel flux\n[m$^3$ s$^{-1}$]"),
        ("rate_total_mm_day", "Infilling (+) / scour (-)\n[mm day$^{-1}$]"),
        ("rate_lateral_mm_day", "Lateral supply\n[mm day$^{-1}$]"),
    ]
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}

    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(170 / 25.4, 125 / 25.4),
                                 sharex=True, sharey="row")
        letters = iter("abcdef")
        for c, (flow, scenarios) in enumerate(COMPARISON_PAIRS.items()):
            for r, (key, label) in enumerate(rows):
                ax = axes[r, c]
                for name in scenarios:
                    p = profiles.get(name)
                    if p is not None:
                        ax.plot(p["x_km"], p[key], color=SCENARIO_COLORS[name],
                                linewidth=1.3,
                                label=name.replace("_mean_flow", "")
                                          .replace("_peak_flow", ""))
                ax.axhline(0, color="black", linewidth=0.6)
                ax.grid(alpha=0.3)
                ax.text(0.02, 0.96, f"({next(letters)})", transform=ax.transAxes,
                        va="top", fontweight="bold")
                if c == 0:
                    ax.set_ylabel(label)
            axes[0, c].set_title(flow.replace("_", " ").title())
            axes[0, c].legend(frameon=False, fontsize=7, loc="upper right")
            axes[-1, c].set_xlabel("Distance x [km]")

        sea_left = not POSITIVE_X_IS_SEAWARD
        for c in range(2):
            axes[-1, c].text(0.01, 0.04, "sea" if sea_left else "river",
                             transform=axes[-1, c].transAxes, fontsize=7)
            axes[-1, c].text(0.99, 0.04, "river" if sea_left else "sea",
                             transform=axes[-1, c].transAxes, fontsize=7, ha="right")

        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        save_figure(fig, OUTPUT_BASE / "channel_infilling_peak_vs_mean")


def plot_transport_by_elevation(profiles):
    """Where is sediment moving? Net transport per bed-level bin and zone.

    Columns: zones from sea (left) to river (right). Row 1: net transport
    magnitude, all runs (log). Rows 2-3: net down-slope transport for peak and
    mean flow (constant run as reference in both).
    """
    if not profiles:
        print("  No elevation profiles available, figure skipped.")
        return

    zones = next(iter(profiles.values()))["zones"]
    rows = [
        ("net", "all", "Net transport\n[m$^3$ s$^{-1}$ m$^{-1}$]"),
        ("downslope", "peak_flow", "Net down-slope, PEAK\n[m$^3$ s$^{-1}$ m$^{-1}$]"),
        ("downslope", "mean_flow", "Net down-slope, MEAN\n[m$^3$ s$^{-1}$ m$^{-1}$]"),
    ]
    all_runs = list(dict.fromkeys(n for runs in COMPARISON_PAIRS.values() for n in runs))
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}

    with plt.rc_context(style):
        fig, axes = plt.subplots(len(rows), len(zones),
                                 figsize=(250 / 25.4, 150 / 25.4),
                                 sharex=True, sharey="row", squeeze=False)
        for r, (key, flow, label) in enumerate(rows):
            names = all_runs if flow == "all" else COMPARISON_PAIRS[flow]
            for c, (z_lo, z_hi) in enumerate(zones):
                ax = axes[r, c]
                for name in names:
                    p = profiles.get(name)
                    if p is not None:
                        ax.plot(p["z"], p[key][c], color=SCENARIO_COLORS[name],
                                linewidth=1.2, label=name)
                ax.axvline(CHANNEL_BED_LEVEL_MAX_M, color="black",
                           linestyle=":", linewidth=0.8)
                if key == "net":
                    ax.set_yscale("log")
                else:
                    ax.axhline(0, color="black", linewidth=0.6)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_title(f"{z_lo:g}-{z_hi:g} km")
                if c == 0:
                    ax.set_ylabel(label)
                if r == len(rows) - 1:
                    ax.set_xlabel("Bed level [m]")
            axes[r, -1].legend(frameon=False, fontsize=6, loc="best")
        fig.suptitle("Net transport per bed-level bin, by zone (sea left, river "
                     "right); + down-slope = towards the channel; dotted = "
                     f"{CHANNEL_BED_LEVEL_MAX_M:g} m", fontsize=8)
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        save_figure(fig, OUTPUT_BASE / "transport_by_bed_level")


def zone_rate_mm_day(p):
    """Channel-footprint infilling rate per 5-km zone [mm/day], area-weighted.

    ZONE_RATE_SOURCE = "measured": bed-level change of the channel cells;
    "predicted" (or no measured bed change): flux divergence.
    """
    measured = ZONE_RATE_SOURCE == "measured" and "measured_m3_s" in p
    volume = p["measured_m3_s"] if measured else p["conv_total_m3_s"] * MORFAC
    out = np.full(ZONE_EDGES_KM.size - 1, np.nan)
    for z, (lo, hi) in enumerate(zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:])):
        sel = (p["x_km"] >= lo) & (p["x_km"] < hi)
        area = p["area_m2"][sel].sum()
        if area > 0:
            out[z] = volume[sel].sum() / area * 86400.0 * 1000.0
    return out


def scenario_pairs(profiles):
    """Scenario keys (e.g. 'pm5_n1') that have both a mean- and a peak-flow run."""
    suffix = "_mean_flow"
    return [name[:-len(suffix)] for name in profiles
            if name.endswith(suffix)
            and name[:-len(suffix)] + "_peak_flow" in profiles]


def plot_zone_rates(profiles):
    """Raw channel infilling rate per 5-km zone for every simulation (no weights).

    One panel per scenario: its mean-flow run, its peak-flow run and, as grey
    diamonds, the constant run. Prints the share of time at peak flow at which
    peak and mean flow cancel in each zone (needs opposite signs).
    """
    keys = scenario_pairs(profiles)
    if not keys:
        print("  Mean and peak profiles missing, zone-rate figure skipped.")
        return

    zone_names = [f"{lo:.0f}-{hi:.0f}" for lo, hi
                  in zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:])]
    x = np.arange(len(zone_names))
    width = 0.35
    constant = (zone_rate_mm_day(profiles["constant"])
                if "constant" in profiles else None)
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(keys), figsize=(170 / 25.4, 95 / 25.4),
                                 sharey=True, squeeze=False)
        for ax, key in zip(axes[0], keys):
            mean_rate = zone_rate_mm_day(profiles[f"{key}_mean_flow"])
            peak_rate = zone_rate_mm_day(profiles[f"{key}_peak_flow"])

            ax.bar(x - width / 2, mean_rate, width,
                   color=SCENARIO_COLORS.get(f"{key}_mean_flow", "tab:blue"),
                   label="Mean-flow run")
            ax.bar(x + width / 2, peak_rate, width,
                   color=SCENARIO_COLORS.get(f"{key}_peak_flow", "tab:orange"),
                   label="Peak-flow run")
            if constant is not None:
                ax.plot(x, constant, "D", color=SCENARIO_COLORS["constant"],
                        markerfacecolor="none", markersize=5, label="Constant run")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(zone_names)
            ax.set_xlabel("Zone [km]")
            ax.set_title(key)
            ax.grid(alpha=0.3, axis="y")
            ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
                      bbox_to_anchor=(0.5, -0.25))

            with np.errstate(invalid="ignore", divide="ignore"):
                breakeven = mean_rate / (mean_rate - peak_rate)
            valid = (mean_rate * peak_rate < 0)
            print(f"  {key}: share of time at peak flow for zero net change:")
            for name, b, v in zip(zone_names, breakeven, valid):
                print(f"    {name} km: " + (f"{b:.0%}" if v else
                                            "n/a (same sign in both runs)"))

        axes[0, 0].set_ylabel("Infilling (+) / scour (-)\n[mm day$^{-1}$]")
        fig.tight_layout()
        save_figure(fig, OUTPUT_BASE / "channel_infilling_by_zone")


def stack_step_profiles(res):
    """Stack per-timestep channel profiles into (n_times, n_bins) arrays."""
    steps = res.get("step_profiles") or []
    if not steps or any(p is None for p in steps):
        return None
    out = {key: np.array([p[key] for p in steps])
           for key in ("rate_total_mm_day", "rate_lateral_mm_day",
                       "rate_along_mm_day", "flux_channel_seaward")}
    out["x_km"] = steps[0]["x_km"]
    out["time_h"] = res["time_hours"]
    return out


def plot_infilling_timeseries(res):
    """Channel infilling for every timestep: x-time map + animated GIF."""
    from matplotlib.animation import PillowWriter

    data = stack_step_profiles(res)
    if data is None:
        return
    x, t = data["x_km"], data["time_h"]
    total = data["rate_total_mm_day"]
    ok = np.isfinite(total).any(axis=0)
    if not ok.any():
        return
    lim = np.nanpercentile(np.abs(total), 99)
    lim = lim if lim > 0 else 1.0
    mean_profile = np.nanmean(total, axis=0)
    out_dir = res["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "channel_infilling_per_timestep.npz", **data)

    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        # --- static x-time map ---
        fig, ax = plt.subplots(figsize=(170 / 25.4, 80 / 25.4))
        mesh = ax.pcolormesh(x, t, total, cmap="BrBG", vmin=-lim, vmax=lim,
                             shading="auto")
        ax.set_xlabel("Distance x [km]")
        ax.set_ylabel("Model time [h]")
        fig.colorbar(mesh, ax=ax, pad=0.01).set_label(
            "Infilling (+) / scour (-) [mm day$^{-1}$]")
        ax.set_title(res["name"])
        fig.tight_layout()
        save_figure(fig, out_dir / "channel_infilling_xt")

        # --- GIF: one frame per (n-th) timestep ---
        # Total sediment flux through the full cross-section (whole width) at
        # the sea mouth and the upstream end, positive = seaward.
        i_sea = int(np.argmin(np.abs(BOUNDARY_X_KM - FLUX_X_KM[0])))
        i_riv = int(np.argmin(np.abs(BOUNDARY_X_KM - FLUX_X_KM[1])))
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(170 / 25.4, 95 / 25.4),
            gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(x, mean_profile, color="0.6", linewidth=1.0, label="Run mean")
        line, = ax1.plot(x, total[0], color="black", linewidth=1.4,
                         label="Timestep")
        ax1.axhline(0, color="black", linewidth=0.6)
        ax1.set_ylim(-1.15 * lim, 1.15 * lim)
        ax1.set_xlim(x[0], x[-1])
        ax1.set_xlabel("Distance x [km]")
        ax1.set_ylabel("Infilling (+) / scour (-)\n[mm day$^{-1}$]")
        ax1.grid(alpha=0.3)
        ax1.legend(frameon=False, fontsize=7, loc="upper right")
        title = ax1.set_title(f"{res['name']}   t = {t[0]:.1f} h")
        ax2.plot(t, res["boundary_flux_seaward"][:, i_sea], color="tab:blue",
                 linewidth=1.0, label=f"x = {BOUNDARY_X_KM[i_sea]:.0f} km (sea)")
        ax2.plot(t, res["boundary_flux_seaward"][:, i_riv], color="tab:orange",
                 linewidth=1.0, label=f"x = {BOUNDARY_X_KM[i_riv]:.0f} km (river)")
        ax2.legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
        cursor = ax2.axvline(t[0], color="red", linewidth=1.0)
        ax2.set_xlabel("Model time [h]")
        ax2.set_ylabel("Sediment flux\n[m$^3$ s$^{-1}$]")
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_xlim(t[0], t[-1])
        ax2.grid(alpha=0.3)
        fig.tight_layout()

        gif_path = out_dir / "channel_infilling.gif"
        writer = PillowWriter(fps=GIF_FPS)
        with writer.saving(fig, str(gif_path), GIF_DPI):
            for k in range(0, t.size, GIF_STEP):
                line.set_ydata(total[k])
                cursor.set_xdata([t[k], t[k]])
                title.set_text(f"{res['name']}   t = {t[k]:.1f} h")
                writer.grab_frame()
        plt.close(fig)
    print(f"  Saved {gif_path}")


def plot_time_weighted_budget(profiles):
    """OPTIONAL: net channel infilling per hydrograph = f*peak + (1-f)*mean.

    Only drawn if PEAK_TIME_FRACTION is set. Bars are each run's rate times
    its share of the hydrograph (NOT the raw rates), the dot is their sum.
    """
    if not PEAK_TIME_FRACTION:
        return
    keys = [k for k in PEAK_TIME_FRACTION
            if f"{k}_mean_flow" in profiles and f"{k}_peak_flow" in profiles]
    if not keys:
        print("  Mean and peak profiles missing, time-weighted figure skipped.")
        return

    print(f"\nTime-weighted figure uses PEAK_TIME_FRACTION = {PEAK_TIME_FRACTION}")
    zone_names = [f"{lo:.0f}-{hi:.0f}" for lo, hi
                  in zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:])]
    x = np.arange(len(zone_names))
    width = 0.28
    constant = (zone_rate_mm_day(profiles["constant"])
                if "constant" in profiles else None)
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(keys), figsize=(170 / 25.4, 95 / 25.4),
                                 sharey=True, squeeze=False)
        for ax, key in zip(axes[0], keys):
            f = PEAK_TIME_FRACTION[key]
            mean_rate = zone_rate_mm_day(profiles[f"{key}_mean_flow"])
            peak_rate = zone_rate_mm_day(profiles[f"{key}_peak_flow"])
            net = f * peak_rate + (1 - f) * mean_rate

            ax.bar(x - width / 2, f * peak_rate, width,
                   color=SCENARIO_COLORS.get(f"{key}_peak_flow", "tab:orange"),
                   label=f"Peak flow ({f:.0%} of time)")
            ax.bar(x + width / 2, (1 - f) * mean_rate, width,
                   color=SCENARIO_COLORS.get(f"{key}_mean_flow", "tab:blue"),
                   label=f"Mean flow ({1 - f:.0%} of time)")
            ax.plot(x, net, "o", color="black", markersize=4, label="Net")
            if constant is not None:
                ax.plot(x, constant, "D", color=SCENARIO_COLORS["constant"],
                        markerfacecolor="none", markersize=5, label="Constant run")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(zone_names)
            ax.set_xlabel("Zone [km]")
            ax.set_title(key)
            ax.grid(alpha=0.3, axis="y")
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.25))

        axes[0, 0].set_ylabel("Infilling (+) / scour (-)\n[mm day$^{-1}$]")
        fig.tight_layout()
        save_figure(fig, OUTPUT_BASE / "channel_infilling_time_weighted")


def convergence_mm_day(q_plus_x):
    """Deposition (+) / erosion (-) rate of the strip between two sections.

    q_plus_x : flux through the sections in the +x direction, shape
               (..., n_sections) at CAPACITY_X_KM [m3 bed/s].
    In minus out of each stretch (sections DEPOSITION_INTERVAL_KM apart),
    divided by the stretch area (length x strip width BUDGET_Y_M) -> mean
    bed-level change rate [mm/day]. Returns (x_mid_km, rate).
    """
    spacing = np.median(np.diff(CAPACITY_X_KM))
    step = max(1, int(round(DEPOSITION_INTERVAL_KM / spacing)))
    length_m = (CAPACITY_X_KM[step:] - CAPACITY_X_KM[:-step]) * 1000.0
    width_m = BUDGET_Y_M[1] - BUDGET_Y_M[0]
    inflow_minus_outflow = q_plus_x[..., :-step] - q_plus_x[..., step:]
    rate = inflow_minus_outflow / (length_m * width_m) * MORFAC * 86400.0 * 1000.0
    x_mid = 0.5 * (CAPACITY_X_KM[:-step] + CAPACITY_X_KM[step:])
    return x_mid, rate


def capacity_profile(res):
    """Time-mean transport capacity vs modelled flux per cross-section.

    capacity : NET MPM capacity through the section [m3 bed/s]: the capacity
               vector (pointing the way the model transports) integrated over
               the width, so flood and ebb cancel like they do in the flux
    capacity_gross : time mean of the capacity magnitude (tides do not cancel)
    flux     : modelled net flux through the section, + = seaward
    Also prints how well the recomputed capacity reproduces the model's own
    transport per face (should be ~1 for bed load with instantaneous response;
    deviations point to wrong sediment parameters or bed-slope effects).
    """
    if "cap_section" not in res:
        return None
    w = res["time_weights"]
    x = CAPACITY_X_KM
    capacity_gross = w @ res["cap_section"]
    capacity = SEAWARD_SIGN * (w @ res["cap_net_section"])   # net, + = seaward
    flux = SEAWARD_SIGN * (w @ res["flux_section"])          # net, + = seaward
    i_in = int(np.argmin(np.abs(x - SUPPLY_X_KM)))

    check = res["wet"] & (res["cap_mean"] > 0) & (res["mean_abs"] > 0)
    if check.any():
        ratio = res["mean_abs"][check] / res["cap_mean"][check]
        print(f"  Model transport / recomputed MPM capacity (per wet face): "
              f"median {np.median(ratio):.2f}, 10-90% "
              f"{np.percentile(ratio, 10):.2f}-{np.percentile(ratio, 90):.2f} "
              f"(~1 means the formula and parameters reproduce the model)")
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = flux / np.where(capacity > 0, capacity, np.nan)
    print(f"  Section flux / net capacity: median {np.nanmedian(agree):.2f} "
          f"(~1 = net capacity explains the net flux; the gap to gross "
          f"capacity is tidal back-and-forth transport)")
    x_mid, dep_capacity = convergence_mm_day(w @ res["cap_net_section"])
    _, dep_flux = convergence_mm_day(w @ res["flux_section"])
    period = (f"{str(res['time'][0])[:16]} to {str(res['time'][-1])[:16]} "
              f"({res['time'].size} outputs)")
    print(f"  Capacity/flux profiles are time means over {period}")
    return {
        "period": period,
        "x_km": x,
        "capacity": capacity,
        "capacity_gross": capacity_gross,
        "x_mid_km": x_mid,
        "dep_capacity_mm_day": dep_capacity,
        "dep_flux_mm_day": dep_flux,
        "flux": flux,
        "flux_in": flux[i_in],
        "capacity_in": capacity[i_in],
        "supply_ratio": np.where(
            (capacity > 0) & (flux[i_in] > 0),
            flux[i_in] / np.where(capacity > 0, capacity, np.nan), np.nan),
    }


def plot_capacity_gif(res):
    """GIF of capacity vs flux along x for every output step.

    Top: net capacity C(x) (solid, directed like the transport), modelled flux F(x) (dashed, positive seaward,
    negative = landward) and the inflow F(km 45) (dotted) at this timestep.
    Middle: along-flow change of capacity (solid) and flux (dashed) as a
    deposition (+) / erosion (-) rate of the stretches between sections.
    Bottom: flux at the sea mouth and the river end over time (cursor).
    """
    from matplotlib.animation import PillowWriter

    if "cap_section" not in res:
        return
    x, t = CAPACITY_X_KM, res["time_hours"]
    cap = SEAWARD_SIGN * res["cap_net_section"]      # net (x-directed) capacity
    flux = SEAWARD_SIGN * res["flux_section"]
    i_in = int(np.argmin(np.abs(x - SUPPLY_X_KM)))
    i_sea = int(np.argmin(np.abs(x - FLUX_X_KM[0])))
    i_riv = int(np.argmin(np.abs(x - FLUX_X_KM[1])))
    x_mid, dep_cap = convergence_mm_day(res["cap_net_section"])
    _, dep_flux = convergence_mm_day(res["flux_section"])
    dep_lim = 1.1 * np.percentile(np.abs(np.concatenate([dep_cap, dep_flux])), 99)
    dep_lim = dep_lim if dep_lim > 0 else 1.0
    y_hi = 1.1 * max(cap.max(), flux.max())
    y_lo = min(0.0, 1.1 * min(cap.min(), flux.min()))

    out_dir = res["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(170 / 25.4, 130 / 25.4),
            gridspec_kw={"height_ratios": [3, 3, 1.3]})
        ax2.sharex(ax1)
        line_c, = ax1.plot(x, cap[0], color="black", linewidth=1.4,
                           label="Net capacity C(x)")
        line_f, = ax1.plot(x, flux[0], color="black", linewidth=1.0,
                           linestyle="--", label="Modelled flux F(x)")
        line_in = ax1.axhline(flux[0, i_in], color="tab:red", linewidth=0.8,
                              linestyle=":", label=f"Inflow F(km {SUPPLY_X_KM:.0f})")
        ax1.axhline(0, color="black", linewidth=0.6)
        ax1.set_ylim(y_lo, y_hi)
        ax1.set_xlim(x[0], x[-1])
        ax1.set_ylabel("Transport\n[m$^3$ s$^{-1}$]")
        ax1.grid(alpha=0.3)
        ax1.legend(frameon=False, fontsize=7, loc="upper right", ncol=3)
        plt.setp(ax1.get_xticklabels(), visible=False)
        title = ax1.set_title(f"{res['name']}   t = {t[0]:.1f} h")

        line_dc, = ax2.plot(x_mid, dep_cap[0], color="black", linewidth=1.4,
                            label="from capacity change")
        line_df, = ax2.plot(x_mid, dep_flux[0], color="black", linewidth=1.0,
                            linestyle="--", label="from flux change")
        ax2.axhline(0.0, color="black", linewidth=0.8)
        ax2.set_ylim(-dep_lim, dep_lim)
        ax2.set_xlabel("Distance x [km]")
        ax2.set_ylabel("Deposition (+) / erosion (-)\n[mm day$^{-1}$]")
        ax2.grid(alpha=0.3)
        ax2.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)

        ax3.plot(t, flux[:, i_sea], color="tab:blue", linewidth=1.0,
                 label=f"x = {x[i_sea]:.0f} km")
        ax3.plot(t, flux[:, i_riv], color="tab:orange", linewidth=1.0,
                 label=f"x = {x[i_riv]:.0f} km")
        ax3.axhline(0, color="black", linewidth=0.6)
        cursor = ax3.axvline(t[0], color="red", linewidth=1.0)
        ax3.set_xlim(t[0], t[-1])
        ax3.set_xlabel("Model time [h]")
        ax3.set_ylabel("Flux\n[m$^3$ s$^{-1}$]")
        ax3.legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
        ax3.grid(alpha=0.3)
        fig.tight_layout()

        gif_path = out_dir / "capacity_vs_flux.gif"
        writer = PillowWriter(fps=GIF_FPS)
        with writer.saving(fig, str(gif_path), GIF_DPI):
            for k in range(0, t.size, GIF_STEP):
                line_c.set_ydata(cap[k])
                line_f.set_ydata(flux[k])
                line_in.set_ydata([flux[k, i_in]] * 2)
                line_dc.set_ydata(dep_cap[k])
                line_df.set_ydata(dep_flux[k])
                cursor.set_xdata([t[k], t[k]])
                title.set_text(f"{res['name']}   t = {t[k]:.1f} h")
                writer.grab_frame()
        plt.close(fig)
    print(f"  Saved {gif_path}")


def export_capacity_csv(res, p):
    keys = ["x_km", "capacity", "capacity_gross", "flux", "supply_ratio"]
    write_csv(res["output_dir"] / "csv" / "capacity_vs_flux_profile.csv",
              keys, zip(*[p[k] for k in keys]))
    keys = ["x_mid_km", "dep_capacity_mm_day", "dep_flux_mm_day"]
    write_csv(res["output_dir"] / "csv" / "capacity_vs_flux_deposition.csv",
              keys, zip(*[p[k] for k in keys]))


def plot_capacity_vs_supply(profiles):
    """Is the inflowing sediment more than the flow can carry downstream?

    (a) Net transport capacity C(x) of each cross-section (MPM from the
        model's own velocity and Chezy coefficient, directed like the modelled
        transport so tides cancel as in the flux), the modelled net flux F(x),
        and the inflow F(45 km).
    (b) Along-flow change of both, as a mean bed-level change rate of the
        stretches between sections: deposition (+) where the flux (dashed) or
        the capacity (solid) decreases in the flow direction, erosion (-)
        where it increases.
    """
    if not profiles:
        print("  No capacity profiles available, figure skipped.")
        return

    from matplotlib.lines import Line2D
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(170 / 25.4, 120 / 25.4),
                                 sharex=True, sharey="row")
        letters = iter("abcd")
        for c, (flow, scenarios) in enumerate(COMPARISON_PAIRS.items()):
            for name in scenarios:
                p = profiles.get(name)
                if p is None:
                    continue
                col = SCENARIO_COLORS[name]
                short = name.replace("_mean_flow", "").replace("_peak_flow", "")
                axes[0, c].plot(p["x_km"], p["capacity"], color=col,
                                linewidth=1.4, label=short)
                axes[0, c].plot(p["x_km"], p["flux"], color=col,
                                linewidth=1.0, linestyle="--")
                axes[0, c].axhline(p["flux_in"], color=col, linewidth=0.8,
                                   linestyle=":")
                axes[1, c].plot(p["x_mid_km"], p["dep_capacity_mm_day"],
                                color=col, linewidth=1.4)
                axes[1, c].plot(p["x_mid_km"], p["dep_flux_mm_day"],
                                color=col, linewidth=1.0, linestyle="--")
            axes[1, c].axhline(0.0, color="black", linewidth=0.8)
            axes[0, c].set_title(flow.replace("_", " ").title())
            axes[0, c].legend(frameon=False, fontsize=7, loc="best")
            axes[1, c].set_xlabel("Distance x [km]")
            for r in range(2):
                axes[r, c].grid(alpha=0.3)
                axes[r, c].text(0.0, 1.02, f"({next(letters)})",
                                transform=axes[r, c].transAxes, va="bottom",
                                ha="left", fontweight="bold")

        axes[0, 0].set_ylabel("Net transport\n[m$^3$ s$^{-1}$]")
        axes[1, 0].set_ylabel("Deposition (+) / erosion (-)\n[mm day$^{-1}$]")
        handles = [Line2D([], [], color="0.3", linewidth=1.4,
                          label="Net capacity (solid)"),
                   Line2D([], [], color="0.3", linewidth=1.0, linestyle="--",
                          label="Modelled flux (dashed)"),
                   Line2D([], [], color="0.3", linewidth=0.8, linestyle=":",
                          label=f"Inflow at km {SUPPLY_X_KM:.0f} (dotted)")]
        fig.legend(handles=handles, frameon=False, fontsize=7, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, 0.035))
        sea_left = not POSITIVE_X_IS_SEAWARD
        for c in range(2):
            axes[-1, c].text(0.01, 0.04, "sea" if sea_left else "river",
                             transform=axes[-1, c].transAxes, fontsize=7)
            axes[-1, c].text(0.99, 0.04, "river" if sea_left else "sea",
                             transform=axes[-1, c].transAxes, fontsize=7,
                             ha="right")
        periods = sorted({p["period"] for p in profiles.values()})
        fig.text(0.5, 0.008, "Time mean over " + "; ".join(periods),
                 ha="center", va="bottom", fontsize=6, color="0.4")
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        save_figure(fig, OUTPUT_BASE / "capacity_vs_supply")

    print("\nShare of the inflow at the river end that stays in the estuary "
          "(1 - flux at km 20 / inflow; negative = net erosion):")
    for name, p in profiles.items():
        out = p["flux"][int(np.argmin(np.abs(p["x_km"] - FLUX_X_KM[0])))]
        trapped = (1.0 - out / p["flux_in"]) if p["flux_in"] > 0 else np.nan
        print(f"  {name:20s} inflow {p['flux_in']:.3g} m3/s, outflow at "
              f"km {FLUX_X_KM[0]:.0f}: {out:.3g} m3/s, trapped {trapped:.0%}")


def plot_channel_local_budget(profiles):
    """Why does the channel fill (or scour)? Local supply versus local capacity,
    channels AND bars.

    Time mean over the run, per 1-km bin. "Channel" = footprint with bed level
    below CHANNEL_BED_LEVEL_MAX_M, "bars/banks" = all other cells of the strip.
    Rows (the same in the mean-flow and peak-flow column):
    1. Net transport through the channel part of the section: modelled flux
       (dashed) and MPM capacity (solid).
    2. The same for the bars/banks part. The flux of the channel and of the
       bars adds up to the section flux.
    3. Along-channel deposition (+) / erosion (-) implied by the decrease of
       the channel flux (dashed) or capacity (solid) in the flow direction.
    4. Channel bed change (+ = infilling): MEASURED from mesh2d_mor_bl (solid,
       last minus first output), PREDICTED from the flux divergence of all
       four edges of the channel cells (dashed), and the part of the
       prediction that is LATERAL supply from bars/banks across the channel
       edges (dotted). predicted - lateral = along-channel part.
    """
    if not profiles:
        print("  No channel profiles available, figure skipped.")
        return

    from matplotlib.lines import Line2D
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(4, 2, figsize=(170 / 25.4, 205 / 25.4),
                                 sharex=True, sharey="row")
        letters = iter("abcdefgh")
        for c, (flow, scenarios) in enumerate(COMPARISON_PAIRS.items()):
            handles = []
            for name in scenarios:
                p = profiles.get(name)
                if p is None:
                    continue
                col = SCENARIO_COLORS[name]
                short = name.replace("_mean_flow", "").replace("_peak_flow", "")
                handles.append(Line2D([], [], color=col, linewidth=1.4, label=short))
                if "cap_channel_seaward" in p:
                    axes[0, c].plot(p["x_km"], p["cap_channel_seaward"],
                                    color=col, linewidth=1.4)
                    axes[1, c].plot(p["x_km"], p["cap_other_seaward"],
                                    color=col, linewidth=1.4)
                    axes[2, c].plot(p["x_km"], p["dep_cap_channel_mm_day"],
                                    color=col, linewidth=1.4)
                axes[0, c].plot(p["x_km"], p["flux_channel_seaward"], color=col,
                                linewidth=1.0, linestyle="--")
                axes[1, c].plot(p["x_km"], p["flux_other_seaward"], color=col,
                                linewidth=1.0, linestyle="--")
                axes[2, c].plot(p["x_km"], p["dep_flux_channel_mm_day"],
                                color=col, linewidth=1.0, linestyle="--")
                if "measured_mm_day" in p:
                    axes[3, c].plot(p["x_km"], p["measured_mm_day"], color=col,
                                    linewidth=1.4)
                axes[3, c].plot(p["x_km"], p["rate_total_mm_day"], color=col,
                                linewidth=1.0, linestyle="--")
                axes[3, c].plot(p["x_km"], p["rate_lateral_mm_day"], color=col,
                                linewidth=1.0, linestyle=":")
            axes[0, c].set_title(flow.replace("_", " ").title())
            axes[0, c].legend(handles=handles, frameon=False, fontsize=7,
                              loc="best")
            axes[-1, c].set_xlabel("Distance x [km]")
            for r in range(4):
                axes[r, c].axhline(0.0, color="black", linewidth=0.6)
                axes[r, c].grid(alpha=0.3)
                axes[r, c].text(0.0, 1.02, f"({next(letters)})",
                                transform=axes[r, c].transAxes, va="bottom",
                                ha="left", fontweight="bold")

        axes[0, 0].set_ylabel("Channel net transport\n[m$^3$ s$^{-1}$]")
        axes[1, 0].set_ylabel("Bar/bank net transport\n[m$^3$ s$^{-1}$]")
        axes[2, 0].set_ylabel("Along-channel\ndeposition (+) [mm day$^{-1}$]")
        axes[3, 0].set_ylabel("Channel infilling (+)\n[mm day$^{-1}$]")
        legend = [Line2D([], [], color="0.3", linewidth=1.4,
                         label="solid: capacity (rows 1-3); measured bed change (row 4)"),
                  Line2D([], [], color="0.3", linewidth=1.0, linestyle="--",
                         label="dashed: modelled flux (rows 1-3); predicted from flux divergence (row 4)"),
                  Line2D([], [], color="0.3", linewidth=1.0, linestyle=":",
                         label="dotted: lateral supply from bars/banks into the channel (row 4)")]
        fig.legend(handles=legend, frameon=False, fontsize=7, ncol=1,
                   loc="lower center", bbox_to_anchor=(0.5, 0.0))
        sea_left = not POSITIVE_X_IS_SEAWARD
        for c in range(2):
            axes[-1, c].text(0.01, 0.04, "sea" if sea_left else "river",
                             transform=axes[-1, c].transAxes, fontsize=7)
            axes[-1, c].text(0.99, 0.04, "river" if sea_left else "sea",
                             transform=axes[-1, c].transAxes, fontsize=7,
                             ha="right")
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save_figure(fig, OUTPUT_BASE / "channel_local_supply_vs_capacity")

    print("\nChannel footprint: modelled flux / net capacity (median over the "
          "bins with channel and capacity):")
    for name, p in profiles.items():
        if "flux_over_cap_channel" in p and np.isfinite(p["flux_over_cap_channel"]).any():
            print(f"  {name:20s} {np.nanmedian(p['flux_over_cap_channel']):.2f}")
    print("  ~1 = the channel flux is at capacity everywhere (the bed load "
          "responds instantly), so the deposition\n  is set by how the capacity "
          "changes along the channel, not by a shortage or surplus of supply.")


def plot_measured_channel_vs_bars(profiles):
    """Where did the bed gain and lose sediment? Measured, no fluxes involved.

    Per 1-km bin, measured bed VOLUME change (mesh2d_mor_bl last minus first
    output, times cell area, per day of model time):
      row 1: channel cells (solid) and bars/banks (dashed)
      row 2: channels + bars (net change of the strip)
    Channel gain with bar loss in the same bins, and a net change close to
    zero there, means sediment moved from the bars into the channel.
    """
    if not any("measured_bar_m3_s" in p for p in profiles.values()):
        return
    from matplotlib.lines import Line2D
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(170 / 25.4, 110 / 25.4),
                                 sharex=True, sharey="row")
        letters = iter("abcd")
        for c, (flow, scenarios) in enumerate(COMPARISON_PAIRS.items()):
            handles = []
            for name in scenarios:
                p = profiles.get(name)
                if p is None or "measured_bar_m3_s" not in p:
                    continue
                col = SCENARIO_COLORS[name]
                short = name.replace("_mean_flow", "").replace("_peak_flow", "")
                handles.append(Line2D([], [], color=col, linewidth=1.4, label=short))
                ch = p["measured_m3_s"] * 86400.0
                ba = p["measured_bar_m3_s"] * 86400.0
                axes[0, c].plot(p["x_km"], ch, color=col, linewidth=1.4)
                axes[0, c].plot(p["x_km"], ba, color=col, linewidth=1.0,
                                linestyle="--")
                axes[1, c].plot(p["x_km"], ch + ba, color=col, linewidth=1.4)
            axes[0, c].set_title(flow.replace("_", " ").title())
            axes[0, c].legend(handles=handles, frameon=False, fontsize=7,
                              loc="best")
            axes[-1, c].set_xlabel("Distance x [km]")
            for r in range(2):
                axes[r, c].axhline(0.0, color="black", linewidth=0.6)
                axes[r, c].grid(alpha=0.3)
                axes[r, c].text(0.0, 1.02, f"({next(letters)})",
                                transform=axes[r, c].transAxes, va="bottom",
                                ha="left", fontweight="bold")
        axes[0, 0].set_ylabel("Measured bed volume change\n[m$^3$ day$^{-1}$ per km]")
        axes[1, 0].set_ylabel("Channels + bars\n[m$^3$ day$^{-1}$ per km]")
        fig.legend(handles=[
            Line2D([], [], color="0.3", linewidth=1.4, label="solid: channel cells"),
            Line2D([], [], color="0.3", linewidth=1.0, linestyle="--",
                   label="dashed: bars/banks")],
            frameon=False, fontsize=7, ncol=2, loc="lower center",
            bbox_to_anchor=(0.5, 0.0))
        sea_left = not POSITIVE_X_IS_SEAWARD
        for c in range(2):
            axes[-1, c].text(0.01, 0.04, "sea" if sea_left else "river",
                             transform=axes[-1, c].transAxes, fontsize=7)
            axes[-1, c].text(0.99, 0.04, "river" if sea_left else "sea",
                             transform=axes[-1, c].transAxes, fontsize=7,
                             ha="right")
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save_figure(fig, OUTPUT_BASE / "measured_bed_change_channels_vs_bars")


def print_budget_closure(res, p):
    """Do the channel (finite-volume) and cross-section budgets agree?

    Both are the time-mean in-minus-out sediment volume rate [m3 bed/s] of the
    same 5-km zones (+ = deposition). They use different discretisations, so
    they should agree up to the sediment that crosses the sides of the strip
    (y = BUDGET_Y_M edges), which only the finite-volume budget sees; that
    difference is printed as "(sides)".
    Also splits the finite-volume budget into channel and other cells.
    """
    zone = res["time_weights"] @ res["net_zone_rate"]
    print("  Budget closure per zone [m3/s, + = deposition]:")
    print("    zone         sections   all cells  (sides)    channel     other"
          "   channel area share")
    for z, (lo, hi) in enumerate(zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:])):
        sel = (p["x_km"] > lo) & (p["x_km"] < hi)
        all_cells = p["conv_window_m3_s"][sel].sum()
        channel = p["conv_total_m3_s"][sel].sum()
        share = p["area_m2"][sel].sum() / max(
            (CHANNEL_BIN_M * (Y_LIMITS_M[1] - Y_LIMITS_M[0])) * sel.sum(), 1.0)
        print(f"    {lo:2.0f}-{hi:2.0f} km  {zone[z]:11.3g} {all_cells:11.3g} "
              f"{all_cells - zone[z]:9.3g} {channel:10.3g} "
              f"{all_cells - channel:10.3g}   {share:6.1%}")


def his_zone_volume(folder, map_times):
    """Cumulative zone volume [m3 bed, + = deposition] from the HIS sections.

    Uses the model's own cumulative bed-load through the observation
    cross-sections at ZONE_EDGES_KM (loaded with load_and_cache_scenario, one
    cache file per variable). The seaward direction of every section is taken
    from the sign of its time-mean water discharge, so it does not depend on
    how the section was digitised. Mass is converted with CDryB. The cumulative
    curves are interpolated to the MAP output times and set to zero at the
    first MAP time, so the result is directly comparable with
    res["zone_volume"]. Returns None if the HIS data cannot be used.
    """
    folder = Path(folder)
    his_path = folder / "output" / HIS_FILE_NAME
    if not his_path.exists():
        print(f"  [HIS] {his_path} not found; HIS check skipped.")
        return None

    def load(var):
        cache_file = (ASSESSMENT_DIR
                      / f"hisoutput_{folder.name.replace('.', '_')}_{var}.nc")
        _, data = load_and_cache_scenario(
            scenario_dir=folder, his_file_paths=[his_path],
            cache_file=cache_file, boxes=[], var_name=var)
        return data

    try:
        d_bed = load(HIS_BEDLOAD_VAR)
        d_q = load(HIS_DISCHARGE_VAR)
    except Exception as exc:  # noqa: BLE001
        print(f"  [HIS] could not load the cross-section data ({exc}); "
              f"HIS check skipped.")
        return None
    d_susp = None
    if HIS_INCLUDE_SUSPENDED:
        try:
            d_susp = load(HIS_SUSPENDED_VAR)
        except Exception as exc:  # noqa: BLE001
            print(f"  [HIS] suspended load not available ({exc}); bed load only.")

    km = np.asarray(d_bed["km_positions"], dtype=float)
    idx = []
    for edge in ZONE_EDGES_KM:
        i = int(np.nanargmin(np.abs(km - edge)))
        if abs(km[i] - edge) > HIS_KM_TOL:
            print(f"  [HIS] no cross-section within {HIS_KM_TOL} km of "
                  f"km {edge:g}; HIS check skipped.")
            return None
        idx.append(i)
    idx = np.array(idx)
    offset = km[idx] - ZONE_EDGES_KM
    if np.abs(offset).max() > HIS_EDGE_WARN_KM:
        print("  [HIS] sections are NOT at the zone edges, so the HIS zones are "
              "shifted (compare with care): "
              + ", ".join(f"km{e:g} at {k:.2f}" for e, k in zip(ZONE_EDGES_KM, km[idx])))

    mass = np.asarray(d_bed[HIS_BEDLOAD_VAR], dtype=float)[:, idx]
    if d_susp is not None:
        mass = mass + np.asarray(d_susp[HIS_SUSPENDED_VAR], dtype=float)[:, idx]
    q_mean = np.nanmean(np.asarray(d_q[HIS_DISCHARGE_VAR], dtype=float)[:, idx],
                        axis=0)
    sign = np.where(q_mean >= 0, 1.0, -1.0)
    print("  [HIS] seaward sign per section (from mean discharge): "
          + ", ".join(f"km{km[i]:g}:{'+' if s > 0 else '-'}"
                      for i, s in zip(idx, sign)))
    if not np.isfinite(mass).all():
        print("  [HIS] cumulative transport contains NaNs; HIS check skipped.")
        return None

    cum = sign * mass / DRY_BED_DENSITY_KG_M3          # seaward, [m3 bed]
    t0 = map_times[0]
    his_s = ((np.asarray(d_bed["t"]).astype("datetime64[ns]") - t0)
             / np.timedelta64(1, "s")).astype(float)
    map_s = ((map_times - t0) / np.timedelta64(1, "s")).astype(float)
    if his_s[0] > map_s[0] + 1.0 or his_s[-1] < map_s[-1] - 1.0:
        print(f"  [HIS] HIS times ({str(d_bed['t'][0])[:16]} to "
              f"{str(d_bed['t'][-1])[:16]}) do not cover the MAP period; "
              f"HIS check skipped.")
        return None
    cum_map = np.column_stack(
        [np.interp(map_s, his_s, cum[:, j]) for j in range(cum.shape[1])])
    cum_map -= cum_map[0]
    # zone between edge z (downstream) and z+1 (upstream): in - out
    return (cum_map[:, 1:] - cum_map[:, :-1]) * MORFAC


def zone_volume_series(res):
    """Cumulative whole-width zone volume [m3 bed, + = deposition] and its source.

    Returns (n_times x n_zones array, label) following ZONE_BUDGET_SOURCE;
    falls back to the MAP reconstruction (which does not close the zone
    budgets, see print_his_validation) only if the bed change is unavailable.
    """
    if ZONE_BUDGET_SOURCE == "bed" and res.get("zone_bed_volume") is not None:
        return res["zone_bed_volume"], "measured bed-level change"
    if ZONE_BUDGET_SOURCE != "map":
        print(f"  [WARNING] '{ZONE_BUDGET_SOURCE}' zone budget not available for "
              f"{res['name']}; using the MAP reconstruction, which does not "
              f"close the zone budgets (signs can be wrong).")
    return res["zone_volume"], "MAP sections"


def print_his_validation(res, zone_labels):
    """Zone budget three ways (whole width, + = deposition, m3 bed material).

    MAP sections : flux through the x boundaries reconstructed from the
                   cell-centre transports, over the strip BUDGET_Y_M.
    HIS sections : the model's own cumulative bed-load through its observation
                   cross-sections (reference for the section flux).
    bed dz       : measured bed-level change times cell area per zone (last
                   minus first output of mesh2d_mor_bl), over the strip and
                   over the full width: the model's own bed result.
    A zone budget is the difference of two large fluxes, so small relative
    errors in a section flux become large errors in the zone budget.
    """
    his = res.get("his_zone_volume")
    strip, full = res.get("zone_bed_strip_m3"), res.get("zone_bed_all_m3")
    if his is None and strip is None:
        return
    mapv = res["zone_volume"]
    nan = np.full(len(zone_labels), np.nan)
    his_end = his[-1] if his is not None else nan
    strip = strip if strip is not None else nan
    full = full if full is not None else nan
    print(f"  Zone budget over the run [m3 bed, + = deposition]; MAP sections "
          f"span y = {BUDGET_Y_M[0] / 1000:g}-{BUDGET_Y_M[1] / 1000:g} km:")
    print("    zone        MAP sect.  HIS sect.   bed dz strip   bed dz full width")
    for z, label in enumerate(zone_labels):
        print(f"    {label:9s} {mapv[-1, z]:10.4g} {his_end[z]:10.4g} "
              f"{strip[z]:13.4g} {full[z]:16.4g}")
    print(f"    {'total':9s} {mapv[-1].sum():10.4g} {np.nansum(his_end):10.4g} "
          f"{np.nansum(strip):13.4g} {np.nansum(full):16.4g}")
    print("    The zone budget is a small difference of two large fluxes; a few "
          "percent error in a section flux\n    already changes (or reverses) a "
          "zone. The totals over 20-45 km are far less sensitive. bed dz is\n"
          "    the reference (exact zone edges); the HIS sections are shifted from "
          "the zone edges. Strip and full\n    width differ when sediment moves "
          "outside the strip.")
    if his is not None:
        scale = max(np.abs(his).max(), 1e-12)
        print(f"    largest MAP-HIS difference over time: "
              f"{np.abs(mapv - his).max() / scale:.1%} of the largest HIS zone volume")
        write_csv(
            res["output_dir"] / "csv" / "zone_volume_map_vs_his.csv",
            ["time", *[f"{lab}_{src}_m3_bed" for lab in zone_labels
                       for src in ("map", "his")]],
            ([str(t), *[v for z in range(len(zone_labels))
                        for v in (mapv[k, z], his[k, z])]]
             for k, t in enumerate(res["time"])),
        )


def export_channel_csv(res, p):
    keys = ["x_km", "flux_channel_seaward", "flux_other_seaward",
            "rate_total_mm_day", "rate_along_mm_day", "rate_lateral_mm_day",
            "channel_width_m", "bed_level_mean_m", "cap_channel_seaward",
            "flux_over_cap_channel", "dep_flux_channel_mm_day",
            "dep_cap_channel_mm_day", "measured_mm_day"]
    keys = [k for k in keys if k in p]
    write_csv(
        res["output_dir"] / "csv" / "channel_infilling_profile.csv",
        keys,
        zip(*[p[k] for k in keys]),
    )


def print_directionality_summary(res):
    """Is net really ~ gross? Median net/gross of wet faces per 5-km zone.

    ~1.00 = transport never reverses (net == gross, the two maps look alike);
    clearly < 1 = ebb/flood or flow reversals cancel part of the transport.
    """
    d, wet, x = res["directionality"], res["wet"], res["face_x"]
    print("  Directionality (net/gross), wet faces with transport:")
    for lo, hi in zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:]):
        sel = wet & (x >= lo * 1000) & (x < hi * 1000) & np.isfinite(d)
        if sel.any():
            print(f"    {lo:.0f}-{hi:.0f} km: median {np.median(d[sel]):.3f}, "
                  f"10th percentile {np.percentile(d[sel], 10):.3f}")


def print_channel_summary(profiles):
    print("\nChannel footprint, area-weighted over the zoom domain "
          "[mm/day of bed level, + = infilling]:")
    print(f"  {'scenario':20s}{'predicted':>10s}{'along':>9s}{'lateral':>9s}"
          f"{'MEASURED':>10s}")
    for name, p in profiles.items():
        factor = MORFAC * 86400.0 * 1000.0 / p["area_m2"].sum()
        total = p["conv_total_m3_s"].sum() * factor
        lateral = p["conv_lateral_m3_s"].sum() * factor
        measured = (p["measured_m3_s"].sum() * 86400.0 * 1000.0
                    / p["area_m2"].sum()) if "measured_m3_s" in p else np.nan
        print(f"  {name:20s}{total:10.2f}{total - lateral:9.2f}{lateral:9.2f}"
              f"{measured:10.2f}")
    print("  predicted = flux divergence of the time-mean transport; MEASURED = "
          "change of mesh2d_mor_bl\n  (last - first output) averaged over the "
          "same channel cells. They should agree if the\n  transport "
          "reconstruction is right and the run morphology is active.")

    if CHANNEL_MAIN_ONLY:
        first = next(iter(profiles.values()))
        print("\nChannel footprint (%s, largest connected component only):"
              % (first.get("definition") or "bed level"))
        for name, p in profiles.items():
            a_ch, a_pit = p["area_m2"].sum(), p["area_pits_m2"].sum()
            print(f"  {name:20s} channel {a_ch / 1e6:6.2f} km2, isolated pits "
                  f"excluded {a_pit / 1e6:5.2f} km2 ({a_pit / max(a_ch + a_pit, 1):.0%})")
    if any("measured_bar_m3_s" in p for p in profiles.values()):
        print("\nMEASURED bed volume change of the zoom domain [m3 per day, "
              "+ = gain]: channels vs bars/banks.")
        print("(no transport reconstruction involved; channel gain with bar "
              "loss = sediment moved from bars to channels)")
        print(f"  {'scenario':20s}{'channels':>11s}{'bars/banks':>12s}{'sum':>10s}"
              f"{'channel mm/d':>14s}{'bars mm/d':>11s}")
        for name, p in profiles.items():
            if "measured_bar_m3_s" not in p:
                continue
            ch = p["measured_m3_s"].sum() * 86400.0
            ba = p["measured_bar_m3_s"].sum() * 86400.0
            ch_mm = ch * 1000.0 / p["area_m2"].sum()
            ba_mm = ba * 1000.0 / max(p["area_other_m2"].sum(), 1.0)
            print(f"  {name:20s}{ch:11.1f}{ba:12.1f}{ch + ba:10.1f}"
                  f"{ch_mm:14.2f}{ba_mm:11.2f}")


# %% Analysis of one scenario (streams through time in blocks)
def analyse_dataset(ds, scenario_name):
    missing = [name for name in (SX_VAR, SY_VAR) if name not in ds]
    if missing:
        raise KeyError(f"Missing variables in dataset: {missing}")

    all_x, all_y = _get_face_coords(ds)
    all_x = np.asarray(all_x, dtype=float)
    all_y = np.asarray(all_y, dtype=float)

    face_dim = get_face_dimension(ds, SX_VAR, all_x.size)
    if ds.sizes[face_dim] != all_x.size:
        raise ValueError(
            f"Face coordinates ({all_x.size}) do not match dimension "
            f"{face_dim} ({ds.sizes[face_dim]})."
        )

    # Only read the faces that are actually used (big saving for the
    # 45 x 15 km domain when the analysis zoom is 26 x 5 km).
    face_idx = domain_face_indices(all_x, all_y)
    if face_idx.size == 0:
        raise ValueError("No faces inside the analysis domain.")
    face_x, face_y = all_x[face_idx], all_y[face_idx]
    n_faces = face_idx.size

    fraction_dims = [d for d in ds[SX_VAR].dims if d not in ("time", face_dim)]
    print(f"  Face dimension: {face_dim}; summing over: "
          f"{fraction_dims or 'nothing'}; "
          f"using {n_faces} of {all_x.size} faces")

    # --- time axis ---
    time = np.asarray(ds["time"].values).astype("datetime64[ns]")
    start = TIME_START_INDEX
    end = time.size if TIME_END_INDEX is None else TIME_END_INDEX
    if not (0 <= start < end - 1 <= time.size - 1):
        raise ValueError(
            f"Invalid time selection: start={start}, end={end}, "
            f"available={time.size} (need at least 2 timesteps)."
        )
    time = time[start:end]
    n_times = time.size

    time_seconds = ((time - time[0]) / np.timedelta64(1, "s")).astype(float)
    if np.any(np.diff(time_seconds) < 0):
        raise ValueError(
            "Time axis is not monotonic (overlap between stitched runs?). "
            "Integrals would be wrong."
        )
    time_morph = time_seconds * MORFAC
    duration = time_morph[-1] - time_morph[0]
    if duration <= 0:
        raise ValueError("Zero-length time selection.")
    weights = trapezoid_weights(time_morph)
    print(f"  Time span: {duration / MORFAC / 3600:.1f} h of model time "
          f"({n_times} outputs, mean interval "
          f"{duration / MORFAC / 3600 / (n_times - 1):.2f} h)")

    units_mode = resolve_transport_units(ds, SX_VAR)
    factor = bed_volume_factor(units_mode)

    boundary_weights = build_boundary_weights(face_x, face_y)

    da_x = ds[SX_VAR].isel({face_dim: face_idx})
    da_y = ds[SY_VAR].isel({face_dim: face_idx})
    da_d = (ds[WATER_DEPTH_VAR].isel({face_dim: face_idx})
            if WATER_DEPTH_VAR in ds else None)

    # bed level (for the channel footprint), at one time level
    bed_level = None
    bed_change = None    # measured bed-level change, last minus first output [m]
    bed_var = next((v for v in (BED_LEVEL_VAR, BED_LEVEL_FALLBACK)
                    if v in ds), None)
    if bed_var is not None:
        da_bl = ds[bed_var].isel({face_dim: face_idx})
        if "time" in da_bl.dims:
            t_idx = end - 1 if BED_LEVEL_AT == "last" else start
            bed_level = read_block(da_bl, face_dim, t_idx, t_idx + 1,
                                   fill_nan=False)[0]
            bed_change = (
                read_block(da_bl, face_dim, end - 1, end, fill_nan=False)[0]
                - read_block(da_bl, face_dim, start, start + 1,
                             fill_nan=False)[0])
            if np.isfinite(bed_change).any():
                moved = np.nanmax(np.abs(bed_change))
                print(f"  Measured bed-level change ({bed_var}, last - first "
                      f"output): max |dz| = {moved:.3g} m over "
                      f"{duration / MORFAC / 86400:.2f} d")
                if moved < 1e-6:
                    print("  [WARNING] the bed level did not change: the run "
                          "may have a fixed bed, so the infilling below is "
                          "potential, not realised.")
        else:
            bed_level = np.asarray(da_bl.values, dtype=float)
            print(f"  [WARNING] '{bed_var}' has no time axis: the channel "
                  f"footprint uses that static bed level, BED_LEVEL_AT is "
                  f"ignored. Use '{BED_LEVEL_VAR}' for the time-varying bed.")
        if bed_var != BED_LEVEL_VAR:
            print(f"  [WARNING] '{BED_LEVEL_VAR}' not in dataset, using "
                  f"'{bed_var}'.")
    else:
        print(f"  [WARNING] no bed level variable in dataset; "
              f"no channel diagnostic.")

    # --- channel footprint: bed-level cut and/or "wet at low water" ---
    channel_raw, channel_text = None, None
    if bed_level is not None:
        with np.errstate(invalid="ignore"):
            channel_bed = bed_level < CHANNEL_BED_LEVEL_MAX_M
        channel_raw = channel_bed
        channel_text = f"bed level < {CHANNEL_BED_LEVEL_MAX_M:g} m ({BED_LEVEL_AT} output)"
        if CHANNEL_DEFINITION == "wet":
            if WATER_LEVEL_VAR not in ds or "time" not in ds[WATER_LEVEL_VAR].dims:
                print(f"  [WARNING] '{WATER_LEVEL_VAR}' not available: channel "
                      f"footprint falls back to the bed-level cut.")
            else:
                at_last = WET_WINDOW_AT == "last"
                tt = (time - time[-1] if at_last else time - time[0]) / np.timedelta64(1, "h")
                win = np.flatnonzero(tt >= -WET_WINDOW_H) if at_last \
                    else np.flatnonzero(tt <= WET_WINDOW_H)
                if win.size == 0:
                    win = np.array([n_times - 1 if at_last else 0])
                da_s1 = ds[WATER_LEVEL_VAR].isel({face_dim: face_idx})
                s1 = read_block(da_s1, face_dim, start + int(win[0]),
                                start + int(win[-1]) + 1, fill_nan=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    s1_ref = np.nanmin(s1, axis=0) if WET_LEVEL_STAT == "min" \
                        else np.nanmax(s1, axis=0)
                # bed at the same moment as the water level, as in the cross-section script
                t_bed = end - 1 if at_last else start
                bed_ref = (read_block(da_bl, face_dim, t_bed, t_bed + 1,
                                      fill_nan=False)[0]
                           if "time" in da_bl.dims else bed_level)
                with np.errstate(invalid="ignore"):
                    channel_wet = (s1_ref - bed_ref) > WET_DEPTH_THRESHOLD_M
                channel_raw = channel_wet
                channel_text = (f"wet at {WET_LEVEL_STAT} water level "
                                f"(depth > {WET_DEPTH_THRESHOLD_M:g} m, {WET_WINDOW_AT} "
                                f"{WET_WINDOW_H:g} h)")
                g = native_grid(face_x, face_y)
                if g is not None:
                    gx, gy, _, _, gdx, gdy = g
                    a = (gdx[gx] * gdy[gy]) * ((face_x >= X_LIMITS_M[0]) & (face_x <= X_LIMITS_M[1])
                                              & (face_y >= Y_LIMITS_M[0]) & (face_y <= Y_LIMITS_M[1]))
                    both = (channel_wet & channel_bed) * a
                    print(f"  Channel area in the zoom window: wet {np.sum(channel_wet * a) / 1e6:.2f} km2, "
                          f"bed level < {CHANNEL_BED_LEVEL_MAX_M:g} m {np.sum(channel_bed * a) / 1e6:.2f} km2, "
                          f"overlap {both.sum() / 1e6:.2f} km2")

        # shared footprint: reuse the footprint of another run (same grid and
        # same initial bed), so mean and peak runs split channel/bars alike
        source = CHANNEL_FOOTPRINT_FROM.get(scenario_name, scenario_name)
        if source != scenario_name:
            if source in FOOTPRINTS:
                src_x, src_y, src_channel, src_text = FOOTPRINTS[source]
                lookup = dict(zip(zip(np.round(src_x, 1), np.round(src_y, 1)), src_channel))
                channel_raw = np.array([lookup.get(k, False) for k in
                                        zip(np.round(face_x, 1), np.round(face_y, 1))])
                channel_text = f"{src_text}, footprint of {source}"
                print(f"  Channel footprint taken from {source}.")
            else:
                print(f"  [WARNING] footprint of {source} not available (process it "
                      f"first); {scenario_name} uses its own footprint.")
        if channel_raw is not None:
            FOOTPRINTS[scenario_name] = (face_x, face_y, channel_raw, channel_text)

    # Measured bed VOLUME change per zone (bed level minus the first output,
    # times the cell area) at every output: the model's own answer to "where
    # did the sediment go", on the exact zone edges. Over the full width (time
    # series) and over the budget strip BUDGET_Y_M (final value only).
    zone_bed_strip = zone_bed_all = zone_bed_volume = None
    if bed_var is not None and "time" in ds[bed_var].dims:
        grid_all = native_grid(all_x, all_y)
        if grid_all is not None:
            gix, giy, _, _, gdx, gdy = grid_all
            cell_area_all = gdx[gix] * gdy[giy]
            zone_w = np.column_stack([
                np.where((all_x >= lo * 1000.0) & (all_x < hi * 1000.0),
                         cell_area_all, 0.0)
                for lo, hi in zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:])])
            in_strip = (all_y >= BUDGET_Y_M[0]) & (all_y <= BUDGET_Y_M[1])
            da_full = ds[bed_var]
            bl0 = read_block(da_full, face_dim, start, start + 1, fill_nan=False)[0]
            zone_bed_volume = np.empty((n_times, zone_w.shape[1]))
            for s in range(0, n_times, TIME_CHUNK):
                e = min(s + TIME_CHUNK, n_times)
                dz = np.nan_to_num(read_block(da_full, face_dim, start + s,
                                              start + e, fill_nan=False) - bl0)
                zone_bed_volume[s:e] = dz @ zone_w
            zone_bed_all = zone_bed_volume[-1].copy()
            zone_bed_strip = dz[-1] @ (zone_w * in_strip[:, None])

    step_grid = (native_grid(face_x, face_y)
                 if (TIMESTEP_PROFILES and bed_level is not None) else None)
    step_res = {"face_x": face_x, "face_y": face_y, "bed_level": bed_level,
                "channel_raw": channel_raw}
    step_profiles = []

    # --- optional transport capacity (needs tau or velocity, and Chezy) ---
    theta_var = THETA_VARS[THETA_SOURCE]
    layer_on = MPM_FIXED_LAYER_THRESH_M is not None
    needed = [CHEZY_VAR, theta_var]
    use_capacity = CAPACITY_ANALYSIS and all(v in ds for v in needed)
    if use_capacity and layer_on and DPSED_VAR not in ds:
        print(f"  [WARNING] '{DPSED_VAR}' missing: fixed-layer factor ignored.")
        layer_on = False
    if CAPACITY_ANALYSIS and not use_capacity:
        print(f"  [WARNING] capacity analysis needs {needed}; missing "
              f"{[v for v in needed if v not in ds]}. Skipped.")
    if use_capacity:
        da_c = ds[CHEZY_VAR].isel({face_dim: face_idx})
        da_t = ds[theta_var].isel({face_dim: face_idx})
        da_p = ds[DPSED_VAR].isel({face_dim: face_idx}) if layer_on else None
        cap_weights = build_boundary_weights(face_x, face_y,
                                             CAPACITY_X_KM * 1000.0)
        cap_section = np.zeros((n_times, CAPACITY_X_KM.size))      # gross |capacity|
        cap_net_section = np.zeros((n_times, CAPACITY_X_KM.size))  # x-directed
        flux_section = np.zeros((n_times, CAPACITY_X_KM.size))
        cap_int = np.zeros(n_faces)
        capx_int = np.zeros(n_faces)     # x-directed (net) capacity per face

    net_x = np.zeros(n_faces)
    net_y = np.zeros(n_faces)
    gross = np.zeros(n_faces)
    depth_sum = np.zeros(n_faces)
    depth_count = np.zeros(n_faces)
    boundary_flux = np.empty((n_times, BOUNDARY_X_M.size))

    print(f"  Reading {n_times} timesteps in blocks of {TIME_CHUNK} ...")
    for s in range(0, n_times, TIME_CHUNK):
        e = min(s + TIME_CHUNK, n_times)
        sx = read_block(da_x, face_dim, start + s, start + e) * factor
        sy = read_block(da_y, face_dim, start + s, start + e) * factor
        w = weights[s:e]

        net_x += w @ sx
        net_y += w @ sy
        gross += w @ np.hypot(sx, sy)
        boundary_flux[s:e] = sx @ boundary_weights

        if use_capacity:
            depth_c = read_block(da_d, face_dim, start + s, start + e) \
                if da_d is not None else np.full_like(sx, np.inf)
            chezy = read_block(da_c, face_dim, start + s, start + e)
            raw = read_block(da_t, face_dim, start + s, start + e)
            with np.errstate(invalid="ignore", divide="ignore"):
                tau = raw if THETA_SOURCE != "velocity" else np.where(
                    chezy > 0, RHO_WATER * GRAVITY * (raw / chezy) ** 2, 0.0)
            layer = 1.0
            if da_p is not None:
                dpsed = read_block(da_p, face_dim, start + s, start + e)
                layer = np.clip(dpsed / MPM_FIXED_LAYER_THRESH_M, 0.0, 1.0)
            cap = mpm_capacity(tau, chezy, depth_c, layer)
            cap_int += w @ cap
            cap_section[s:e] = cap @ cap_weights
            # capacity as a vector pointing the way the model transports:
            # its x-component integrates to a net (tide-cancelling) capacity
            mag = np.hypot(sx, sy)
            dir_x = np.divide(sx, mag, out=np.zeros_like(sx), where=mag > 0)
            cap_net_section[s:e] = (cap * dir_x) @ cap_weights
            capx_int += w @ (cap * dir_x)
            flux_section[s:e] = sx @ cap_weights

        if step_grid is not None:
            for k in range(e - s):
                step_profiles.append(channel_infilling_profiles(
                    step_res, qx_face=sx[k], qy_face=sy[k], grid=step_grid))

        if da_d is not None:
            depth = read_block(da_d, face_dim, start + s, start + e,
                               fill_nan=False)
            finite = np.isfinite(depth)
            depth_sum += np.where(finite, depth, 0.0).sum(axis=0)
            depth_count += finite.sum(axis=0)

    if da_d is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_depth = depth_sum / depth_count
        wet = np.isfinite(mean_depth) & (mean_depth >= MIN_WATER_DEPTH_M)
    else:
        wet = np.ones(n_faces, dtype=bool)

    if not wet.any():
        print("  No wet faces found, skipping.")
        return None

    zone_rate, zone_volume = compute_zone_budgets(boundary_flux, time_morph)

    res = {
        "name": scenario_name,
        "output_dir": OUTPUT_BASE / scenario_name,
        "time": time,
        "face_x": face_x,
        "face_y": face_y,
        "wet": wet,
        "bed_level": bed_level,
        "channel_raw": channel_raw,            # bool per face, see CHANNEL_DEFINITION
        "channel_text": channel_text,
        "bed_change": bed_change,              # measured dz, last - first [m]
        "duration_model_s": time_seconds[-1] - time_seconds[0],
        "zone_bed_strip_m3": zone_bed_strip,   # measured dz x area per zone, strip
        "zone_bed_all_m3": zone_bed_all,       # same, full domain width
        "zone_bed_volume": zone_bed_volume,    # full width, every output (n_times, n_zones)
        "boundary_flux": boundary_flux,                       # +x direction
        "boundary_flux_seaward": SEAWARD_SIGN * boundary_flux,
        "net_zone_rate": zone_rate,
        "zone_volume": zone_volume,
        "time_hours": time_seconds / 3600.0,
        "time_weights": weights / duration,    # time mean = time_weights @ x
        "step_profiles": step_profiles,
    }
    res.update(derive_fields(net_x, net_y, gross, duration))
    if use_capacity:
        res["cap_mean"] = cap_int / duration            # time-mean capacity per face
        res["cap_net_x_mean"] = capx_int / duration     # time-mean x-directed capacity
        res["cap_section"] = cap_section                # (n_times, n_sections)
        res["cap_net_section"] = cap_net_section        # x-directed capacity
        res["flux_section"] = flux_section              # +x direction
    return res


def analyse_scenario(scenario_name, folder_path):
    """Load the cached MAP data of one scenario and analyse it.

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
        run_paths = [folder]  # folder is already an absolute path

    # --- cache load (identical call to the sensitivity script) ---
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=ASSESSMENT_DIR,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=[SX_VAR, SY_VAR, WATER_DEPTH_VAR, BED_LEVEL_VAR]
        + ([WATER_LEVEL_VAR] if CHANNEL_DEFINITION == "wet" else [])
        + ([CHEZY_VAR, THETA_VARS[THETA_SOURCE]]
           + ([DPSED_VAR] if MPM_FIXED_LAYER_THRESH_M is not None else [])
           if CAPACITY_ANALYSIS else []),
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )
    if ds is None:
        print(f"  No cached data for {folder.name}, skipping.")
        return None

    try:
        res = analyse_dataset(ds, scenario_name)
    finally:
        ds.close()  # always release the file handle, even on errors

    if res is not None and HIS_VALIDATION:
        res["his_zone_volume"] = his_zone_volume(folder, res["time"])
    return res


# %% Run all scenarios (plots/CSV per scenario, then free memory)
zone_labels = [
    f"{ZONE_EDGES_KM[i]:.0f}-{ZONE_EDGES_KM[i + 1]:.0f} km"
    for i in range(ZONE_EDGES_KM.size - 1)
]
zone_centres = 0.5 * (ZONE_EDGES_KM[:-1] + ZONE_EDGES_KM[1:])

final_volumes = {}
final_sources = {}
channel_profiles = {}
elevation_profiles = {}
capacity_profiles = {}

for scenario_name, folder_path in SCENARIOS.items():
    print("\n" + "=" * 80)
    print(f"Processing: {scenario_name}")
    print(f"Folder: {folder_path}")
    print("=" * 80)

    res = analyse_scenario(scenario_name, folder_path)
    if res is None:
        continue

    plot_mean_transport_vectors(res)
    for field in CUMULATIVE_FIELDS:
        plot_cumulative_transport(res, field)
    plot_potential_change(res)
    plot_measured_bed_change(res)
    plot_flux_timeseries(res)
    plot_zone_volumes(res, zone_labels)
    export_csv(res, zone_labels, zone_centres)
    print_his_validation(res, zone_labels)
    print_directionality_summary(res)

    plot_infilling_timeseries(res)

    profile = channel_infilling_profiles(res)
    if profile is not None:
        channel_profiles[scenario_name] = profile
        export_channel_csv(res, profile)
        print_budget_closure(res, profile)

    capacity = capacity_profile(res)
    if capacity is not None:
        capacity_profiles[scenario_name] = capacity
        export_capacity_csv(res, capacity)
        plot_capacity_gif(res)

    elevation = transport_by_elevation(res)
    if elevation is not None:
        elevation_profiles[scenario_name] = elevation

    series, final_sources[scenario_name] = zone_volume_series(res)
    final_volumes[scenario_name] = series[-1].copy()
    print(f"  Final zone volumes [m3] ({final_sources[scenario_name]}): "
          f"{np.round(final_volumes[scenario_name]).astype(int)}")
    del res  # release the per-face arrays before the next scenario


# %% Scenario comparison: final zone budget
for flow_condition, scenarios in COMPARISON_PAIRS.items():
    fig, ax = plt.subplots(figsize=(8, 4))

    for scenario_name in scenarios:
        if scenario_name not in final_volumes:
            print(f"   Skipping {scenario_name} in {flow_condition}: "
                  f"not loaded.")
            continue

        ax.plot(zone_centres, final_volumes[scenario_name], marker="o",
                linewidth=1.5, color=SCENARIO_COLORS[scenario_name],
                label=scenario_name)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Zone centre [km]")
    ax.set_ylabel(r"Final cumulative net volume "
                  r"[m$^3$ bed material] (+ = deposition)")
    sources = sorted({final_sources[s] for s in scenarios if s in final_sources})
    ax.set_title(f"{flow_condition.replace('_', ' ').title()} scenarios:\n"
                 f"final zone sediment budget ({', '.join(sources)})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, OUTPUT_BASE / f"comparison_{flow_condition}_final_zone_budget")


# %% Channel infilling figure: peak vs mean flow
plot_channel_infilling(channel_profiles)
plot_transport_by_elevation(elevation_profiles)
plot_zone_rates(channel_profiles)
plot_capacity_vs_supply(capacity_profiles)
plot_channel_local_budget(channel_profiles)
plot_measured_channel_vs_bars(channel_profiles)
plot_time_weighted_budget(channel_profiles)     # only if PEAK_TIME_FRACTION is set
if channel_profiles:
    print_channel_summary(channel_profiles)


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