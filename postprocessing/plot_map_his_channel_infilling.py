# %% Do channels infill at peak flow because bars feed them? (slim version)
"""
Question: why are channels not deeper under high discharge variability? Is it
because, at peak flow, sediment supply to the channel exceeds the local
transport capacity (infilling), and where does that sediment come from?

Everything here is either MEASURED by the model or a direct function of its
output. No zone budget from reconstructed cross-section fluxes is used (for the
exact cross-section budget use his_flux_budget.py, which reads the HIS file).

What it does, per scenario (MAP files, streamed through time)
-------------------------------------------------------------
1. Channel footprint = faces that are wet at LOW water (min water level of the
   last WET_WINDOW_H hours minus bed level > WET_DEPTH_THRESHOLD_M), as in the
   cross-section script. Largest connected component only; isolated pits are
   excluded from both channel and bars. Everything else in the zoom window is
   "bars/banks". (CHANNEL_DEFINITION = "bed_level" gives a fixed-elevation cut.)
2. MEASURED bed change: mesh2d_mor_bl (last - first output) x cell area, per
   1-km bin, for channel and for bars/banks separately.
   Channel gain together with bar loss = sediment moved from bars to channel.
3. Time-mean seaward bed-load flux through the channel part and through the
   bar part of every column (from mesh2d_sxtot), and the time-mean net MPM
   capacity of the channel part (D-Morphology manual 8.213-8.216, before the
   bed-slope and spiral-flow corrections; directed like the model transport).
4. Along-channel deposition implied by the change of capacity (or of flux) in
   the flow direction. Measured minus this = what the along-channel flux does
   NOT explain, i.e. lateral supply from the bars.

Outputs (OUTPUT_BASE)
---------------------
channel_vs_bars_profiles   4 rows x (mean | peak flow): flux and capacity of the
                           channel, flux of the bars, capacity-implied deposition,
                           MEASURED channel and bar change [mm/day]
measured_dz_maps           measured bed change maps with the channel outline
channel_vs_bars_table.csv  per scenario and reach: areas, volumes, mm/day
channel_vs_bars_profiles.csv  per-bin values
Console: the same table, and the median channel flux / capacity.

Limitations: the detailed-hydro runs are short (MorFac = 1) so the measured
change is a tendency over the run, not a hydrograph-integrated result. Bed load
responds instantly: the flux cannot exceed the capacity, deposition follows from
convergence of capacity (along flow) or from lateral supply.
"""

# %% IMPORTS
from pathlib import Path
import csv
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Add repo root to path (one level up from script location)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions.F_general import get_variability_map
from functions.F_map_cache import (
    cache_tag_from_bbox,
    load_or_update_map_cache_multi,
    _get_face_coords,
)
from functions.F_loaddata import get_stitched_map_run_paths


# %% BASE PATHS
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

OUTPUT_BASE = RUN_DIR / "output_plots" / "sediment_channel_vs_bars"

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


# %% SETTINGS
# Zoom domain (x = distance along the estuary, 45 km = river side)
X_LIMITS_M = (19_000.0, 45_000.0)
Y_LIMITS_M = (5_000.0, 10_000.0)
FACE_MARGIN_M = 500.0            # faces read around the zoom (for the footprint)
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

BIN_M = 1000.0                   # along-estuary bin of the profiles
TABLE_REACHES_KM = ((20.0, 33.0), (33.0, 45.0), (20.0, 45.0))   # rows of the table

# Channel footprint
CHANNEL_DEFINITION = "wet"       # "wet" (wet at low water) | "bed_level"
WATER_LEVEL_VAR = "mesh2d_s1"
WET_DEPTH_THRESHOLD_M = 0.1      # wet = water level - bed level > this
WET_WINDOW_H = 12.0              # hours that define the water level
WET_WINDOW_AT = "last"           # "last" | "first"
WET_LEVEL_STAT = "min"           # "min" = channel only | "max" = also flooded bars
CHANNEL_BED_LEVEL_MAX_M = 2.2    # only for CHANNEL_DEFINITION = "bed_level"
BED_LEVEL_AT = "first"           # bed for the "bed_level" cut: "first" | "last"
CHANNEL_MAIN_ONLY = True         # largest connected component only

# Variables (MAP files)
SX_VAR = "mesh2d_sxtot"
SY_VAR = "mesh2d_sytot"
WATER_DEPTH_VAR = "mesh2d_waterdepth"
BED_LEVEL_VAR = "mesh2d_mor_bl"
UCMAG_VAR = "mesh2d_ucmag"
CHEZY_VAR = "mesh2d_czs"

# Transport units / sediment (sediment file: CDryB = 1600, RhoSol = 2650, MPM, 2e-4 m)
TRANSPORT_UNITS = "volume_solid"     # used only if the 'units' attribute is unclear
SEDIMENT_DENSITY_KG_M3 = 2650.0
DRY_BED_DENSITY_KG_M3 = 1600.0
POROSITY = 1.0 - DRY_BED_DENSITY_KG_M3 / SEDIMENT_DENSITY_KG_M3
BED_VOLUME_FACTOR = 1.0 / (1.0 - POROSITY)     # = RhoSol / CDryB
MORFAC = 1.0
POSITIVE_X_IS_SEAWARD = False    # river at ~44 km, distance counted from the sea
SEAWARD_SIGN = 1.0 if POSITIVE_X_IS_SEAWARD else -1.0

# MPM capacity
MPM_D50_M = 2.0e-4
MPM_D90_M = 2.0e-4
MPM_ALPHA = 1.0
MPM_THETA_CR = 0.047
MPM_XI = 1.0
RHO_WATER = 1000.0
GRAVITY = 9.81
MIN_WATER_DEPTH_M = 0.05

TIME_START_INDEX = 0
TIME_END_INDEX = None
TIME_CHUNK = 50

DPI = 300
SHOW_FIGURES = False
SCENARIO_COLORS = {
    "constant": "#7f7f7f",
    "pm3_n5_mean_flow": plt.cm.Blues(0.55),
    "pm5_n1_mean_flow": plt.cm.Blues(0.92),
    "pm3_n5_peak_flow": plt.cm.Oranges(0.55),
    "pm5_n1_peak_flow": plt.cm.Oranges(0.92),
}
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 8, "pdf.fonttype": 42}


# %% Helpers (copied unchanged from sediment_budget_analysis.py)
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


def short_name(name):
    return name.replace("_mean_flow", "").replace("_peak_flow", "")


# %% Channel footprint
def wet_or_bed_footprint(ds, da_bl, face_dim, face_idx, start, end, time, bed_first):
    """Boolean channel mask per face (before the connected-component filter)."""
    with np.errstate(invalid="ignore"):
        by_bed = bed_first < CHANNEL_BED_LEVEL_MAX_M
    text = f"bed level < {CHANNEL_BED_LEVEL_MAX_M:g} m ({BED_LEVEL_AT} output)"
    if CHANNEL_DEFINITION != "wet":
        return by_bed, text
    if WATER_LEVEL_VAR not in ds or "time" not in ds[WATER_LEVEL_VAR].dims:
        print(f"  [WARNING] '{WATER_LEVEL_VAR}' not available: using the "
              f"bed-level cut instead.")
        return by_bed, text

    n_times = time.size
    at_last = WET_WINDOW_AT == "last"
    hours = (time - (time[-1] if at_last else time[0])) / np.timedelta64(1, "h")
    win = np.flatnonzero(hours >= -WET_WINDOW_H) if at_last \
        else np.flatnonzero(hours <= WET_WINDOW_H)
    if win.size == 0:
        win = np.array([n_times - 1 if at_last else 0])
    s1 = read_block(ds[WATER_LEVEL_VAR].isel({face_dim: face_idx}), face_dim,
                    start + int(win[0]), start + int(win[-1]) + 1, fill_nan=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s1_ref = np.nanmin(s1, axis=0) if WET_LEVEL_STAT == "min" \
            else np.nanmax(s1, axis=0)
    t_bed = end - 1 if at_last else start     # bed at the same moment as the water level
    bed_ref = read_block(da_bl, face_dim, t_bed, t_bed + 1, fill_nan=False)[0]
    with np.errstate(invalid="ignore"):
        wet = (s1_ref - bed_ref) > WET_DEPTH_THRESHOLD_M
    text = (f"wet at {WET_LEVEL_STAT} water level (depth > "
            f"{WET_DEPTH_THRESHOLD_M:g} m, {WET_WINDOW_AT} {WET_WINDOW_H:g} h)")
    return wet, text


def keep_main_component(mask, ix, iy, ny, nx):
    """Largest 8-connected component of a face mask -> (main, excluded) per face."""
    grid = np.zeros((ny, nx), dtype=bool)
    grid[iy, ix] = mask
    labels, n_comp = ndimage.label(grid, structure=np.ones((3, 3)))
    if n_comp <= 1:
        return mask, np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == int(np.argmax(sizes))
    return main[iy, ix], (grid & ~main)[iy, ix]


# %% Analysis of one scenario
def analyse_dataset(ds, name):
    all_x, all_y = _get_face_coords(ds)
    all_x = np.asarray(all_x, dtype=float)
    all_y = np.asarray(all_y, dtype=float)
    face_dim = get_face_dimension(ds, SX_VAR, all_x.size)

    keep = ((all_x >= X_LIMITS_M[0] - FACE_MARGIN_M) & (all_x <= X_LIMITS_M[1] + FACE_MARGIN_M)
            & (all_y >= Y_LIMITS_M[0] - FACE_MARGIN_M) & (all_y <= Y_LIMITS_M[1] + FACE_MARGIN_M))
    face_idx = np.flatnonzero(keep)
    if face_idx.size == 0:
        raise ValueError("No faces inside the analysis domain.")
    face_x, face_y = all_x[face_idx], all_y[face_idx]
    n_faces = face_idx.size

    grid = native_grid(face_x, face_y)
    if grid is None:
        raise ValueError("Faces do not form a rectilinear grid (needed for the footprint).")
    ix, iy, ux, uy, dx, dy = grid
    ny, nx = uy.size, ux.size
    cell_area = dx[ix] * dy[iy]

    # --- time ---
    time = np.asarray(ds["time"].values).astype("datetime64[ns]")
    start = TIME_START_INDEX
    end = time.size if TIME_END_INDEX is None else TIME_END_INDEX
    if not (0 <= start < end - 1 <= time.size - 1):
        raise ValueError("Need at least 2 timesteps in the time selection.")
    time = time[start:end]
    n_times = time.size
    t_s = ((time - time[0]) / np.timedelta64(1, "s")).astype(float)
    if np.any(np.diff(t_s) < 0):
        raise ValueError("Time axis is not monotonic (overlap between stitched runs?).")
    duration = (t_s[-1] - t_s[0]) * MORFAC
    weights = trapezoid_weights(t_s * MORFAC)
    print(f"  Time span: {t_s[-1] / 3600:.1f} h ({n_times} outputs); {n_faces} faces")

    # --- measured bed change and footprint ---
    if BED_LEVEL_VAR not in ds or "time" not in ds[BED_LEVEL_VAR].dims:
        raise KeyError(f"'{BED_LEVEL_VAR}' with a time axis is required.")
    da_bl = ds[BED_LEVEL_VAR].isel({face_dim: face_idx})
    bed_a = read_block(da_bl, face_dim, start, start + 1, fill_nan=False)[0]
    bed_b = read_block(da_bl, face_dim, end - 1, end, fill_nan=False)[0]
    dz = bed_b - bed_a
    if np.isfinite(dz).any() and np.nanmax(np.abs(dz)) < 1e-6:
        print("  [WARNING] the bed did not change: fixed-bed run? The measured "
              "columns below are then meaningless.")
    bed_first = bed_a if BED_LEVEL_AT == "first" else bed_b
    raw, definition = wet_or_bed_footprint(ds, da_bl, face_dim, face_idx, start,
                                           end, time, bed_first)
    raw = np.asarray(raw, dtype=bool)
    if CHANNEL_MAIN_ONLY:
        channel, pits = keep_main_component(raw, ix, iy, ny, nx)
    else:
        channel, pits = raw, np.zeros_like(raw)
    in_win = ((face_x >= X_LIMITS_M[0]) & (face_x <= X_LIMITS_M[1])
              & (face_y >= Y_LIMITS_M[0]) & (face_y <= Y_LIMITS_M[1]))
    bars = ~channel & ~pits & in_win & np.isfinite(dz)
    chan = channel & in_win & np.isfinite(dz)
    print(f"  Channel = {definition}: {(cell_area * chan).sum() / 1e6:.2f} km2, "
          f"bars/banks {(cell_area * bars).sum() / 1e6:.2f} km2, "
          f"isolated pits excluded {(cell_area * (pits & in_win)).sum() / 1e6:.2f} km2")

    # --- stream: time-mean transport and net capacity ---
    units_mode = resolve_transport_units(ds, SX_VAR)
    factor = bed_volume_factor(units_mode)
    have_cap = all(v in ds for v in (CHEZY_VAR, UCMAG_VAR))
    if not have_cap:
        print(f"  [WARNING] {CHEZY_VAR}/{UCMAG_VAR} missing: capacity skipped.")
    da_x = ds[SX_VAR].isel({face_dim: face_idx})
    da_y = ds[SY_VAR].isel({face_dim: face_idx})
    da_d = ds[WATER_DEPTH_VAR].isel({face_dim: face_idx}) if WATER_DEPTH_VAR in ds else None
    if have_cap:
        da_c = ds[CHEZY_VAR].isel({face_dim: face_idx})
        da_u = ds[UCMAG_VAR].isel({face_dim: face_idx})
    net_x = np.zeros(n_faces)
    capx = np.zeros(n_faces)
    for s in range(0, n_times, TIME_CHUNK):
        e = min(s + TIME_CHUNK, n_times)
        sx = read_block(da_x, face_dim, start + s, start + e) * factor
        sy = read_block(da_y, face_dim, start + s, start + e) * factor
        w = weights[s:e]
        net_x += w @ sx
        if have_cap:
            depth = read_block(da_d, face_dim, start + s, start + e) \
                if da_d is not None else np.full_like(sx, np.inf)
            chezy = read_block(da_c, face_dim, start + s, start + e)
            ucmag = read_block(da_u, face_dim, start + s, start + e)
            with np.errstate(invalid="ignore", divide="ignore"):
                tau = np.where(chezy > 0, RHO_WATER * GRAVITY * (ucmag / chezy) ** 2, 0.0)
            cap = mpm_capacity(tau, chezy, depth)
            mag = np.hypot(sx, sy)
            dir_x = np.divide(sx, mag, out=np.zeros_like(sx), where=mag > 0)
            capx += w @ (cap * dir_x)
    mean_x = net_x / duration
    cap_x = capx / duration if have_cap else None

    # --- per-bin profiles ---
    edges = np.arange(X_LIMITS_M[0], X_LIMITS_M[1] + 0.5 * BIN_M, BIN_M)
    nb = edges.size - 1
    x_km = 0.5 * (edges[:-1] + edges[1:]) / 1000.0
    fbin = np.digitize(face_x, edges) - 1
    cbin = np.digitize(ux, edges) - 1
    n_cols = np.bincount(cbin[(cbin >= 0) & (cbin < nb)], minlength=nb)
    n_cols = np.maximum(n_cols, 1)
    dy_f = dy[iy]

    def per_bin(sel, values):
        ok = sel & (fbin >= 0) & (fbin < nb)
        return np.bincount(fbin[ok], weights=values[ok], minlength=nb)

    area_ch = per_bin(chan, cell_area)
    area_ba = per_bin(bars, cell_area)
    vol_ch = per_bin(chan, dz * cell_area)
    vol_ba = per_bin(bars, dz * cell_area)
    days = duration / MORFAC / 86400.0
    with np.errstate(invalid="ignore", divide="ignore"):
        meas_ch_mm = np.where(area_ch > 0, vol_ch / area_ch * 1000.0 / days, np.nan)
        meas_ba_mm = np.where(area_ba > 0, vol_ba / area_ba * 1000.0 / days, np.nan)

    prof = {
        "x_km": x_km, "area_channel_m2": area_ch, "area_bars_m2": area_ba,
        "vol_channel_m3": vol_ch, "vol_bars_m3": vol_ba,
        "measured_channel_mm_day": meas_ch_mm, "measured_bars_mm_day": meas_ba_mm,
        "flux_channel_seaward": SEAWARD_SIGN * per_bin(chan, mean_x * dy_f) / n_cols,
        "flux_bars_seaward": SEAWARD_SIGN * per_bin(bars, mean_x * dy_f) / n_cols,
        "definition": definition, "days": days,
    }
    to_mm_day = MORFAC * 86400.0 * 1000.0
    x_m = x_km * 1000.0

    def along(f_seaward):
        """Deposition in the channel implied by the change of a seaward flux
        in the flow direction [mm/day]: +dF_seaward/dx for a landward-positive x."""
        valid = np.isfinite(f_seaward)
        out = np.full(nb, np.nan)
        if valid.sum() >= 3:
            g = np.full(nb, np.nan)
            g[valid] = np.gradient(f_seaward[valid], x_m[valid])
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(area_ch > 0, -SEAWARD_SIGN * g * BIN_M / area_ch * to_mm_day, np.nan)
        return out

    prof["dep_flux_mm_day"] = along(np.where(area_ch > 0, prof["flux_channel_seaward"], np.nan))
    if cap_x is not None:
        prof["cap_channel_seaward"] = SEAWARD_SIGN * per_bin(chan, cap_x * dy_f) / n_cols
        prof["dep_cap_mm_day"] = along(np.where(area_ch > 0, prof["cap_channel_seaward"], np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            prof["flux_over_cap"] = np.where(
                (area_ch > 0) & (prof["cap_channel_seaward"] > 0),
                prof["flux_channel_seaward"] / prof["cap_channel_seaward"], np.nan)

    maps = {"ux": ux, "uy": uy, "dz": None, "channel": None}
    dz_g = np.full((ny, nx), np.nan)
    dz_g[iy, ix] = dz
    ch_g = np.zeros((ny, nx))
    ch_g[iy, ix] = channel
    maps.update(dz=dz_g, channel=ch_g)
    return {"name": name, "profile": prof, "maps": maps, "definition": definition}


def analyse_scenario(scenario_name, folder_path):
    folder = Path(folder_path)
    run_paths = get_stitched_map_run_paths(
        base_path=BASE_PATH, folder_name=folder.name, timed_out_dir=TIMED_OUT_DIR,
        variability_map=VARIABILITY_MAP, analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder]
    ds = load_or_update_map_cache_multi(
        cache_dir=ASSESSMENT_DIR, folder_name=folder.name, run_paths=run_paths,
        var_names=[SX_VAR, SY_VAR, WATER_DEPTH_VAR, BED_LEVEL_VAR, UCMAG_VAR, CHEZY_VAR]
        + ([WATER_LEVEL_VAR] if CHANNEL_DEFINITION == "wet" else []),
        bbox=CACHE_BBOX, append_time=APPEND_TIMESTEPS, append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        print(f"  No cached data for {folder.name}, skipping.")
        return None
    try:
        return analyse_dataset(ds, scenario_name)
    finally:
        ds.close()


# %% Table and CSV
def reach_row(p, lo, hi):
    sel = (p["x_km"] >= lo) & (p["x_km"] < hi)
    a_ch, a_ba = p["area_channel_m2"][sel].sum(), p["area_bars_m2"][sel].sum()
    v_ch, v_ba = p["vol_channel_m3"][sel].sum(), p["vol_bars_m3"][sel].sum()
    d = p["days"]
    return {
        "area_channel_km2": a_ch / 1e6, "area_bars_km2": a_ba / 1e6,
        "channel_m3_day": v_ch / d, "bars_m3_day": v_ba / d, "sum_m3_day": (v_ch + v_ba) / d,
        "channel_mm_day": v_ch / a_ch * 1000.0 / d if a_ch > 0 else np.nan,
        "bars_mm_day": v_ba / a_ba * 1000.0 / d if a_ba > 0 else np.nan,
    }


def print_and_export_table(results):
    print("\n" + "=" * 100)
    print("MEASURED bed-volume change (mesh2d_mor_bl, last - first output); + = gain.")
    print("Channel gain with bar loss = sediment moved from bars/banks into the channel.")
    print("=" * 100)
    head = (f"{'scenario':18s}{'reach km':>10s}{'channel km2':>12s}{'bars km2':>10s}"
            f"{'chan m3/d':>11s}{'bars m3/d':>11s}{'sum m3/d':>10s}"
            f"{'chan mm/d':>11s}{'bars mm/d':>11s}")
    print(head)
    rows = []
    for name, r in results.items():
        for lo, hi in TABLE_REACHES_KM:
            q = reach_row(r["profile"], lo, hi)
            print(f"{name:18s}{lo:5.0f}-{hi:<4.0f}{q['area_channel_km2']:12.2f}"
                  f"{q['area_bars_km2']:10.2f}{q['channel_m3_day']:11.1f}"
                  f"{q['bars_m3_day']:11.1f}{q['sum_m3_day']:10.1f}"
                  f"{q['channel_mm_day']:11.2f}{q['bars_mm_day']:11.2f}")
            rows.append([name, lo, hi] + [q[k] for k in (
                "area_channel_km2", "area_bars_km2", "channel_m3_day", "bars_m3_day",
                "sum_m3_day", "channel_mm_day", "bars_mm_day")])
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_BASE / "channel_vs_bars_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "reach_from_km", "reach_to_km", "area_channel_km2",
                    "area_bars_km2", "channel_m3_per_day", "bars_m3_per_day",
                    "sum_m3_per_day", "channel_mm_per_day", "bars_mm_per_day"])
        w.writerows(rows)

    print("\nMedian channel flux / net MPM capacity (bins with channel and capacity):")
    for name, r in results.items():
        fc = r["profile"].get("flux_over_cap")
        if fc is not None and np.isfinite(fc).any():
            print(f"  {name:18s}{np.nanmedian(fc):7.2f}")
    print("\nChannel footprints:")
    for name, r in results.items():
        print(f"  {name:18s}{r['definition']}")

    keys = ["area_channel_m2", "area_bars_m2", "vol_channel_m3", "vol_bars_m3",
            "measured_channel_mm_day", "measured_bars_mm_day", "flux_channel_seaward",
            "flux_bars_seaward", "cap_channel_seaward", "flux_over_cap",
            "dep_flux_mm_day", "dep_cap_mm_day"]
    with (OUTPUT_BASE / "channel_vs_bars_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "x_km"] + keys)
        for name, r in results.items():
            p = r["profile"]
            for i, xk in enumerate(p["x_km"]):
                w.writerow([name, f"{xk:.2f}"] + [
                    "" if k not in p or not np.isfinite(p[k][i]) else f"{p[k][i]:.6g}"
                    for k in keys])


# %% Figures
def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {path.name}.png / .pdf")


def plot_profiles(results):
    """4 rows x (mean | peak flow). Solid = flux or measured channel,
    dashed = capacity or flux-implied, dotted = bars."""
    rows = [
        "Channel transport\n[m$^3$ s$^{-1}$]",
        "Bars/banks transport\n[m$^3$ s$^{-1}$]",
        "Along-channel\ndeposition [mm day$^{-1}$]",
        "MEASURED change\n[mm day$^{-1}$]",
    ]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(170 / 25.4, 200 / 25.4),
                                 sharex=True, sharey="row")
        letters = iter("abcdefgh")
        for c, (flow, names) in enumerate(COMPARISON_PAIRS.items()):
            for name in names:
                r = results.get(name)
                if r is None:
                    continue
                p, col = r["profile"], SCENARIO_COLORS[name]
                lab = short_name(name)
                axes[0, c].plot(p["x_km"], p["flux_channel_seaward"], color=col, lw=1.3, label=lab)
                if "cap_channel_seaward" in p:
                    axes[0, c].plot(p["x_km"], p["cap_channel_seaward"], color=col, lw=1.0, ls="--")
                    axes[2, c].plot(p["x_km"], p["dep_cap_mm_day"], color=col, lw=1.3)
                axes[1, c].plot(p["x_km"], p["flux_bars_seaward"], color=col, lw=1.3)
                axes[2, c].plot(p["x_km"], p["dep_flux_mm_day"], color=col, lw=1.0, ls="--")
                axes[3, c].plot(p["x_km"], p["measured_channel_mm_day"], color=col, lw=1.3)
                axes[3, c].plot(p["x_km"], p["measured_bars_mm_day"], color=col, lw=1.0, ls=":")
            for k in range(4):
                ax = axes[k, c]
                ax.axhline(0, color="black", linewidth=0.6)
                ax.grid(alpha=0.3)
                ax.text(0.02, 0.96, f"({next(letters)})", transform=ax.transAxes,
                        va="top", fontweight="bold")
                if c == 0:
                    ax.set_ylabel(rows[k])
            axes[0, c].set_title(flow.replace("_", " ").title())
            axes[3, c].set_xlabel("Distance along the estuary [km]")
        axes[0, 0].legend(frameon=False, fontsize=7, loc="center right")
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        fig.text(0.5, -0.005, "Line styles:  top row solid = model transport, dashed = "
                 "MPM capacity;  third row solid = from capacity, dashed = from flux;\n"
                 "bottom row solid = channel, dotted = bars/banks (+ = infilling).  "
                 "Colours as in the legend.", ha="center", va="top", fontsize=6.5,
                 color="0.3")
        save_figure(fig, OUTPUT_BASE / "channel_vs_bars_profiles")


def plot_maps(results):
    names = [n for n in SCENARIOS if n in results]
    if not names:
        return
    vmax = max(np.nanpercentile(np.abs(results[n]["maps"]["dz"]), 99) for n in names)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1e-3
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(names), 1, sharex=True, constrained_layout=True,
                                 figsize=(170 / 25.4, (30 * len(names) + 15) / 25.4))
        axes = np.atleast_1d(axes)
        mesh = None
        for ax, name in zip(axes, names):
            m = results[name]["maps"]
            mesh = ax.pcolormesh(m["ux"], m["uy"], np.ma.masked_invalid(m["dz"]),
                                 cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest",
                                 rasterized=True)
            ax.contour(m["ux"], m["uy"], m["channel"], levels=[0.5],
                       colors="limegreen", linewidths=0.7)
            ax.set_xlim(X_LIMITS_M)
            ax.set_ylim(Y_LIMITS_M)
            ax.set_aspect("equal")
            ax.set_ylabel("y [m]")
            ax.set_title(f"{name}  (green: channel footprint)", fontsize=8, loc="left")
        axes[-1].set_xlabel("x [m]")
        fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.01,
                     label="Measured bed change, last - first output [m] (+ = infilling)")
        save_figure(fig, OUTPUT_BASE / "measured_dz_maps")



def edge_flux(p, x_edge_km):
    """Total (channel + bars) time-mean seaward flux at a bin edge [m3/s]:
    mean of the two neighbouring bins (the last edge uses the last bin)."""
    tot = p["flux_channel_seaward"] + p["flux_bars_seaward"]
    i = int(np.argmin(np.abs(p["x_km"] + 0.5 * BIN_M / 1000.0 - x_edge_km)))   # bin below the edge
    vals = [tot[j] for j in (i, i + 1) if 0 <= j < tot.size and np.isfinite(tot[j])]
    return float(np.mean(vals)) if vals else np.nan


def plot_mechanism(results, reaches=((20.0, 33.0), (33.0, 45.0)),
                   scen=("pm3_n5_mean_flow", "pm5_n1_mean_flow",
                         "pm3_n5_peak_flow", "pm5_n1_peak_flow")):
    """Evidence figure: (a) measured volume change of channel and bars per reach,
    (b) closure: measured net change vs the change implied by the section fluxes,
    (c) channel transport vs capacity at peak flow with the two reaches marked."""
    scen = [n for n in scen if n in results]
    if not scen:
        return
    c_ch, c_ba = "#2a9d8f", "#e9c46a"
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(170 / 25.4, 150 / 25.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.42)
        ax = fig.add_subplot(gs[0, :])
        ticks, labels, x = [], [], 0.0
        for lo, hi in reaches:
            for n in scen:
                q = reach_row(results[n]["profile"], lo, hi)
                ax.bar(x - 0.27, q["channel_m3_day"], 0.26, color=c_ch,
                       label="Channel" if x == 0 else None)
                ax.bar(x, q["bars_m3_day"], 0.26, color=c_ba,
                       label="Bars/banks" if x == 0 else None)
                ax.plot(x + 0.27, q["sum_m3_day"], "kD", ms=4,
                        label="Net (channel + bars)" if x == 0 else None)
                ticks.append(x)
                labels.append(short_name(n) + ("\npeak" if "peak" in n else "\nmean"))
                x += 1.0
            ax.text(x - 0.5 * len(scen) - 0.5, 1.02, f"km {lo:.0f}-{hi:.0f}",
                    transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
            ax.axvline(x - 0.5, color="0.6", lw=0.6)
            x += 0.4
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Measured bed-volume change\n[m$^3$ day$^{-1}$]  (+ = gain)")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2))
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("(a)", loc="left", fontsize=8, fontweight="bold", pad=14)

        axb = fig.add_subplot(gs[1, 0])
        lim = 1.0
        for n in scen:
            p = results[n]["profile"]
            for lo, hi in list(reaches) + [(reaches[0][0], reaches[-1][1])]:
                q = reach_row(p, lo, hi)
                implied = (edge_flux(p, hi) - edge_flux(p, lo)) * MORFAC * 86400.0
                axb.plot(implied, q["sum_m3_day"], "o", ms=5, color=SCENARIO_COLORS[n],
                         mfc=SCENARIO_COLORS[n] if (lo, hi) != (reaches[0][0], reaches[-1][1]) else "none",
                         label=short_name(n) + (" peak" if "peak" in n else " mean"))
                lim = max(lim, abs(implied), abs(q["sum_m3_day"]))
        axb.plot([-lim, lim], [-lim, lim], color="black", lw=0.6)
        axb.axhline(0, color="0.7", lw=0.5)
        axb.axvline(0, color="0.7", lw=0.5)
        axb.set_xlabel("Implied by section fluxes\n$F(\\mathrm{upstream})-F(\\mathrm{downstream})$ [m$^3$ day$^{-1}$]")
        axb.set_ylabel("Measured net change [m$^3$ day$^{-1}$]")
        axb.set_title("(b)  filled: one reach, open: km 20-45", loc="left", fontsize=8,
                      fontweight="bold")
        axb.locator_params(axis="x", nbins=4)
        from matplotlib.lines import Line2D
        axb.legend([Line2D([], [], marker="o", ls="", color=SCENARIO_COLORS[n]) for n in scen],
                   [short_name(n) + (" peak" if "peak" in n else " mean") for n in scen],
                   frameon=False, fontsize=6, loc="lower right")
        axb.grid(alpha=0.3)

        axc = fig.add_subplot(gs[1, 1])
        for n in [m for m in scen if "peak" in m]:
            p, col = results[n]["profile"], SCENARIO_COLORS[n]
            axc.plot(p["x_km"], p["flux_channel_seaward"], color=col, lw=1.4,
                     label=short_name(n) + " model")
            if "cap_channel_seaward" in p:
                axc.plot(p["x_km"], p["cap_channel_seaward"], color=col, lw=1.0, ls="--",
                         label=short_name(n) + " capacity")
        for (lo, hi), shade in zip(reaches, ("#2a9d8f", "#e9c46a")):
            axc.axvspan(lo, hi, color=shade, alpha=0.12, lw=0)
        axc.set_xlabel("Distance along the estuary [km]")
        axc.set_ylabel("Channel transport, peak flow\n[m$^3$ s$^{-1}$]")
        axc.set_title("(c)  shaded: the two reaches", loc="left", fontsize=8, fontweight="bold")
        axc.legend(frameon=False, fontsize=6, loc="upper left")
        axc.grid(alpha=0.3)
        save_figure(fig, OUTPUT_BASE / "peak_flow_mechanism")


# %% Run all scenarios
results = {}
for scenario_name, folder_path in SCENARIOS.items():
    print("\n" + "=" * 80)
    print(f"Processing: {scenario_name}\nFolder: {folder_path}")
    print("=" * 80)
    r = analyse_scenario(scenario_name, folder_path)
    if r is not None:
        results[scenario_name] = r

if results:
    print_and_export_table(results)
    plot_profiles(results)
    plot_maps(results)
    plot_mechanism(results)
else:
    print("No scenario could be analysed.")

# %% Finish
print(f"\nOutput written to: {OUTPUT_BASE}")