# %% Fig. 3: morphology of the constant and the R_peak = 5 scenario
#    AND the hydrodynamics of the final morphology
"""
Opening figure of the results section. Constant run = 01_Qr500_pm1_n0, peak scenario
= 05_Qr500_pm5_n1 (R_peak = 5, n_peaks = 1).

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide):
(a)   discharge at km20 (downstream, sea) and km45 (upstream, river), last 24 h of the
      detailed-hydro runs (positive = seaward, i.e. towards decreasing x)
(b)   one-year hydrograph of the R_peak = 5 scenario at km45, with the mean and peak flow
(c)   sediment budget of the constant run: net bed-load export per tidal cycle at
      km20 divided by the river supply at km45 (his cross-sections), vs morphological
      time
(d)   initial bed level, with the positive flow direction
(e-f) detrended bed level at the end of the constant and the R_peak = 5 run
(g)   tidal range along the estuary (per-face max - min water level over the final 12 h,
      width-averaged over the faces that stay wet) on the final beds, same scenarios as (a)

Detailed-hydro runs: 2 days of hydrodynamics on the FINAL bed; their own dates
2031-01-01 ... 01-03 are NOT on the morphodynamic time axis, so hours are shown.
Colours as in fig_bedlevel_percentiles_pm_n.py: grey = constant, dark blue = R_peak = 5.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from functions.F_general import (
    apply_plot_style,
    create_terrain_colormap,
    setup_variability_run_context,
)
from functions.F_dynamic_equilibrium import equilibrium_time, running_mean, tidal_cycle_budget
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_morphological_activity import build_centerline_reference

# harmless: xugrid CRS notices (no CRS in the D-Flow FM files) and all-NaN slices
# (dry faces in the final-12-h min/max; the thalweg faces are always wet)
warnings.filterwarnings("ignore", category=UserWarning, module="xugrid")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

# %% --- CONFIGURATION ---------------------------------------------------------

DISCHARGE = 500
BASE_DIRECTORY = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
)
run_context = setup_variability_run_context(
    base_directory=BASE_DIRECTORY, discharge=DISCHARGE, scenarios_to_process=None,
    analyze_noisy=False, MORFAC=False,
)
BASE_PATH = run_context["base_path"]
CACHE_DIR = run_context["cache_dir"]
TIMED_OUT_DIR = run_context["timed_out_dir"]

# Base run: constant discharge
MORPH_RUN = BASE_DIRECTORY / "Model_Output" / "Q500" / "01_Qr500_pm1_n0.9600302"
DETAILED_ROOT = BASE_DIRECTORY / "Model_Output" / "Q500" / "detailed-hydro-run"
# constant scenario: Q = Qmean at all times
DETAILED_RUN = DETAILED_ROOT / "dhr_01_Qr500_pm1_n0.9724783"
# peaky scenario (pm5_n1): same final-morphology, run at the constant flow and at a peak
MEAN_DETAILED_RUN = DETAILED_ROOT / "dhr_05_Qr500_pm5_n1_mean.964908"
PEAK_DETAILED_RUN = DETAILED_ROOT / "dhr_05_Qr500_pm5_n1_peak.965221"
# morphodynamic run of the peaky scenario, only used for the one-year hydrograph in (f)
HYDRO_MORPH_RUN = BASE_DIRECTORY / "Model_Output" / "Q500" / "05_Qr500_pm5_n1.9517572"
HYDROGRAPH_KM = 45
YEAR_DAYS_HYDROGRAPH = 365.25
YEAR_DAYS = 365.25
# same colours as fig_bedlevel_percentiles_pm_n.py: grey constant, Blues for R_peak (R_peak = 5 darkest)
COLOR_CONST = "#7f7f7f"
COLOR_PEAK = plt.cm.Blues(0.90)
COLOR_MEAN = "#e08214"      # R_peak = 5 morphology at the constant-scenario flow (own hue, not an R_peak shade)
RPEAK = r"$R_{\mathrm{peak}}$"
NPEAKS = r"$n_{\mathrm{peaks}}$"
LAST_HOURS = 24

BED_VAR = "mesh2d_mor_bl"
WL_VAR = "mesh2d_s1"
HIS_FILENAME = "output/FlowFM_0000_his.nc"
HIS_Q_VAR = "cross_section_discharge"
# dynamic equilibrium from the his-file sediment budget (functions/F_dynamic_equilibrium.py;
# full analysis incl. the R_peak = 5 run: plot_his_dynamic_equilibrium.py)
CDRYB = 1600.0              # dry bed density [kg/m3] (.sed file): kg -> m3 of bed
EQ_SEA_KM, EQ_RIVER_KM = 20, 45   # sediment budget of the estuary between these sections
EQ_SMOOTH_YEARS = 10.0      # running mean of the per-tidal-cycle fluxes [morphological years]
EQ_TOL = 0.10               # equilibrium: |net export - river supply| < EQ_TOL x river supply

# The time axis of the morphodynamic map output is used exactly as it is (datetime64),
# as in the map-plot script. One "year" = one year on that axis.
# Base run = constant discharge, so there is no hydrograph phase to match: every map
# output time after spin-up is a valid sample of the dynamic equilibrium.
MORFAC_FACTOR = 100         # 6 hydrodynamic years x 100 = 600 morphological years
SPINUP_YEARS = 50.0         # end of spin-up in MORPHOLOGICAL years (placeholder; read from panel e)
MAX_SAMPLES = 150           # at most this many map times are evaluated (evenly spread)

CACHE_BBOX = [1, 1, 45000, 15000]
X_KM = np.arange(20, 46)
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y = 500
WET_DEPTH_THRESHOLD_M = 0.1

REACHES = {
    "tidal (km 20-25)":   (20, 25),
    "middle (km 30-35)":  (30, 35),
    "fluvial (km 40-45)": (40, 45),
}
REACH_COLORS = ["#08519c", "#3182bd", "#9ecae1"]
COLOR = "#08519c"

DETREND_LAND_THRESHOLD = 6.0
CENTERLINE_XMIN, CENTERLINE_XMAX, CENTERLINE_Y = 20000, 45000, 7500
BED_CMAP = create_terrain_colormap()
BED_LIM = 5
BED_LIM_INITIAL = 2         # colour limits of the actual initial bed: -2 ... 2 m

MAP_XLIM = (20_000, 45_000)
MAP_YLIM = (5_000, 10_000)
RANGE_BIN_KM = 0.5          # along-estuary bins of the width-averaged tidal range

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170       # AGU full-page width
FONT_SIZE = 8               # one size for all text: ticks, labels, titles, legend

# Layout in inches, so the panels fill the full width at the final print size
LEFT_MARGIN_IN = 0.45       # y tick labels + y-label
RIGHT_MARGIN_IN = 0.08      # small buffer (line panels)
CBAR_PAD_IN = 0.08          # map -> colorbar
CBAR_W_IN = 0.10
CBAR_LABEL_IN = 0.42        # colorbar tick labels + label
LINE_WSPACE_IN = 0.55       # gap between line panels (each has its own y-label)
LINE_PANEL_H_IN = 1.45
LEGEND_H_IN = 0.40          # shared legend above the line panels
LINE_XLABEL_IN = 0.45       # x tick labels + x-label below the line panels
MAP_HSPACE_IN = 0.06        # gap between stacked maps
MAP_XLABEL_IN = 0.36        # x tick labels + x-label below a map block
RANGE_GAP_IN = 0.10         # last map -> tidal-range panel (shared x axis)
RANGE_PANEL_H_IN = 1.10

OUTPUT_DIR = BASE_DIRECTORY / "Model_Output" / "output_plots_combined" / "morphology_overview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

apply_plot_style("AGU", font_size=FONT_SIZE, title_delta=0)

# %% --- FUNCTIONS -------------------------------------------------------------


def load_ds(run_dir, var_names, cache_dir, timed_out_dir=None):
    name = run_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(base_path=run_dir.parent, folder_name=name,
                  timed_out_dir=timed_out_dir, analyze_noisy=False)
    if timed_out_dir is None:                           # detailed-hydro run, as in the cross-section script
        kwargs["variability_map"] = None
    run_paths = get_stitched_map_run_paths(**kwargs) or [run_dir]   # morph run: as in the map-plot script
    ds = load_or_update_map_cache_multi(
        cache_dir=cache_dir, folder_name=name, run_paths=run_paths,
        var_names=var_names, bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, None),
    )
    if ds is None:
        raise FileNotFoundError(f"No cached map data for {name}")
    # the cache appends time blocks out of order (e.g. 2031-01-01 at index 5): sort, drop duplicates
    t = np.asarray(ds.time.values)
    if np.any(np.diff(t) <= np.timedelta64(0)):
        ds = ds.isel(time=np.unique(t, return_index=True)[1])
    return ds


def face_dim(ds, var):
    dims = [d for d in ds[var].dims if d != "time"]
    assert len(dims) == 1, f"Expected one face dim in {var}, got {ds[var].dims}"
    return dims[0]


def build_section_index(ds):
    """Nearest-face index per (cross-section, y-sample); built once per grid."""
    fc = np.asarray(ds.grid.face_coordinates, dtype=float)
    tree = cKDTree(fc[:, :2])
    y = np.linspace(*Y_RANGE_M, N_Y)
    idx = np.empty((len(X_KM), N_Y), dtype=int)
    for i, xk in enumerate(X_KM):
        idx[i] = tree.query(np.column_stack((np.full(N_Y, xk * 1000.0), y)))[1]
    return y, idx


def count_channels(bed_sec, wl_sec, thr=WET_DEPTH_THRESHOLD_M):
    """Number of separate wet channels per cross-section. Inputs: (n_sections, n_y)."""
    depth = wl_sec - bed_sec
    wet = np.isfinite(depth) & (depth > thr)
    return wet[:, 0].astype(int) + (wet[:, 1:] & ~wet[:, :-1]).sum(axis=1)


def segment_mean(counts):
    """1-km segments: mean of the two bounding cross-sections."""
    return 0.5 * (counts[..., :-1] + counts[..., 1:])


def sample_indices(times):
    """Evenly spread subset of the map output times; returns (indices, MORPHOLOGICAL years since start)."""
    idx = np.unique(np.linspace(0, len(times) - 1, min(len(times), MAX_SAMPLES)).astype(int))
    hydro_days = (times[idx] - times[0]) / np.timedelta64(1, "D")
    return idx, hydro_days * MORFAC_FACTOR / YEAR_DAYS


def detailed_profiles(run_dir):
    """
    Hydrodynamics of the detailed-hydro run on its (final) bed.
    Returns per-section thalweg quantities from the final 12 h, the fixed reference stage
    wl_min_full and the tidal range along the estuary (range_x_km, range_x): per-face
    max - min water level over the final 12 h (one tidal cycle), averaged per RANGE_BIN_KM
    bin over the faces that stay wet (falling-dry faces would clip the range at the bed).
    """
    ds = load_ds(run_dir, [WL_VAR, BED_VAR], cache_dir=run_dir.parent / "cached_data")
    try:
        fd_bed, fd_wl = face_dim(ds, BED_VAR), face_dim(ds, WL_VAR)
        assert fd_bed == fd_wl, f"{BED_VAR} and {WL_VAR} on different faces"
        _, idx = build_section_index(ds)
        t = np.asarray(ds.time.values).astype("datetime64[ns]")
        sel = np.where(t >= t[-1] - np.timedelta64(12, "h"))[0]
        wl_fin = ds[WL_VAR].isel(time=sel).transpose("time", fd_wl).values
        bed_faces = ds[BED_VAR].isel(time=-1).transpose(fd_bed).values
        wl_hi, wl_lo = np.nanmax(wl_fin, axis=0), np.nanmin(wl_fin, axis=0)
        fc = np.asarray(ds.grid.face_coordinates, dtype=float)
        edges = np.arange(MAP_XLIM[0], MAP_XLIM[1] + 1, RANGE_BIN_KM * 1000)
        keep = ((wl_lo - bed_faces) > WET_DEPTH_THRESHOLD_M) & (fc[:, 1] >= Y_RANGE_M[0]) & (fc[:, 1] <= Y_RANGE_M[1])
        b = np.digitize(fc[keep, 0], edges) - 1
        ok = (b >= 0) & (b < len(edges) - 1)
        n = np.bincount(b[ok], minlength=len(edges) - 1)
        s = np.bincount(b[ok], weights=(wl_hi - wl_lo)[keep][ok], minlength=len(edges) - 1)
        range_x = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        wl_max = np.nanmax(wl_fin, axis=0)[idx]
        wl_min = np.nanmin(wl_fin, axis=0)[idx]
        bed = bed_faces[idx]
        rows = np.arange(len(X_KM))
        ith = np.nanargmin(bed, axis=1)                    # thalweg sample per section
        faces = idx[rows, ith]
        bed_t, wmax, wmin = bed[rows, ith], wl_max[rows, ith], wl_min[rows, ith]
        return dict(
            bed=bed_t, wl_max=wmax, wl_min=wmin,
            depth_max=np.clip(wmax - bed_t, 0, None),
            depth_min=np.clip(wmin - bed_t, 0, None),
            wl_min_full=wl_min, bed_full=bed,
            range_x_km=0.5 * (edges[:-1] + edges[1:]) / 1000, range_x=range_x,
        )
    finally:
        ds.close()


def _decode(name_array):
    out = []
    for item in np.asarray(name_array):
        if isinstance(item, bytes):
            s = item.decode("utf-8", errors="ignore")
        elif np.ndim(item) == 0:
            s = str(item)
        else:
            s = "".join(c.decode("utf-8", errors="ignore") if isinstance(c, bytes) else str(c)
                        for c in item)
        out.append(s.replace("\x00", "").strip())
    return out


def his_discharge(his_path, km):
    """Time (datetime64) and discharge at ObservationCrossSection_Estuary_km{km}."""
    ds = xr.open_dataset(his_path)
    try:
        names = _decode(ds["cross_section_name"].values)
        section_name = f"ObservationCrossSection_Estuary_km{km}"
        if section_name in names:                       # same matching as the his-file script
            i = names.index(section_name)
        else:
            matches = [j for j, n in enumerate(names) if section_name.lower() in n.lower()]
            if len(matches) != 1:
                raise ValueError(f"Cross-section '{section_name}': {len(matches)} matches in {names}")
            i = matches[0]
        da = ds[HIS_Q_VAR].isel(cross_section=i)
        for d in list(da.dims):
            if d != "time":
                da = da.isel({d: 0})
        return np.asarray(ds.time.values).astype("datetime64[ns]"), np.asarray(da.values)
    finally:
        ds.close()


def panel_letter(ax, letter, background=False):
    """AGU panel label: lowercase letter in parentheses, top-left inside the axes."""
    bbox = dict(fc="white", ec="none", alpha=0.8, pad=1.0) if background else None
    ax.text(0.015, 0.95, f"({letter})", transform=ax.transAxes, ha="left", va="top", bbox=bbox)


def clean(ax):
    ax.grid(True, color="0.9", linewidth=0.5)


# %% --- LOAD AND COMPUTE ------------------------------------------------------

# --- detailed-hydro run (2 days on the t = end-of-run bed) ---
prof_const = detailed_profiles(DETAILED_RUN)       # constant scenario, on its final bed
prof_mean = detailed_profiles(MEAN_DETAILED_RUN)   # peaky scenario, flow = constant-scenario flow
prof_peak = detailed_profiles(PEAK_DETAILED_RUN)   # peaky scenario, peak flow
wl_ref = prof_const["wl_min_full"]                 # fixed reference stage for the wet mask

SCEN = {"constant": (DETAILED_RUN, prof_const, COLOR_CONST),
        f"{RPEAK} = 5, mean flow": (MEAN_DETAILED_RUN, prof_mean, COLOR_MEAN),
        f"{RPEAK} = 5, peak flow": (PEAK_DETAILED_RUN, prof_peak, COLOR_PEAK)}


def last_window(t, q, hours=LAST_HOURS):
    m = t >= t[-1] - np.timedelta64(hours, "h")
    return (t[m] - t[m][0]) / np.timedelta64(1, "h"), q[m]


q_series = {}   # (scenario, km) -> (hours, Q)
for sname, (rdir, _, _) in SCEN.items():
    for km in (20, 45):
        t, q = his_discharge(rdir / HIS_FILENAME, km)
        q_series[(sname, km)] = last_window(t, q)

# hydrograph of the variable scenario (one year) + constant level, at km45
t_h, q_h = his_discharge(HYDRO_MORPH_RUN / HIS_FILENAME, HYDROGRAPH_KM)
m_h = t_h <= t_h[0] + np.timedelta64(int(YEAR_DAYS_HYDROGRAPH * 86400e9), "ns")
days_h, q_h = (t_h[m_h] - t_h[0]) / np.timedelta64(1, "D"), q_h[m_h]
yr_h = days_h / YEAR_DAYS_HYDROGRAPH               # normalised time: fraction of the hydrological year
i_pk = int(np.nanargmax(q_h))
t_c, q_c = his_discharge(MORPH_RUN / HIS_FILENAME, HYDROGRAPH_KM)
q_const = float(np.nanmean(q_c))
i_mn = int(np.nanargmin(np.abs(q_h - q_const)))   # moment with the constant-scenario flow
print(f"Hydrograph km{HYDROGRAPH_KM}: peak {q_h[i_pk]:.0f} m3/s on day {days_h[i_pk]:.0f}; "
      f"constant run mean {q_const:.0f} m3/s")

# --- dynamic equilibrium of the constant scenario: net export (sea) = river supply per tidal cycle ---
eq = tidal_cycle_budget(MORPH_RUN / HIS_FILENAME, EQ_SEA_KM, EQ_RIVER_KM, MORFAC_FACTOR, CDRYB)
eq_yr, eq_supply, eq_export, eq_period = eq["morph_yr"], eq[EQ_RIVER_KM], eq[EQ_SEA_KM], eq["period_h"]
n_win = max(1, int(round(EQ_SMOOTH_YEARS * YEAR_DAYS / MORFAC_FACTOR * 24 / eq_period.mean())))
eq_supply_s, eq_export_s = running_mean(eq_supply, n_win), running_mean(eq_export, n_win)
T_EQ = equilibrium_time(eq_yr, eq_export_s / eq_supply_s, EQ_TOL)   # None: not reached by the end
late = eq_yr > eq_yr[-1] - 200
print(f"\nSediment budget km{EQ_SEA_KM}-{EQ_RIVER_KM}, constant run ({len(eq_yr)} tidal cycles of "
      f"{eq_period.mean():.2f} h, running mean {EQ_SMOOTH_YEARS:g} morph. yr = {n_win} cycles):")
print(f"  river supply {np.mean(eq_supply):.0f} m3/cycle, net export {np.mean(eq_export):.0f} m3/cycle; "
      f"export / supply: first 100 yr {np.mean(eq_export[eq_yr < 100]) / np.mean(eq_supply[eq_yr < 100]):.2f}, "
      f"last 200 yr {np.mean(eq_export[late]) / np.mean(eq_supply[late]):.2f}")
print(f"  net storage over the run: {np.sum(eq_supply - eq_export) * MORFAC_FACTOR:.3e} m3 of bed")
print(f"  dynamic equilibrium (|export - supply| < {EQ_TOL:.0%} of supply from then on): "
      + (f"reached after {T_EQ:.0f} morphological years" if T_EQ is not None else "NOT reached"))

# --- morphodynamic run ---
ds = load_ds(MORPH_RUN, [BED_VAR], cache_dir=CACHE_DIR, timed_out_dir=TIMED_OUT_DIR)
fdim = face_dim(ds, BED_VAR)
_, sec_idx = build_section_index(ds)
times = np.asarray(ds.time.values).astype("datetime64[ns]")
samp_i, samp_t = sample_indices(times)
print(f"Map times: {times[0]} -> {times[-1]} ({len(times)} outputs, "
      f"{(times[-1] - times[0]) / np.timedelta64(1, 'D') / YEAR_DAYS:.2f} hydrodynamic years = "
      f"{samp_t[-1]:.0f} morphological years at MorFac {MORFAC_FACTOR}); {len(samp_i)} sampled")
print("  first sampled morphological years: " + ", ".join(f"{d:.1f}" for d in samp_t[:12]))

# consistency check: the detailed run should sit on the final bed of the morphodynamic run
bed_end = ds[BED_VAR].isel(time=-1).transpose(fdim).values[sec_idx]
print(f"Bed check (final morph bed vs constant-scenario detailed-run bed): max |diff| = "
      f"{np.nanmax(np.abs(bed_end - prof_const['bed_full'])):.3f} m")

# peaky scenario (pm5_n1) morphodynamic run, same grid and same processing
ds_p = load_ds(HYDRO_MORPH_RUN, [BED_VAR], cache_dir=CACHE_DIR, timed_out_dir=TIMED_OUT_DIR)
times_p = np.asarray(ds_p.time.values).astype("datetime64[ns]")
samp_i_p, samp_t_p = sample_indices(times_p)
print(f"Peaky run map times: {times_p[0]} -> {times_p[-1]} ({len(times_p)} outputs, "
      f"{samp_t_p[-1]:.0f} morphological years)")

RUNS = {}   # scenario -> dict(ds, ref, idx, t)
for name, (d, i_, t_) in {"constant": (ds, samp_i, samp_t),
                          "peaky": (ds_p, samp_i_p, samp_t_p)}.items():
    RUNS[name] = dict(
        ds=d, idx=i_, t=t_,
        ref=build_centerline_reference(d, BED_VAR, 0, xmin=CENTERLINE_XMIN,
                                       xmax=CENTERLINE_XMAX, centerline_y=CENTERLINE_Y),
        k_mid=int(np.abs(t_ - 0.5 * t_[-1]).argmin()), k_end=len(i_) - 1,
    )


def raw_bed(run, k):
    """Actual bed level (not detrended); land above the threshold masked."""
    da = run["ds"][BED_VAR].isel(time=int(run["idx"][k]))
    raw = np.asarray(da.values)
    return da.copy(data=np.where(raw > DETREND_LAND_THRESHOLD, np.nan, raw))


def detrended_bed(run, k):
    da = run["ds"][BED_VAR].isel(time=int(run["idx"][k]))
    raw = np.asarray(da.values)
    return da.copy(data=np.where(raw > DETREND_LAND_THRESHOLD, np.nan, raw) - run["ref"])


# --- numbers for the text ---
print("\nHydrodynamics of the final morphology (detailed run):")
for sname, (_, pr, _) in SCEN.items():
    for km in (20, 45):
        q = q_series[(sname, km)][1]
        i = int(np.where(X_KM == km)[0][0])
        print(f"  [{sname}] km{km}: Q mean {q.mean():.0f} m3/s, tidal amplitude "
              f"+/-{(q.max() - q.min()) / 2:.0f} m3/s, tidal range "
              f"{pr['wl_max'][i] - pr['wl_min'][i]:.2f} m, thalweg depth "
              f"{pr['depth_min'][i]:.1f}-{pr['depth_max'][i]:.1f} m")
    rx, r = pr["range_x_km"], pr["range_x"]
    print(f"  [{sname}] width-averaged tidal range: km 20-21 {np.nanmean(r[rx < 21]):.2f} m, "
          f"max {np.nanmax(r):.2f} m at km {rx[np.nanargmax(r)]:.1f}, km 44-45 {np.nanmean(r[rx > 44]):.2f} m")

# %% --- FIGURE LAYOUT (inches) -----------------------------------------------

fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
map_w = fig_width_in - LEFT_MARGIN_IN - CBAR_PAD_IN - CBAR_W_IN - CBAR_LABEL_IN
map_h = map_w * (MAP_YLIM[1] - MAP_YLIM[0]) / (MAP_XLIM[1] - MAP_XLIM[0])   # equal-aspect maps
line_w = (fig_width_in - LEFT_MARGIN_IN - RIGHT_MARGIN_IN - 2 * LINE_WSPACE_IN) / 3

y_line = LEGEND_H_IN                                        # top of each block, from the figure top
y_maps = y_line + LINE_PANEL_H_IN + LINE_XLABEL_IN
y_range = y_maps + 3 * map_h + 2 * MAP_HSPACE_IN + RANGE_GAP_IN
fig_height_in = y_range + RANGE_PANEL_H_IN + MAP_XLABEL_IN

fig = plt.figure(figsize=(fig_width_in, fig_height_in))


def add_axes_in(x, y_top, w, h):
    """Axes at x [in] from the left edge and y_top [in] below the top edge of the figure."""
    return fig.add_axes([x / fig_width_in, 1 - (y_top + h) / fig_height_in,
                         w / fig_width_in, h / fig_height_in])


def cbar_axes_in(y_top, h):
    return add_axes_in(LEFT_MARGIN_IN + map_w + CBAR_PAD_IN, y_top, CBAR_W_IN, h)


km_formatter = FuncFormatter(lambda v, pos: f"{v / 1000:g}")


def format_map(ax, show_x, show_y=True):
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_title("")
    ax.set_xlabel("x [km]" if show_x else "")
    ax.set_ylabel("y [km]" if show_y else "")
    ax.xaxis.set_major_locator(MultipleLocator(5_000))
    ax.yaxis.set_major_locator(FixedLocator([6_000, 7_500, 9_000]))   # no clash between stacked maps
    ax.xaxis.set_major_formatter(km_formatter)
    ax.yaxis.set_major_formatter(km_formatter)
    ax.tick_params(labelbottom=show_x, labelleft=show_y)


def map_label(ax, text):
    ax.text(0.99, 0.95, text, transform=ax.transAxes, ha="right", va="top")


# %% --- (a-c) FORCING AND HYDRODYNAMICS OF THE FINAL MORPHOLOGY ---------------

axg, axf, axh = (add_axes_in(LEFT_MARGIN_IN + i * (line_w + LINE_WSPACE_IN), y_line, line_w, LINE_PANEL_H_IN)
                 for i in range(3))

# (a) discharge at km20 (solid) and km45 (dashed), last 24 h of each detailed run
for sname, (_, _, c) in SCEN.items():
    for km, ls in ((20, "-"), (45, "--")):
        h, q = q_series[(sname, km)]
        axg.plot(h, q, color=c, ls=ls, lw=1.0)
axg.axhline(0, color="k", lw=0.5)
axg.set_xlabel("time [h]")
axg.set_ylabel("discharge [m$^3$/s]")
axg.set_xlim(0, LAST_HOURS)
axg.set_ylim(top=1.4 * axg.get_ylim()[1])   # room for the panel label above the flood peak
axg.xaxis.set_major_locator(MultipleLocator(6))
panel_letter(axg, "a")

# (b) one-year hydrograph of the R_peak = 5 scenario, constant level, dots at mean and peak
#     (colours explained by the shared legend above the row)
axf.plot(yr_h, q_h, color=COLOR_PEAK, lw=1.0)
axf.axhline(q_const, color=COLOR_CONST, ls="--", lw=0.9)
axf.plot(yr_h[i_mn], q_h[i_mn], "o", color=COLOR_MEAN, mec="white", mew=0.5, ms=5, zorder=5)
axf.plot(yr_h[i_pk], q_h[i_pk], "o", color=COLOR_PEAK, mec="white", mew=0.5, ms=5, zorder=5)
axf.set_xlabel("normalised time [t / year]")
axf.set_ylabel("river discharge (km 45) [m$^3$/s]")
axf.set_xlim(0, 1)
panel_letter(axf, "b")

# (c) sediment budget of the constant run: net export at km20 / river supply at km45 per
#     tidal cycle (running mean); the equilibrium itself is discussed in the text
axh.plot(eq_yr, eq_export_s / eq_supply_s, color=COLOR_CONST, lw=1.0)
axh.set_xlabel("morphological time [yr]")
axh.set_ylabel("net export / river supply [–]")
axh.set_xlim(0, round(eq_yr[-1], -2))
axh.set_ylim(0, None)
axh.xaxis.set_major_locator(MultipleLocator(200))
panel_letter(axh, "c")

for ax in (axg, axf, axh):
    clean(ax)

x_mid = (LEFT_MARGIN_IN + 0.5 * (fig_width_in - LEFT_MARGIN_IN - RIGHT_MARGIN_IN)) / fig_width_in
fig.legend(handles=[Line2D([], [], color=c, lw=1.0, label=s) for s, (_, _, c) in SCEN.items()]
           + [Line2D([], [], color="k", ls="-", lw=1.0, label="sea (km 20)"),
              Line2D([], [], color="k", ls="--", lw=1.0, label="river (km 45)")],
           loc="upper center", bbox_to_anchor=(x_mid, 1 - 0.02 / fig_height_in), ncol=5,
           frameon=True, handlelength=1.8, handletextpad=0.5, columnspacing=1.2, borderaxespad=0)

# %% --- (d-f) BED LEVEL: initial, constant and R_peak = 5 scenario at the end of the run ---

layout = [("d", "constant", 0, "initial"),
          ("e", "constant", "k_end", "constant"),
          ("f", "peaky", "k_end", f"{RPEAK} = 5, {NPEAKS} = 1")]
axm = []
for j, (letter, scen, kk, name) in enumerate(layout):
    ax = add_axes_in(LEFT_MARGIN_IN, y_maps + j * (map_h + MAP_HSPACE_IN), map_w, map_h)
    run = RUNS[scen]
    k = kk if isinstance(kk, int) else run[kk]
    if letter == "d":          # initial bed: actual bed level, not detrended (identical for both runs)
        pc_a = raw_bed(run, 0).ugrid.plot(ax=ax, cmap=BED_CMAP, vmin=-BED_LIM_INITIAL,
                                          vmax=BED_LIM_INITIAL, add_colorbar=False, edgecolors="none")
        map_label(ax, name)
        # positive (seaward) flow direction: right to left
        ax.annotate("", xy=(0.88, 0.72), xytext=(0.99, 0.72), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.75, mutation_scale=7))
        ax.text(0.87, 0.72, "positive flow", transform=ax.transAxes, ha="right", va="center")
    else:                      # end beds: detrended w.r.t. the t=0 centreline profile
        pc = detrended_bed(run, k).ugrid.plot(ax=ax, cmap=BED_CMAP, vmin=-BED_LIM, vmax=BED_LIM,
                                              add_colorbar=False, edgecolors="none")
        map_label(ax, f"{name} – end ({run['t'][k]:.0f} yr)")
    format_map(ax, show_x=False)    # x axis labelled once, below the tidal-range panel (g)
    panel_letter(ax, letter)
    axm.append(ax)

cb_a = fig.colorbar(pc_a, cax=cbar_axes_in(y_maps, map_h))
cb_a.set_label("bed level [m]")
cb = fig.colorbar(pc, cax=cbar_axes_in(y_maps + map_h + MAP_HSPACE_IN, 2 * map_h + MAP_HSPACE_IN))
cb.set_label("detrended bed level [m]")

# %% --- (g) TIDAL RANGE along the estuary, final beds (same width / x axis as the maps) ---

axr = add_axes_in(LEFT_MARGIN_IN, y_range, map_w, RANGE_PANEL_H_IN)
for sname, (_, pr, c) in SCEN.items():
    axr.plot(pr["range_x_km"], pr["range_x"], color=c, lw=1.0)
axr.set_xlim(MAP_XLIM[0] / 1000, MAP_XLIM[1] / 1000)
axr.set_ylim(0, 1.25 * max(np.nanmax(pr["range_x"]) for _, pr, _ in SCEN.values()))   # room for the label
axr.xaxis.set_major_locator(MultipleLocator(5))
axr.yaxis.set_major_locator(MultipleLocator(1))
axr.set_xlabel("x [km]")
axr.set_ylabel("tidal range [m]")
clean(axr)
panel_letter(axr, "g")
fig.align_ylabels([axg] + axm + [axr])

# %% --- SAVE -------------------------------------------------------------------

print(f"\nFigure size: {fig_width_in / MM_TO_IN:.0f} x {fig_height_in / MM_TO_IN:.0f} mm")
save_name = f"AGU_morphology_overview_Q{DISCHARGE}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=DPI, bbox_inches="tight", transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: {save_path}")
ds.close()
ds_p.close()

# %%

