# %% Fig. S?: autogenic variability - smooth initial bed and the three noisy repeats
"""
Supplementary figure to the autogenic variability envelope (grey +/-2 sigma band in
fig_bedlevel_percentiles_pm_n.py and fig_bed_slope_along_estuary_allQ.py).

Four runs, all at constant river discharge (Q = 500 m3/s) with identical forcing and
settings; they differ only in the initial bed:
    smooth  : 01_Qr500_pm1_n0 (the constant scenario of the main text)
    seed 0-2: 1_Q500_noisy{0,1,2}, smooth bed + spatially uncorrelated, uniformly
              distributed noise of +/-0.20 m on the initial channel samples (km 20-45)

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one row per run:
left  column (a-d): initial bed (t = 0) as the model sees it (samples interpolated
                           onto the grid); (a) bed level of the smooth run, (b-d) the
                           perturbation = noisy minus smooth initial bed
right column (e-h): detrended bed level at the envelope snapshot (TARGET_DATE,
                           ~600 morphological years), same colour scale as Fig. 3e

Also prints the numbers for the SI text: the noise on the grid, the bed-level
differences between the four runs at the snapshot, and the half-width (2 sigma) of the
autogenic envelope of the p95 / p5 bed level (same population as the main figure).
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cmocean
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from functions.F_general import (
    apply_plot_style,
    create_terrain_colormap,
    get_snapshot_matches_by_target_dates,
    setup_variability_run_context,
)
from functions.F_bedlevel_percentiles import bedlevel_profiles
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_map_cache import _get_face_coords, cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_morphological_activity import build_centerline_reference

warnings.filterwarnings("ignore", category=UserWarning, module="xugrid")

# %% --- CONFIGURATION ---------------------------------------------------------

DISCHARGE = 500
# smooth run = constant scenario of the main text (Gaussian domain)
BASE_DIRECTORY = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
run_context = setup_variability_run_context(
    base_directory=BASE_DIRECTORY, discharge=DISCHARGE, scenarios_to_process=None,
    analyze_noisy=False, MORFAC=False,
)
SMOOTH_RUN = BASE_DIRECTORY / "Model_Output" / "Q500" / "01_Qr500_pm1_n0.9600302"
SMOOTH_CACHE = run_context["cache_dir"]
SMOOTH_TIMED_OUT = run_context["timed_out_dir"]

# noisy repeats (same set as the envelope in fig_bedlevel_percentiles_pm_n.py)
NOISY_BASE_PATH = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
                       r"\Model_Output\Q500\0_Noise_Q500")
NOISY_SUBFOLDERS = {
    0: "1_Q500_noisy0.9095347",
    1: "1_Q500_noisy1_rst.9160657",
    2: "1_Q500_noisy2_rst.9160663",
}
NOISE_AMPLITUDE = 0.20      # [m] half-range of the uniform noise on the samples (noisybathy_seed*.xyz)

# envelope snapshot, as in fig_bedlevel_percentiles_pm_n.py (nearest map time per run)
TARGET_DATE = np.datetime64("2031-01-01")
MORFAC_FACTOR = 100
YEAR_DAYS = 365.25

BED_VAR = "mesh2d_mor_bl"
CACHE_BBOX = [1, 1, 45000, 15000]

# detrending as in fig_morphology_overview.py; the reference comes from the SMOOTH t = 0 bed
# for all four runs, so the initial noise does not leak into the reference
DETREND_LAND_THRESHOLD = 6.0
CENTERLINE_XMIN, CENTERLINE_XMAX, CENTERLINE_Y = 20000, 45000, 7500
BED_CMAP = create_terrain_colormap()
BED_LIM = 5                 # detrended bed level [m], as Fig. 3e
INIT_LIM = 2.5              # initial bed level of the smooth run [m]
NOISE_CMAP = cmocean.cm.balance

# envelope numbers (same population as fig_bedlevel_percentiles_pm_n.py)
METRICS = ["p5", "mean", "p95"]
METRIC_YLABEL = {"p5": "p5 bed level [m]", "mean": "mean bed level [m]", "p95": "p95 bed level [m]"}
MASK_MODE = "union"
CHANNEL_INIT_THRESHOLD = 2.2
X_BINS = np.arange(20000, 45000, 1000)
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
Y_RANGE_M = (5000, 10000)
ENVELOPE_DDOF = 0           # 0 = population std, as np.nanstd in the main figures; 1 = sample std
COLOR_SMOOTH = "k"          # smooth run (black: the grey envelope band would hide a grey line)
COLOR_SEEDS = ["#1b9e77", "#d95f02", "#7570b3"]
ENV_COLOR, ENV_ALPHA = "0.55", 0.25   # envelope band, as in fig_bedlevel_percentiles_pm_n.py

MAP_XLIM = (20_000, 45_000)
MAP_YLIM = (5_000, 10_000)

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FONT_SIZE = 8

LEFT_MARGIN_IN = 0.45       # y tick labels + y-label
COL_GAP_IN = 0.16           # left colorbar label -> right map (right maps share the y axis)
CBAR_PAD_IN = 0.06
CBAR_W_IN = 0.08
CBAR_LABEL_IN = 0.42
TOP_MARGIN_IN = 0.20        # column titles
MAP_HSPACE_IN = 0.06
MAP_XLABEL_IN = 0.50        # x tick labels + x-label below the maps
LINE_PANEL_H_IN = 1.30      # p5 / mean / p95 profiles
LINE_WSPACE_IN = 0.45       # gap between line panels (each has its own y-label)
RIGHT_MARGIN_IN = 0.08
LINE_XLABEL_IN = 0.36       # x tick labels + x-label below the line panels
LEGEND_H_IN = 0.25          # legend below the line panels

SMOOTH_LABEL = "smooth initial bed"
OUTPUT_DIR = BASE_DIRECTORY / "Model_Output" / "output_plots_combined" / "noisy_bathymetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

apply_plot_style("AGU", font_size=FONT_SIZE, title_delta=0)

# %% --- FUNCTIONS -------------------------------------------------------------


def load_bed(cache_dir, run_dir, timed_out_dir=None):
    """Bed level of a run from the map cache (stitched over timed-out parts), time sorted."""
    name = run_dir.name
    run_paths = None
    if timed_out_dir is not None:
        run_paths = get_stitched_map_run_paths(base_path=run_dir.parent, folder_name=name,
                                               timed_out_dir=timed_out_dir, analyze_noisy=False)
    ds = load_or_update_map_cache_multi(
        cache_dir=cache_dir, folder_name=name, run_paths=run_paths or [run_dir],
        var_names=[BED_VAR], bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, None),
    )
    if ds is None:
        raise FileNotFoundError(f"No cached map data for {name}")
    t = np.asarray(ds.time.values)
    if np.any(np.diff(t) <= np.timedelta64(0)):
        ds = ds.isel(time=np.unique(t, return_index=True)[1])
    return ds


def snapshot(ds):
    """(time index, morphological years since t0) of the map time nearest to TARGET_DATE."""
    t = np.asarray(ds.time.values).astype("datetime64[ns]")
    _, k, t_k = get_snapshot_matches_by_target_dates(t, [TARGET_DATE])[0]
    return k, (t_k - t[0]) / np.timedelta64(1, "D") * MORFAC_FACTOR / YEAR_DAYS


def face_map(ds_from, ds_to):
    """Index array that maps faces of ds_from onto the faces of ds_to (identity if same grid)."""
    a = np.column_stack(_get_face_coords(ds_from))
    b = np.column_stack(_get_face_coords(ds_to))
    if a.shape == b.shape and np.allclose(a, b):
        return None
    d, idx = cKDTree(a).query(b)
    print(f"  [grid] different face sets, nearest-face mapping (max distance {d.max():.1f} m)")
    return idx


def remap(values, idx):
    return values if idx is None else values[idx]


def panel_letter(ax, letter):
    ax.text(0.015, 0.95, f"({letter})", transform=ax.transAxes, ha="left", va="top")


def map_label(ax, text):
    ax.text(0.99, 0.95, text, transform=ax.transAxes, ha="right", va="top")


# %% --- LOAD ------------------------------------------------------------------

runs = {}   # label -> dict(ds, k, yr, idx (faces -> smooth grid))

print(f"Loading smooth run: {SMOOTH_RUN.name}")
ds_s = load_bed(SMOOTH_CACHE, SMOOTH_RUN, SMOOTH_TIMED_OUT)
k_s, yr_s = snapshot(ds_s)
runs[SMOOTH_LABEL] = dict(ds=ds_s, k=k_s, yr=yr_s, idx=None)

for seed, sub in NOISY_SUBFOLDERS.items():
    print(f"Loading noisy run: {sub}")
    ds_n = load_bed(NOISY_BASE_PATH / "cached_data", NOISY_BASE_PATH / sub)
    k_n, yr_n = snapshot(ds_n)
    runs[f"seed {seed}"] = dict(ds=ds_n, k=k_n, yr=yr_n, idx=face_map(ds_n, ds_s))

for lab, r in runs.items():
    print(f"  {lab}: snapshot {np.asarray(r['ds'].time.values)[r['k']]} = {r['yr']:.0f} morphological years")

ref = build_centerline_reference(ds_s, BED_VAR, 0, xmin=CENTERLINE_XMIN, xmax=CENTERLINE_XMAX,
                                 centerline_y=CENTERLINE_Y)
bed0_smooth = np.asarray(ds_s[BED_VAR].isel(time=0).values)
face_x, face_y = _get_face_coords(ds_s)
template = ds_s[BED_VAR].isel(time=0)   # UgridDataArray on the smooth grid, for plotting


def on_smooth_grid(r, time_idx):
    return remap(np.asarray(r["ds"][BED_VAR].isel(time=time_idx).values), r["idx"])


def detrended(values):
    return np.where(values > DETREND_LAND_THRESHOLD, np.nan, values) - ref


fields = {}   # label -> (left-column field, right-column field) on the smooth grid
for lab, r in runs.items():
    b0 = on_smooth_grid(r, 0)
    # smooth run: its bed level (land masked); noisy runs: perturbation relative to the smooth bed
    left = np.where(bed0_smooth > DETREND_LAND_THRESHOLD, np.nan,
                    b0 if lab == SMOOTH_LABEL else b0 - bed0_smooth)
    fields[lab] = (left, detrended(on_smooth_grid(r, r["k"])))

# %% --- NUMBERS FOR THE TEXT -------------------------------------------------

chan = (face_x >= MAP_XLIM[0]) & (bed0_smooth < CHANNEL_INIT_THRESHOLD) & (face_y >= Y_RANGE_M[0]) & (face_y <= Y_RANGE_M[1])
print(f"\nInitial noise on the grid ({chan.sum()} channel faces, km 20-45):")
for lab, (dz, _) in list(fields.items())[1:]:
    v = dz[chan]
    print(f"  {lab}: mean {np.nanmean(v):+.4f} m, std {np.nanstd(v):.3f} m, "
          f"range {np.nanmin(v):+.3f} .. {np.nanmax(v):+.3f} m, outside channel max |dz| "
          f"{np.nanmax(np.abs(dz[~chan & (face_x >= MAP_XLIM[0])])):.3f} m")

# bed-level differences at the snapshot within the estuary (y 5-10 km, x 20-45 km, below land threshold)
box = (face_x >= MAP_XLIM[0]) & (face_x <= MAP_XLIM[1]) & (face_y >= Y_RANGE_M[0]) & (face_y <= Y_RANGE_M[1])
finals = np.vstack([f[1] for f in fields.values()])
ok = box & np.all(np.isfinite(finals), axis=0)
spread = np.nanstd(finals[:, ok], axis=0)
change_s = (on_smooth_grid(runs[SMOOTH_LABEL], k_s) - bed0_smooth)[ok]
print(f"\nBed at the snapshot ({ok.sum()} estuary faces): std over the 4 runs per face: "
      f"median {np.median(spread):.2f} m, 90th pct {np.percentile(spread, 90):.2f} m; "
      f"for comparison RMS bed change of the smooth run since t0: {np.sqrt(np.mean(change_s ** 2)):.2f} m")
labs = list(fields)
for i in range(1, len(labs)):
    d = finals[i, ok] - finals[0, ok]
    print(f"  RMS difference {labs[i]} - smooth: {np.sqrt(np.mean(d ** 2)):.2f} m")

# autogenic envelope of p95 / p5 (as in fig_bedlevel_percentiles_pm_n.py: noisy + smooth, mean +/- 2 std)
wmask = (face_y >= Y_RANGE_M[0]) & (face_y <= Y_RANGE_M[1])
profs = {m: [] for m in METRICS}
for lab, r in runs.items():
    fx, fy = _get_face_coords(r["ds"])
    wm = (fy >= Y_RANGE_M[0]) & (fy <= Y_RANGE_M[1])
    p, _ = bedlevel_profiles(np.asarray(r["ds"][BED_VAR].isel(time=r["k"]).values),
                             np.asarray(r["ds"][BED_VAR].isel(time=0).values),
                             fx, fy, wm, X_BINS, METRICS, MASK_MODE, threshold=CHANNEL_INIT_THRESHOLD)
    for m in METRICS:
        profs[m].append(p[m])
envelope = {}   # metric -> (mean, std) per x-bin over the four runs
print(f"\nAutogenic envelope (n = {len(runs)} runs, ddof = {ENVELOPE_DDOF}):")
for m in METRICS:
    stk = np.vstack(profs[m])
    sd = np.nanstd(stk, axis=0, ddof=ENVELOPE_DDOF)
    envelope[m] = (np.nanmean(stk, axis=0), sd)
    print(f"  {m}: 2 sigma half-width median {np.nanmedian(2 * sd):.2f} m, "
          f"range {np.nanmin(2 * sd):.2f}-{np.nanmax(2 * sd):.2f} m over the 1-km bins; "
          f"(sample std, ddof=1: median 2 sigma {np.nanmedian(2 * np.nanstd(stk, axis=0, ddof=1)):.2f} m)")

# %% --- FIGURE ----------------------------------------------------------------

fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
col_extra = CBAR_PAD_IN + CBAR_W_IN + CBAR_LABEL_IN
map_w = (fig_width_in - LEFT_MARGIN_IN - COL_GAP_IN - 2 * col_extra) / 2
map_h = map_w * (MAP_YLIM[1] - MAP_YLIM[0]) / (MAP_XLIM[1] - MAP_XLIM[0])
n_rows = len(fields)
y_line = TOP_MARGIN_IN + n_rows * map_h + (n_rows - 1) * MAP_HSPACE_IN + MAP_XLABEL_IN
line_w = (fig_width_in - LEFT_MARGIN_IN - RIGHT_MARGIN_IN - (len(METRICS) - 1) * LINE_WSPACE_IN) / len(METRICS)
fig_height_in = y_line + LINE_PANEL_H_IN + LINE_XLABEL_IN + LEGEND_H_IN
fig = plt.figure(figsize=(fig_width_in, fig_height_in))


def add_axes_in(x, y_top, w, h):
    return fig.add_axes([x / fig_width_in, 1 - (y_top + h) / fig_height_in,
                         w / fig_width_in, h / fig_height_in])


x_left = LEFT_MARGIN_IN
x_right = LEFT_MARGIN_IN + map_w + col_extra + COL_GAP_IN
row_top = [TOP_MARGIN_IN + i * (map_h + MAP_HSPACE_IN) for i in range(n_rows)]
km_formatter = FuncFormatter(lambda v, pos: f"{v / 1000:g}")


def format_map(ax, show_x, show_y):
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_title("")
    ax.set_xlabel("x [km]" if show_x else "")
    ax.set_ylabel("y [km]" if show_y else "")
    ax.xaxis.set_major_locator(MultipleLocator(5_000))
    ax.yaxis.set_major_locator(FixedLocator([6_000, 7_500, 9_000]))
    ax.xaxis.set_major_formatter(km_formatter)
    ax.yaxis.set_major_formatter(km_formatter)
    ax.tick_params(labelbottom=show_x, labelleft=show_y)


LEFT_LETTERS, RIGHT_LETTERS = "abcd", "efgh"   # left column a-d, right column e-h
pc_init = pc_noise = pc_bed = None
for i, (lab, (left, right)) in enumerate(fields.items()):
    last = i == n_rows - 1
    axl = add_axes_in(x_left, row_top[i], map_w, map_h)
    if i == 0:
        pc_init = template.copy(data=left).ugrid.plot(ax=axl, cmap=BED_CMAP, vmin=-INIT_LIM, vmax=INIT_LIM,
                                                      add_colorbar=False, edgecolors="none")
    else:
        pc_noise = template.copy(data=left).ugrid.plot(ax=axl, cmap=NOISE_CMAP, vmin=-NOISE_AMPLITUDE,
                                                       vmax=NOISE_AMPLITUDE, add_colorbar=False,
                                                       edgecolors="none", rasterized=True)
    map_label(axl, lab)
    format_map(axl, show_x=last, show_y=True)
    panel_letter(axl, LEFT_LETTERS[i])

    axr = add_axes_in(x_right, row_top[i], map_w, map_h)
    pc_bed = template.copy(data=right).ugrid.plot(ax=axr, cmap=BED_CMAP, vmin=-BED_LIM, vmax=BED_LIM,
                                                  add_colorbar=False, edgecolors="none")
    # map output every 18 days: the noisy runs' nearest time is 2031-01-06 (601 yr); label rounded
    # map_label(axr, f"after {round(runs[lab]['yr'], -1):.0f} years")
    format_map(axr, show_x=last, show_y=False)
    panel_letter(axr, RIGHT_LETTERS[i])
    if i == 0:
        axl.set_title("initial bed (t = 0)", loc="center")
        axr.set_title("detrended bed level after 600 years", loc="center")

# colorbars: (a) initial bed; (b-d) noise; (e-h) detrended bed
cb_x_left = x_left + map_w + CBAR_PAD_IN
cb_x_right = x_right + map_w + CBAR_PAD_IN
cb = fig.colorbar(pc_init, cax=add_axes_in(cb_x_left, row_top[0], CBAR_W_IN, map_h))
cb.set_label("bed level [m]")
cb.set_ticks([-2, 0, 2])
cb = fig.colorbar(pc_noise, cax=add_axes_in(cb_x_left, row_top[1], CBAR_W_IN, row_top[-1] + map_h - row_top[1]))
cb.set_label("perturbation [m]")
cb.set_ticks([-NOISE_AMPLITUDE, -NOISE_AMPLITUDE / 2, 0, NOISE_AMPLITUDE / 2, NOISE_AMPLITUDE])
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
cb = fig.colorbar(pc_bed, cax=add_axes_in(cb_x_right, row_top[0], CBAR_W_IN, row_top[-1] + map_h - row_top[0]))
cb.set_label("detrended bed level [m]")

# %% --- (i-k) p5 / mean / p95 along the estuary: the four runs and the +/-2 sigma envelope ---

run_colors = [COLOR_SMOOTH] + COLOR_SEEDS
for j, m in enumerate(METRICS):
    ax = add_axes_in(LEFT_MARGIN_IN + j * (line_w + LINE_WSPACE_IN), y_line, line_w, LINE_PANEL_H_IN)
    mean, sd = envelope[m]
    ax.fill_between(X_KM, mean - 2 * sd, mean + 2 * sd, color=ENV_COLOR, alpha=ENV_ALPHA, lw=0)
    for (lab, c), y in zip(zip(runs, run_colors), profs[m]):
        ax.plot(X_KM, y, color=c, lw=0.8)
    ax.plot(X_KM, mean, color="k", ls=":", lw=1.2)
    ax.set_xlim(MAP_XLIM[0] / 1000, MAP_XLIM[1] / 1000)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("x [km]")
    ax.set_ylabel(METRIC_YLABEL[m])
    ax.grid(True, color="0.9", linewidth=0.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))   # room for the panel letter
    panel_letter(ax, "ijk"[j])

handles =([Line2D([], [], color=c, lw=0.8, label=lab) for lab, c in zip(runs, run_colors)]
           + [Line2D([], [], color="k", ls=":", lw=1.2, label="mean of the 4 runs"),
              Patch(fc=ENV_COLOR, alpha=ENV_ALPHA, lw=0, label=r"mean $\pm$ 2$\sigma$")])
x_mid = (LEFT_MARGIN_IN + 0.5 * (fig_width_in - LEFT_MARGIN_IN - RIGHT_MARGIN_IN)) / fig_width_in
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x_mid, 0.0), ncol=len(handles),
           frameon=False, handlelength=1.5, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)

# %% --- SAVE -------------------------------------------------------------------

print(f"\nFigure size: {fig_width_in / MM_TO_IN:.0f} x {fig_height_in / MM_TO_IN:.0f} mm")
save_name = f"AGU_SI_noisy_bathymetry_Q{DISCHARGE}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=DPI, bbox_inches="tight", transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches="tight", transparent=True, dpi=DPI)
plt.show()
print(f"Saved: {save_path}")
for r in runs.values():
    r["ds"].close()

# %%
