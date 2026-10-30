"""Mean flow, ebb: variable discharge gives fewer, wider channels in the tidal-fluvial transition
zone, while the cumulative channel width stays similar.

Detailed-hydrodynamic runs at mean flow, Q = 500 m3/s: the constant run (pm1_n0) and the
n_peaks = 3 runs with R_peak = 2-5. Ebb = minimum water level over the final 12 hours of each run.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one row, along the estuary:
  (a) braiding index: number of separate channels per cross-section;
  (b) width of the widest single channel;
  (c) cumulative width of all channels.
  Lines are a centred running mean over SMOOTH_KM cross-sections (1 = raw values).
  Colours: constant run grey dashed, R_peak Blues palette (light -> dark), as in the other figures.

Channel definition as in plot_map_cross_sections_bedlev_wlev.py: cross-sections every km at
x = 20-45 km, y = 5-10 km, sampled at N_Y_SAMPLES points (nearest face centre). A sample is wet
when ebb water level - final bed level > WET_DEPTH_THRESHOLD_M; a channel is a run of consecutive
wet samples.
"""
#%% --- Imports ---
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style, save_agu_figure
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths

#%% --- 1. SETTINGS ---
DISCHARGE = 500
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
DHR_DIR = BASE_DIR / f"Q{DISCHARGE}" / "detailed-hydro-run"

# R_peak -> run folder (R_peak = 1: constant run, pm1_n0)
RUNS = {
    1: "dhr_01_Qr500_pm1_n0.9724783",
    2: "dhr_11_Qr500_pm2_n3_mean.10280081",
    3: "dhr_10_Qr500_pm3_n3_mean.10280082",
    4: "dhr_06_Qr500_pm4_n3_mean.10280084",
    5: "dhr_09_Qr500_pm5_n3_mean.10280083",
}
N_PEAKS = 3

BEDLEVEL_VAR = "mesh2d_mor_bl"
WATERLEVEL_VAR = "mesh2d_s1"
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None

X_KM = np.arange(20.0, 46.0, 1.0)   # cross-section locations
Y_RANGE_M = (5000.0, 10000.0)
N_Y_SAMPLES = 500
WET_DEPTH_THRESHOLD_M = 0.1         # wet = water level - bed level above this [m]
EBB_WINDOW_H = 12                   # ebb water level = minimum over the final hours
SMOOTH_KM = 3                       # running-mean window [cross-sections]; 1 = no smoothing
XTICK_STEP_KM = 5

RPEAK = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'
GREY_CONST = "#7f7f7f"
LINE_WIDTH = 1.4
LINE_WIDTH_CONST = 0.9
ALL_PM = [2, 3, 4, 5]          # colour scale as in the other figures
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / (len(ALL_PM) - 1)) for i, pm in enumerate(ALL_PM)}

# --- AGU figure sizing ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 62
FONT_SIZE = 8

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'mean_flow_channels_ebb'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#%% --- 2. HELPERS ---
def ebb_cross_sections(folder_name):
    """Final bed level and ebb (minimum) water level on the cross-sections, each (n_x, n_y)."""
    run_paths = get_stitched_map_run_paths(
        base_path=DHR_DIR, folder_name=folder_name, timed_out_dir=None,
        variability_map=None, analyze_noisy=False,
    ) or [DHR_DIR / folder_name]
    ds = load_or_update_map_cache_multi(
        cache_dir=DHR_DIR / 'cached_data', folder_name=folder_name, run_paths=run_paths,
        var_names=[WATERLEVEL_VAR, BEDLEVEL_VAR], bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        raise FileNotFoundError(f"No cached map data for {folder_name}.")
    try:
        times = np.asarray(ds['time'].values).astype('datetime64[ns]')
        ebb_idx = np.where(times >= times[-1] - np.timedelta64(EBB_WINDOW_H, 'h'))[0]
        bed = ds[BEDLEVEL_VAR].isel(time=-1).values
        wl_ebb = ds[WATERLEVEL_VAR].isel(time=ebb_idx).min(dim='time', skipna=True).values
        face_x, face_y = _get_face_coords(ds)
    finally:
        ds.close()
    print(f"  {folder_name}: ebb window {times[ebb_idx[0]]} to {times[-1]} ({ebb_idx.size} maps)")

    y = np.linspace(*Y_RANGE_M, N_Y_SAMPLES)
    xx, yy = np.meshgrid(X_KM * 1000.0, y, indexing='ij')
    _, idx = cKDTree(np.column_stack((face_x, face_y))).query(np.column_stack((xx.ravel(), yy.ravel())))
    idx = idx.reshape(xx.shape)
    return y, np.asarray(bed, dtype=float)[idx], np.asarray(wl_ebb, dtype=float)[idx]


def channel_widths(y, bed, wl):
    """Widths [m] of the channels (runs of consecutive wet samples) in one cross-section."""
    depth = wl - bed
    wet = np.isfinite(depth) & (depth > WET_DEPTH_THRESHOLD_M)
    changes = np.diff(np.concatenate(([0], wet.astype(int), [0])))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    return (ends - starts) * float(np.mean(np.diff(y)))


def smooth(values):
    """Centred running mean over SMOOTH_KM cross-sections (shorter window at the ends)."""
    return pd.Series(values).rolling(SMOOTH_KM, center=True, min_periods=1).mean().to_numpy()


#%% --- 3. LOAD ---
metrics = {}   # R_peak -> {'count', 'widest', 'cumulative'}: value per cross-section
for pm, folder_name in RUNS.items():
    y, bed, wl = ebb_cross_sections(folder_name)
    widths = [channel_widths(y, bed[i], wl[i]) for i in range(X_KM.size)]
    metrics[pm] = {
        'count': np.array([w.size for w in widths], dtype=float),
        'widest': np.array([w.max() if w.size else 0.0 for w in widths]),
        'cumulative': np.array([w.sum() for w in widths]),
    }


#%% --- 4. FIGURE ---
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN), constrained_layout=True,
)
panels = ((ax_a, 'count', 1.0, 'Braiding index', 'Number of channels [-]'),
          (ax_b, 'widest', 1e-3, 'Widest single channel', 'Width [km]'),
          (ax_c, 'cumulative', 1e-3, 'Cumulative channel width', 'Width [km]'))
for ax, key, scale, title, ylabel in panels:
    ax.plot(X_KM, smooth(metrics[1][key]) * scale, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
            linestyle='--', zorder=2)
    for pm in ALL_PM:
        ax.plot(X_KM, smooth(metrics[pm][key]) * scale, color=PM_COLOR[pm], linewidth=LINE_WIDTH, zorder=3)
    ax.set_xlim(X_KM[0], X_KM[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
    ax.set_xlabel('Along-estuary distance [km]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.grid(True, color='0.9', linewidth=0.5)

for ax, letter in zip((ax_a, ax_b, ax_c), 'abc'):
    ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, ha='right', va='bottom')

legend_handles = [mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                                label='constant')]
legend_handles += [mlines.Line2D([], [], color=PM_COLOR[pm], linewidth=LINE_WIDTH,
                                 label=f'{RPEAK} = {pm}, {NPEAKS} = {N_PEAKS}') for pm in ALL_PM]
fig.legend(handles=legend_handles, loc='outside lower center', ncol=len(legend_handles), frameon=True)

save_agu_figure(fig, OUTPUT_DIR / f"AGU_channels_mean_flow_ebb_Q{DISCHARGE}_n{N_PEAKS}")
plt.show()

# %%
