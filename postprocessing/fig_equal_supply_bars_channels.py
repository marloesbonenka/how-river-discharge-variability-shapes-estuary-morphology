"""Same river sediment supply, different bars: a few high peaks build higher bars than many
lower peaks, while the channels stay at the same level.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one row:
  (a) change in river sediment supply relative to the constant run [%] (cumulative bedload
      through km 45 over the whole run), R_peak x n_peaks, Q = 500 m3/s. The two runs of
      EQUAL_SUPPLY_RUNS (same supply) are outlined in their line colour.
  (b) bars: p95 bed level along the estuary for those two runs and the constant run.
  (c) channels: p5 bed level along the estuary, same runs.
  Grey band in (b, c): autogenic variability (+/-2 sigma of the noisy repeats of the constant
  run plus the constant run itself), as in fig_bedlevel_percentiles_pm_n.py.
  Colours: R_peak, Blues palette (light -> dark), as in the other figures.

Data: supply from endvalues_km45_bedload_Q500.csv of plot_his_upstream_cumulative_sediment_km44.py
(run that first); bed levels from the map caches (single snapshot at TARGET_DATE, MASK_MODE
population, see functions/F_bedlevel_percentiles.py).
"""
#%% --- Imports ---
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import cmocean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import (
    apply_plot_style,
    find_variability_model_folders,
    get_snapshot_matches_by_target_dates,
    get_variability_map,
    save_agu_figure,
)
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_bedlevel_percentiles import bedlevel_profiles

#%% --- 1. SETTINGS ---
DISCHARGE = 500
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
base_path = BASE_DIR / f"Q{DISCHARGE}"
SUPPLY_CSV = base_path / 'output_plots' / 'plots_his_sedimentsupply_km44' / f'endvalues_km45_bedload_Q{DISCHARGE}.csv'

EQUAL_SUPPLY_RUNS = [(5, 1), (3, 5)]   # (R_peak, n_peaks) with the same supply (+23 %)
TARGET_DATE = np.datetime64('2031-01-01')

MASK_MODE = 'union'            # 'frozen' | 'union' (see functions/F_bedlevel_percentiles.py)
CHANNEL_INIT_THRESHOLD = 2.2   # channel = bed below this [m]
X_BINS = np.arange(20000, 45000, 1000)   # 1 km bins, 20-44 km
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
y_range = (5000, 10000)
METRICS = ['p95', 'p5']
XTICK_STEP_KM = 5

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None

# Autogenic variability: noisy repeats of the constant scenario
NOISY_BASE_PATH = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
                       r"\Model_Output\Q500\0_Noise_Q500")
NOISY_SUBFOLDERS = ['1_Q500_noisy0.9095347', '1_Q500_noisy1_rst.9160657', '1_Q500_noisy2_rst.9160663']

RPEAK = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'
GREY_CONST = "#7f7f7f"
ENV_COLOR, ENV_ALPHA = '0.55', 0.25
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

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'equal_supply_bars_channels'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#%% --- 2. HELPERS ---
def snapshot_profiles(cache_dir, folder_name, run_paths):
    """{metric: value per x-bin} at TARGET_DATE, or None."""
    ds = load_or_update_map_cache_multi(
        cache_dir=cache_dir, folder_name=folder_name, run_paths=run_paths,
        var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        print(f"  No cached data for {folder_name}, skipping.")
        return None
    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        ds.close()
        return None
    _, ts_idx, actual_dt = matches[0]
    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
    bed_now = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
    ds.close()
    profs, _ = bedlevel_profiles(bed_now, init_bl, face_x, face_y, width_mask, X_BINS, METRICS,
                                 MASK_MODE, threshold=CHANNEL_INIT_THRESHOLD)
    print(f"  {folder_name}: snapshot {actual_dt}")
    return profs


#%% --- 3. LOAD ---
# Sediment supply, % change relative to the constant run
supply = pd.read_csv(SUPPLY_CSV)
v_const = supply.loc[(supply.peak_ratio == 1) & (supply.n_peaks == 0), 'final_value_kg'].iloc[0]
supply['change_pct'] = (supply.final_value_kg - v_const) / abs(v_const) * 100
rows_pm = sorted(supply.peak_ratio.astype(int).unique(), reverse=True)
cols_n = sorted(supply.n_peaks.astype(int).unique())
grid = np.full((len(rows_pm), len(cols_n)), np.nan)
for _, r in supply.iterrows():
    grid[rows_pm.index(int(r.peak_ratio)), cols_n.index(int(r.n_peaks))] = r.change_pct

# Bed-level profiles: the constant run and the equal-supply runs
wanted = {(1, 0)} | set(EQUAL_SUPPLY_RUNS)
profiles = {}
for folder in find_variability_model_folders(base_path=base_path, discharge=DISCHARGE,
                                             scenarios_to_process=None, analyze_noisy=False):
    m = re.search(r'_pm(\d+)_n(\d+)', folder.name)
    if not m or (int(m.group(1)), int(m.group(2))) not in wanted:
        continue
    run_paths = get_stitched_map_run_paths(
        base_path=base_path, folder_name=folder.name, timed_out_dir=base_path / 'timed-out',
        variability_map=get_variability_map(DISCHARGE), analyze_noisy=False,
    ) or [base_path / folder]
    profs = snapshot_profiles(base_path / 'cached_data', folder.name, run_paths)
    if profs is not None:
        profiles[(int(m.group(1)), int(m.group(2)))] = profs

refs = [profiles[(1, 0)]] if (1, 0) in profiles else []
for sub in NOISY_SUBFOLDERS:
    profs = snapshot_profiles(NOISY_BASE_PATH / 'cached_data', sub, [NOISY_BASE_PATH / sub])
    if profs is not None:
        refs.append(profs)
band = {}
if len(refs) >= 3:
    for metric in METRICS:
        stk = np.vstack([p[metric] for p in refs])
        mean, std = np.nanmean(stk, axis=0), np.nanstd(stk, axis=0)
        band[metric] = (mean - 2 * std, mean + 2 * std)

print("\nSupply change [%]:")
for pm, n in EQUAL_SUPPLY_RUNS:
    print(f"  R_peak {pm}, n_peaks {n}: {grid[rows_pm.index(pm), cols_n.index(n)]:+.1f}")


#%% --- 4. FIGURE ---
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
    gridspec_kw={'width_ratios': [1.15, 1, 1]}, constrained_layout=True,
)

# (a) supply heatmap
vmax = np.nanmax(grid)
im = ax_a.imshow(grid, aspect='auto', cmap=cmocean.cm.turbid, vmin=0, vmax=vmax)
for ri in range(len(rows_pm)):
    for ci in range(len(cols_n)):
        val = grid[ri, ci]
        if np.isfinite(val):
            ax_a.text(ci, ri, f"{val:+.0f}%", ha='center', va='center', fontsize=FONT_SIZE - 1.5,
                      color='black' if val < 0.6 * vmax else 'white')
for pm, n in EQUAL_SUPPLY_RUNS:
    ax_a.add_patch(mpatches.Rectangle((cols_n.index(n) - 0.5, rows_pm.index(pm) - 0.5), 1, 1,
                                      fill=False, edgecolor=PM_COLOR[pm], linewidth=2.0, zorder=5))
ax_a.set_xticks(range(len(cols_n)), [str(n) for n in cols_n])
ax_a.set_yticks(range(len(rows_pm)), [str(pm) for pm in rows_pm])
ax_a.set_xlabel(f'Peaks per year {NPEAKS}')
ax_a.set_ylabel(f'Peak ratio {RPEAK}')
ax_a.set_title('River sediment supply', loc='left')
ax_a.grid(False)
cbar = fig.colorbar(im, ax=ax_a, pad=0.02, fraction=0.06)
cbar.set_label('Change relative to constant [%]')

# (b) bars and (c) channels along the estuary
for ax, metric, title, ylabel in ((ax_b, 'p95', 'Bars', 'Bar level (p95 bed level) [m]'),
                                  (ax_c, 'p5', 'Channels', 'Channel level (p5 bed level) [m]')):
    if metric in band:
        ax.fill_between(X_KM, *band[metric], color=ENV_COLOR, alpha=ENV_ALPHA, linewidth=0, zorder=1)
    if (1, 0) in profiles:
        ax.plot(X_KM, profiles[(1, 0)][metric], color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                linestyle='--', zorder=2)
    for pm, n in EQUAL_SUPPLY_RUNS:
        if (pm, n) in profiles:
            ax.plot(X_KM, profiles[(pm, n)][metric], color=PM_COLOR[pm], linewidth=LINE_WIDTH, zorder=3)
    ax.set_xlim(X_KM[0] - 0.5, X_KM[-1] + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
    ax.set_xlabel('Along-estuary distance [km]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.grid(True, color='0.9', linewidth=0.5)

for ax, letter in zip((ax_a, ax_b, ax_c), 'abc'):
    ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, ha='right', va='bottom')

legend_handles = [mpatches.Patch(facecolor=ENV_COLOR, alpha=ENV_ALPHA, label='autogenic variability'),
                  mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                                label='constant')]
legend_handles += [mlines.Line2D([], [], color=PM_COLOR[pm], linewidth=LINE_WIDTH,
                                 label=f'{RPEAK} = {pm}, {NPEAKS} = {n}') for pm, n in EQUAL_SUPPLY_RUNS]
fig.legend(handles=legend_handles, loc='outside lower center', ncol=len(legend_handles), frameon=True)

save_name = "AGU_equal_supply_bars_channels_Q500_" + "_".join(f"pm{pm}n{n}" for pm, n in EQUAL_SUPPLY_RUNS)
save_agu_figure(fig, OUTPUT_DIR / save_name)
plt.show()

# %%
