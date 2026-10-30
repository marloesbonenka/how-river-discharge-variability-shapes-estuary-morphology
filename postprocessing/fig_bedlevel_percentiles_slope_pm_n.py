"""Bed-level profile slope vs. R_peak, one panel per n_peaks value that has more than one
R_peak run (single snapshot), for Q = 250 / 500 / 1000 m3/s. One figure per metric in
FIGURE_SPECS: p95 (bars) and mean bed level.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one row:
  - Columns (left -> right): one per n_peaks value with multiple R_peak runs at Q = 500.
    Q = 250 and Q = 1000 have the n_peaks = 3 series only, so they appear in that panel.
  - Marker colour: R_peak value, Blues palette (light -> dark); marker shape: discharge.
  - Grey marker at R_peak = 1 + dashed line: the constant pm1_n0 run per discharge.

Slope: least-squares line through the profile (per 1 km x-bin) over FIT_RANGE_KM. A uniform
rise (e.g. higher shoals everywhere) leaves the slope unchanged, a landward-increasing rise
steepens it. Bed-level population per x-bin: MASK_MODE (default 'union': faces below
CHANNEL_INIT_THRESHOLD at t=0 OR at the snapshot); see functions/F_bedlevel_percentiles.py.
The grey band is the +/-2 sigma autogenic variability of the noisy repeats of the constant run
(plus the constant run itself), at Q = 500 only.
"""
#%% --- Imports ---
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import (
    _parse_pm_n,
    _scenario_label,
    apply_plot_style,
    save_agu_figure,
    get_variability_map,
    find_variability_model_folders,
    get_snapshot_matches_by_target_dates,
)
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_bedlevel_percentiles import bedlevel_profiles

#%% --- 1. SETTINGS ---
DISCHARGES = [250, 500, 1000]   # tide- to river-dominated
REF_Q = 500                     # the only discharge with all n_peaks series
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

# nearest available timestep per run is used; optional override from the command line, e.g.
# `python fig_bedlevel_percentiles_slope_pm_n.py 2030-01-01` (keep 1 Jan: same hydrograph phase in all runs)
TARGET_DATE = np.datetime64(sys.argv[1] if len(sys.argv) > 1 else '2031-01-01')

# One figure per entry: (metric, y-label)
FIGURE_SPECS = [
    ('p95', 'p95 profile slope [m km$^{-1}$]'),
    ('mean', 'Mean profile slope [m km$^{-1}$]'),
]
METRICS = [metric for metric, _ in FIGURE_SPECS]
MASK_MODE = 'union'            # 'frozen' | 'union' (see functions/F_bedlevel_percentiles.py)
CHANNEL_INIT_THRESHOLD = 2.2   # channel = bed below this [m]
X_BINS = np.arange(20000, 45000, 1000)   # 1 km bins, 20-44 km
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
y_range = (5000, 10000)
FIT_RANGE_KM = (20, 44)        # x-range of the linear fit that defines the profile slope [km]

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None

# Autogenic variability: noisy repeats of the constant scenario, Q = 500 only
NOISY_BASE_PATH = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
                       r"\Model_Output\Q500\0_Noise_Q500")
NOISY_SUBFOLDERS = ['1_Q500_noisy0.9095347', '1_Q500_noisy1_rst.9160657', '1_Q500_noisy2_rst.9160663']

SCENARIO_LABELS = {
    '1':  'pm1_n0 (constant)',
    '2':  'pm2_n1',
    '3':  'pm3_n5',
    '4':  'pm3_n1',
    '5':  'pm5_n1',
    '6':  'pm4_n3',
    '7':  'pm3_n4',
    '8':  'pm2_n6',
    '9':  'pm5_n3',
    '10': 'pm3_n3',
    '11': 'pm2_n3',
    '12': 'pm5_n4',
    '13': 'pm4_n4',
    '14': 'pm2_n4',
}

RPEAK = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'
GREY_CONST = "#7f7f7f"
ENV_COLOR, ENV_ALPHA = '0.55', 0.25
LINE_WIDTH_CONST = 0.9
DISCHARGE_MARKERS = {250: '^', 500: 'o', 1000: 's'}
MARKER_SIZE = 4
TREND_COLOR = '0.6'      # thin line connecting the markers of one discharge

# --- AGU figure sizing ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 70
FONT_SIZE = 8

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'bedlevel_percentiles_slope_pm_n'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#%% --- 2. HELPERS ---
def snapshot_profile(cache_dir, folder_name, run_paths):
    """{metric: value per x-bin} at TARGET_DATE on the MASK_MODE population, or None."""
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
        print(f"  No timesteps found for {folder_name}, skipping.")
        return None
    _, ts_idx, actual_dt = matches[0]

    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
    bed_now = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
    ds.close()

    profs, _ = bedlevel_profiles(bed_now, init_bl, face_x, face_y, width_mask, X_BINS, METRICS,
                                 MASK_MODE, threshold=CHANNEL_INIT_THRESHOLD)
    print(f"  Snapshot {actual_dt}")
    return profs


def profile_slope(y):
    """Least-squares slope [m/km] of a profile over FIT_RANGE_KM (NaN bins skipped)."""
    ok = (X_KM >= FIT_RANGE_KM[0]) & (X_KM <= FIT_RANGE_KM[1]) & np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return np.nan
    x_dev = X_KM[ok] - X_KM[ok].mean()
    return np.sum(x_dev * (y[ok] - y[ok].mean())) / np.sum(x_dev ** 2)


def _fmt(v):
    return str(int(v)) if v == int(v) else str(v)


#%% --- 3. LOAD MODEL DATA + SLOPES (single snapshot, all runs, all discharges) ---
slopes = {m: {} for m in METRICS}   # {metric: {(Q, pm, n): slope [m/km]}}
y_const = {}                        # {Q: {metric: profile} of the constant run}
for Q in DISCHARGES:
    base_path = BASE_DIR / f"Q{Q}"
    model_folders = find_variability_model_folders(
        base_path=base_path, discharge=Q, scenarios_to_process=None, analyze_noisy=False,
    )
    for folder in model_folders:
        pm, n = _parse_pm_n(_scenario_label(folder.name, SCENARIO_LABELS))
        if pm is None:
            continue
        print(f"Processing Q{Q}: {folder.name}")
        run_paths = get_stitched_map_run_paths(
            base_path=base_path, folder_name=folder.name, timed_out_dir=base_path / 'timed-out',
            variability_map=get_variability_map(Q), analyze_noisy=False,
        ) or [base_path / folder]
        profs = snapshot_profile(base_path / 'cached_data', folder.name, run_paths)
        if profs is None:
            continue
        for m in METRICS:
            slopes[m][(Q, pm, n)] = profile_slope(profs[m])
        if n == 0:
            y_const[Q] = profs

noisy_profiles = []
for subfolder in NOISY_SUBFOLDERS:
    print(f"Loading noisy run: {subfolder}")
    profs = snapshot_profile(NOISY_BASE_PATH / 'cached_data', subfolder, [NOISY_BASE_PATH / subfolder])
    if profs is not None:
        noisy_profiles.append(profs)

slope_band = {}   # {metric: (mean - 2 sigma, mean + 2 sigma)} at REF_Q
if noisy_profiles:
    refs = noisy_profiles + ([y_const[REF_Q]] if REF_Q in y_const else [])
    for m in METRICS:
        ref_slopes = [profile_slope(p[m]) for p in refs]
        mean, std = np.nanmean(ref_slopes), np.nanstd(ref_slopes)
        slope_band[m] = (mean - 2 * std, mean + 2 * std)

pm_by_q_n = {}   # {(Q, n_val): [pm_val, ...]}
for Q, pm, n in sorted(slopes[METRICS[0]]):
    if n > 0:
        pm_by_q_n.setdefault((Q, n), []).append(pm)
n_vals = sorted(n for (Q, n), pms in pm_by_q_n.items() if Q == REF_Q and len(pms) > 1)
all_pm_vals = sorted({pm for pms in pm_by_q_n.values() for pm in pms})
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / max(len(all_pm_vals) - 1, 1)) for i, pm in enumerate(all_pm_vals)}


#%% --- 4. FIGURES (one per metric) ---
legend_handles = []
if slope_band:
    legend_handles.append(mpatches.Patch(facecolor=ENV_COLOR, alpha=ENV_ALPHA,
                                         label=f'autogenic variability ($Q$ = {REF_Q})'))
legend_handles.append(mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                                    label='constant'))
legend_handles += [mlines.Line2D([], [], color='white', marker=DISCHARGE_MARKERS[Q], linestyle='none',
                                 markersize=MARKER_SIZE, markeredgecolor='k', markeredgewidth=0.5,
                                 label=f'$Q$ = {Q} m³/s') for Q in DISCHARGES]

for metric, ylabel in FIGURE_SPECS:
    fig, axes = plt.subplots(
        1, len(n_vals), figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
        sharex=True, sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    for ax, n_val, letter in zip(axes, n_vals, 'abcdefghij'):
        q_list = [Q for Q in DISCHARGES if len(pm_by_q_n.get((Q, n_val), [])) > 1]
        for Q in q_list:
            if Q == REF_Q and metric in slope_band:
                ax.axhspan(*slope_band[metric], color=ENV_COLOR, alpha=ENV_ALPHA, linewidth=0, zorder=1)
            slope_const = slopes[metric].get((Q, 1, 0), np.nan)
            if np.isfinite(slope_const):
                ax.axhline(slope_const, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', zorder=2)
                ax.plot(1, slope_const, DISCHARGE_MARKERS[Q], color=GREY_CONST, markersize=MARKER_SIZE,
                        markeredgecolor='k', markeredgewidth=0.3, zorder=4)

            pms = pm_by_q_n[(Q, n_val)]
            s_vals = [slopes[metric][(Q, pm, n_val)] for pm in pms]
            ax.plot(pms, s_vals, color=TREND_COLOR, linewidth=0.6, zorder=3)
            for pm_val, s in zip(pms, s_vals):
                ax.plot(pm_val, s, DISCHARGE_MARKERS[Q], color=PM_COLOR[pm_val], markersize=MARKER_SIZE,
                        markeredgecolor='k', markeredgewidth=0.3, zorder=4)

        ax.set_xlim(0.4, all_pm_vals[-1] + 0.6)
        ax.set_xticks([1] + all_pm_vals)
        ax.set_xlabel(f'Peak ratio {RPEAK} (1 = constant)')
        ax.set_title(f'{NPEAKS} = {_fmt(n_val)}', loc='left')
        ax.grid(True, color='0.9', linewidth=0.5)
        ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, ha='right', va='bottom')

    axes[0].set_ylabel(ylabel)
    fig.legend(handles=legend_handles, loc='outside lower center', ncol=len(legend_handles), frameon=True)

    save_name = (f"AGU_{metric}_slope_pm_n_allQ_{TARGET_DATE}_{MASK_MODE}"
                 f"_fit{FIT_RANGE_KM[0]}-{FIT_RANGE_KM[1]}km")
    save_agu_figure(fig, OUTPUT_DIR / save_name)
plt.show()

# %%
