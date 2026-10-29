"""Where the bars are raised, and how far the tide still reaches, along the estuary.

Discharge peaks raise the bars from the river end seaward; the raised bars push the tidal
intrusion limit seaward. Same view as the tidal-intrusion panel of
plot_map_hydro_tidalmetrics_comparison_allQ.py (x = along-estuary distance, y = R_peak),
with the reach of raised bars added.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one panel per discharge
(Q = 250 / 500 / 1000 m3/s: tide- to river-dominated), n_peaks = PROFILE_N series:
  blue bar     reach with raised bars: from the river end (km 45) seaward to where the p95
               bed level drops back to that of the constant run (within the constant run's
               autogenic variability: K_SIGMA x its temporal std over the last EVAL_YEARS).
  marker       tidal intrusion limit (landward limit of flood flow, mean-flow detailed-hydro
               run on the final morphology), as in the tidal-metrics figure.
  dashed line  tidal intrusion limit of the constant run.

Data: map caches of the morphodynamic runs (bars); LTI_summary.csv of
plot_map_hydro_tidalmetrics_comparison_allQ.py (tidal intrusion limits).
Writes bar_extent_summary.csv (reach end and tidal intrusion limit per run).
"""
#%% --- Imports ---
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style, find_variability_model_folders, get_variability_map
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_bedlevel_percentiles import bedlevel_profiles

#%% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
LTI_CSV = BASE_DIR / 'output_plots_combined' / 'hydro_tidalmetrics_comparison' / 'LTI_summary.csv'
DISCHARGES = [250, 500, 1000]
PROFILE_N = 3                  # n_peaks of the R_peak series run at every Q
TARGET_DATE = np.datetime64('2031-01-01')

MASK_MODE = 'union'            # 'frozen' | 'union' (see functions/F_bedlevel_percentiles.py)
CHANNEL_INIT_THRESHOLD = 2.2   # channel = bed below this [m]
X_BINS = np.arange(20000, 45000, 1000)   # 1 km bins, 20-44 km
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
y_range = (5000, 10000)
RIVER_END_KM = 45.0

# Raised-bar detection
EVAL_YEARS = 3                 # window of the constant run for its autogenic variability
K_SIGMA = 2.0                  # raised = p95 above constant by more than K_SIGMA x std ...
MIN_RISE_M = 0.05              # ... and by at least this much
MAX_GAP_BINS = 1               # bins below the threshold tolerated inside the raised reach

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None

RPEAK = r'$R_{\mathrm{peak}}$'
GREY_CONST = "#7f7f7f"
DISCHARGE_COLORS = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}
DISCHARGE_MARKERS = {250: '^', 500: 'o', 1000: 's'}
BAR_LINE_WIDTH = 6
MARKER_SIZE = 6
XTICK_STEP_KM = 5

# --- AGU figure sizing ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 62
FONT_SIZE = 8

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'bar_extent_tidal_intrusion'
CACHE_DIR = OUTPUT_DIR / 'profile_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


#%% --- 2. HELPERS ---
def p95_profiles(q, folder):
    """p95 bed level per x-bin at TARGET_DATE, and the p95 profiles of every map output in the
    last EVAL_YEARS (for the autogenic variability), cached per run."""
    cache = CACHE_DIR / f"{folder.name}_{TARGET_DATE}_{MASK_MODE}_eval{EVAL_YEARS}y.npz"
    if cache.exists():
        d = np.load(cache)
        return d['now'], d['window']
    base_path = BASE_DIR / f"Q{q}"
    run_paths = get_stitched_map_run_paths(
        base_path=base_path, folder_name=folder.name, timed_out_dir=base_path / 'timed-out',
        variability_map=get_variability_map(q), analyze_noisy=False,
    ) or [base_path / folder]
    ds = load_or_update_map_cache_multi(
        cache_dir=base_path / 'cached_data', folder_name=folder.name, run_paths=run_paths,
        var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    times = np.asarray(ds.time.values).astype('datetime64[ns]')
    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
    init_bl = ds['mesh2d_mor_bl'].isel(time=int(np.argmin(times))).values.copy()

    def prof(i):
        bed = ds['mesh2d_mor_bl'].isel(time=int(i)).values.copy()
        return bedlevel_profiles(bed, init_bl, face_x, face_y, width_mask, X_BINS, ['p95'],
                                 MASK_MODE, threshold=CHANNEL_INIT_THRESHOLD)[0]['p95']

    now = prof(np.argmin(np.abs(times - TARGET_DATE)))
    t_start = times.max() - np.timedelta64(int(EVAL_YEARS * 365.25 * 86400), 's')
    window = np.array([prof(i) for i in np.flatnonzero(times >= t_start)])
    ds.close()
    np.savez(cache, now=now, window=window)
    return now, window


def raised_reach_end(raised):
    """Seaward end [km] of the raised reach, walking from the river towards the sea; the reach
    ends where more than MAX_GAP_BINS consecutive bins are not raised. NaN if none is raised."""
    last, gap = None, 0
    for i in range(raised.size - 1, -1, -1):
        if raised[i]:
            last, gap = i, 0
        elif last is not None:
            gap += 1
            if gap > MAX_GAP_BINS:
                break
    return np.nan if last is None else X_BINS[last] / 1000


#%% --- 3. COMPUTE ---
lti = pd.read_csv(LTI_CSV)
rows = []
for q in DISCHARGES:
    runs = {}
    for folder in find_variability_model_folders(base_path=BASE_DIR / f"Q{q}", discharge=q,
                                                 scenarios_to_process=None, analyze_noisy=False):
        m = re.search(r'_pm(\d+)_n(\d+)', folder.name)
        if m and int(m.group(2)) in (0, PROFILE_N):
            runs[(int(m.group(1)), int(m.group(2)))] = folder
    if (1, 0) not in runs:
        print(f"[SKIP] Q = {q}: no constant run")
        continue
    const_now, const_window = p95_profiles(q, runs[(1, 0)])
    threshold = np.maximum(K_SIGMA * np.nanstd(const_window, axis=0), MIN_RISE_M)
    for (pm, n), folder in sorted(runs.items()):
        lim = lti[(lti.discharge == q) & (lti.pm == pm) & (lti.n == n)].LTI_instant_max_km
        reach_end = np.nan
        if n > 0:
            now, _ = p95_profiles(q, folder)
            reach_end = raised_reach_end(np.nan_to_num(now - const_now) > threshold)
        rows.append({'discharge': q, 'pm': pm, 'n': n, 'raised_bars_from_km': reach_end,
                     'tidal_intrusion_limit_km': lim.iloc[0] if len(lim) else np.nan})

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / 'bar_extent_summary.csv', index=False)
print(summary.to_string(index=False))

q_list = [q for q in DISCHARGES if q in set(summary.discharge)]
all_pm = sorted(summary.loc[summary.n > 0, 'pm'].unique())
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / max(len(all_pm) - 1, 1)) for i, pm in enumerate(all_pm)}


#%% --- 4. FIGURE ---
fig, axes = plt.subplots(1, len(q_list), sharex=True, sharey=True, squeeze=False,
                         figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
                         constrained_layout=True)
axes = axes[0]
for c, (q, ax) in enumerate(zip(q_list, axes)):
    s = summary[summary.discharge == q]
    col, mk = DISCHARGE_COLORS[q], DISCHARGE_MARKERS[q]
    const_lim = s.loc[s.n == 0, 'tidal_intrusion_limit_km']
    if len(const_lim):
        ax.axvline(const_lim.iloc[0], color=col, linewidth=1.2, linestyle='--', zorder=1)
    for _, r in s.iterrows():
        if np.isfinite(r.raised_bars_from_km):
            ax.plot([r.raised_bars_from_km, RIVER_END_KM], [r.pm, r.pm], color=PM_COLOR[r.pm],
                    linewidth=BAR_LINE_WIDTH, solid_capstyle='butt', zorder=2)
        ax.plot(r.tidal_intrusion_limit_km, r.pm, marker=mk, color=col, markersize=MARKER_SIZE,
                markeredgecolor='k', markeredgewidth=0.5, linestyle='', zorder=4)
    ax.set_title(f'$Q$ = {q} m\u00b3/s')
    ax.set_xlabel('Along-estuary distance [km]')
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(-0.02, 1.02, f'({chr(ord("a") + c)})', transform=ax.transAxes, ha='right', va='bottom')
    if c == 0:
        ax.set_ylabel(f'Peak ratio {RPEAK} (1 = constant)')
axes[0].set_xlim(X_BINS[0] / 1000, RIVER_END_KM)
axes[0].xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
axes[0].set_yticks([1] + all_pm)
axes[0].set_ylim(0.5, all_pm[-1] + 0.5)

legend_handles = [mlines.Line2D([], [], color=PM_COLOR[all_pm[-1]], linewidth=BAR_LINE_WIDTH,
                                solid_capstyle='butt', label='reach with raised bars')]
legend_handles += [mlines.Line2D([], [], color=DISCHARGE_COLORS[q], marker=DISCHARGE_MARKERS[q], linestyle='',
                                 markersize=MARKER_SIZE, markeredgecolor='k', markeredgewidth=0.5,
                                 label=f'tidal intrusion limit, $Q$ = {q}') for q in q_list]
legend_handles.append(mlines.Line2D([], [], color='0.4', linewidth=1.2, linestyle='--',
                                    label='tidal intrusion limit, constant'))
fig.legend(handles=legend_handles, loc='outside lower center', ncol=3, frameon=True)

save_name = f"AGU_bar_extent_tidal_intrusion_allQ_n{PROFILE_N}"
fig.savefig(OUTPUT_DIR / f"{save_name}.png", dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {OUTPUT_DIR / save_name}.png")

# %%
