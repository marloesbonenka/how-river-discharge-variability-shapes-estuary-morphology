"""Raised bars, tidal intrusion and tidal prism: where the bars are raised, how far the tide still
reaches along the estuary, and how the tidal prism at the mouth changes.

Discharge peaks raise the bars from the river end seaward; the raised bars push the tidal
intrusion limit seaward. Same view as the tidal-intrusion panel of
plot_map_hydro_tidalmetrics_comparison_allQ.py (x = along-estuary distance, y = R_peak),
with the reach of raised bars added.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), n_peaks = PROFILE_N series, one row
with a shared R_peak axis:
(a)   tidal prism at the mouth (km 20) [10^6 m3], all discharges (marker shape = discharge, as
      in b-d; direct labels = Q [m3/s]; grey marker + dashed line = constant run). Arrow+label:
      largest relative change vs. the constant run [%], so absolute and relative read together.
      Slightly wider than b-d.
(b-d) one panel per discharge (Q = 250 / 500 / 1000 m3/s: tide- to river-dominated), all at the
      same km-per-mm scale; a panel whose raised reach reaches the seaward end of the analysis
      bins (km 20) starts REACH_PAD_KM further seaward so the bar visibly ends:
  blue bar     reach with raised bars: from the river end (km 45) seaward to where the p95
               bed level drops back to that of the constant run (within the constant run's
               autogenic variability: K_SIGMA x its temporal std over the last EVAL_YEARS).
  marker       tidal intrusion limit (landward limit of flood flow, mean-flow detailed-hydro
               run on the final morphology), as in the tidal-metrics figure; grey = constant run.
  dashed line  tidal intrusion limit of the constant run.
  arrow+label  largest reduction of the tidal intrusion length in that panel, normalised by
               the constant run: dL_TI = (L_TI - L_TI,const) / L_TI,const, with L_TI the
               intrusion length measured from the mouth (X_MOUTH_KM). NB: at high Q the
               constant-run intrusion length is short (Q1000: 5.45 km), so a small shift in km
               gives a large percentage.

Data: map caches of the morphodynamic runs (bars); LTI_summary.csv and tidal_prism_summary.csv
of plot_map_hydro_tidalmetrics_comparison_allQ.py (tidal intrusion limits, tidal prism; both
from the mean-flow detailed-hydro runs on the final morphology).
Writes raised_bars_summary.csv (reach end, tidal intrusion limit, dL_TI and prism change per run).
"""
#%% --- Imports ---
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style, find_variability_model_folders, get_variability_map, save_agu_figure
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_bedlevel_percentiles import bedlevel_profiles

#%% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
LTI_CSV = BASE_DIR / 'output_plots_combined' / 'hydro_tidalmetrics_comparison' / 'LTI_summary.csv'
PRISM_CSV = BASE_DIR / 'output_plots_combined' / 'hydro_tidalmetrics_comparison' / 'tidal_prism_summary.csv'
DISCHARGES = [250, 500, 1000]
PROFILE_N = 3                  # n_peaks of the R_peak series run at every Q
TARGET_DATE = np.datetime64('2031-01-01')

MASK_MODE = 'union'            # 'frozen' | 'union' (see functions/F_bedlevel_percentiles.py)
CHANNEL_INIT_THRESHOLD = 2.2   # channel = bed below this [m]
X_BINS = np.arange(20000, 45000, 1000)   # 1 km bins, 20-44 km
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
y_range = (5000, 10000)
RIVER_END_KM = 45.0
X_MOUTH_KM = 20.0              # estuary mouth; tidal intrusion length is measured from here

# Raised-bar detection
EVAL_YEARS = 3                 # window of the constant run for its autogenic variability
K_SIGMA = 2.0                  # raised = p95 above constant by more than K_SIGMA x std ...
MIN_RISE_M = 0.05              # ... and by at least this much
MAX_GAP_BINS = 2               # bins below the threshold tolerated inside the raised reach
                               # (2: Q500 R_peak 2 dips into the envelope at km 36-37 only)

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None

RPEAK = r'$R_{\mathrm{peak}}$'
GREY_CONST = "#7f7f7f"
DISCHARGE_MARKERS = {250: '^', 500: 'o', 1000: 's'}
LINE_WIDTH_CONST = 0.9
BAR_LINE_WIDTH = 6
MARKER_SIZE = 5
LTI_LABEL_DY = 0.22            # arrow + label of the largest reduction: this far below its row [R_peak units]
XTICK_STEP_KM = 5
REACH_PAD_KM = 2.0             # extra x-range seaward of km 20 when a raised reach runs up to km 20
PRISM_WIDTH_RATIO = 1.35       # width of the prism panel relative to a 25-km bar panel
PRISM_XTICK_STEP = 25          # tidal prism panel [10^6 m3]
PRISM_XLIM = (30, 160)         # room for the direct labels around the outermost markers
TREND_COLOR = '0.6'            # thin line connecting the prism markers of one discharge

# --- AGU figure sizing ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 62
FONT_SIZE = 8

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'raised_bars_tidal_intrusion_prism'
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
# Normalised change of the tidal intrusion length (from the mouth) relative to the constant run
L_const = (summary[summary.n == 0].set_index('discharge').tidal_intrusion_limit_km - X_MOUTH_KM)
summary['tidal_intrusion_change_rel'] = ((summary.tidal_intrusion_limit_km - X_MOUTH_KM)
                                         / summary.discharge.map(L_const) - 1)
# Tidal prism at the mouth relative to the constant run
prism = pd.read_csv(PRISM_CSV)[['discharge', 'pm', 'n', 'prism_Mm3']]
summary = summary.merge(prism, on=['discharge', 'pm', 'n'], how='left')
P_const = summary[summary.n == 0].set_index('discharge').prism_Mm3
summary['tidal_prism_change_rel'] = summary.prism_Mm3 / summary.discharge.map(P_const) - 1
summary.to_csv(OUTPUT_DIR / 'raised_bars_summary.csv', index=False)
print(summary.to_string(index=False))

q_list = [q for q in DISCHARGES if q in set(summary.discharge)]
all_pm = sorted(summary.loc[summary.n > 0, 'pm'].unique())
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / max(len(all_pm) - 1, 1)) for i, pm in enumerate(all_pm)}


#%% --- 4. FIGURE ---
# x-range per bar panel: start further seaward when the raised reach runs up to km 20 (the seaward
# end of the analysis bins), so the bar visibly ends instead of being cut by the y-axis
x0_bins = X_BINS[0] / 1000
bar_xlim = {}
for q in q_list:
    reach_min = summary.loc[summary.discharge == q, 'raised_bars_from_km'].min()
    bar_xlim[q] = (x0_bins - REACH_PAD_KM if reach_min <= x0_bins else x0_bins, RIVER_END_KM)
# width ratios: prism panel wider; bar panels proportional to their x-range (same km-per-mm scale)
width_ratios = [PRISM_WIDTH_RATIO] + [(bar_xlim[q][1] - bar_xlim[q][0]) / (RIVER_END_KM - x0_bins)
                                      for q in q_list]

fig, axes = plt.subplots(1, len(q_list) + 1, sharey=True, squeeze=False,
                         figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
                         gridspec_kw={'width_ratios': width_ratios}, constrained_layout=True)
axes = axes[0]
ax_prism, bar_axes = axes[0], axes[1:]

# (b-d) raised-bar reach and tidal intrusion limit, one panel per discharge
lti_labels = []   # direct labels of the largest reduction; kept inside their panel after layout
for c, (q, ax) in enumerate(zip(q_list, bar_axes)):
    s = summary[summary.discharge == q]
    const_lim = s.loc[s.n == 0, 'tidal_intrusion_limit_km']
    if len(const_lim):
        ax.axvline(const_lim.iloc[0], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', zorder=1)
    for _, r in s.iterrows():
        if np.isfinite(r.raised_bars_from_km):
            ax.plot([r.raised_bars_from_km, RIVER_END_KM], [r.pm, r.pm], color=PM_COLOR[r.pm],
                    linewidth=BAR_LINE_WIDTH, solid_capstyle='butt', zorder=2)
        ax.plot(r.tidal_intrusion_limit_km, r.pm, marker=DISCHARGE_MARKERS[q],
                color=GREY_CONST if r.n == 0 else 'white',
                markersize=MARKER_SIZE, markeredgecolor='k', markeredgewidth=0.6, linestyle='', zorder=4)
    var = s[(s.n > 0) & s.tidal_intrusion_change_rel.notna()]
    if len(const_lim) and len(var):   # largest reduction of the tidal intrusion: arrow + direct label
        r = var.loc[var.tidal_intrusion_change_rel.idxmin()]
        y, col = r.pm - LTI_LABEL_DY, PM_COLOR[r.pm]
        ax.annotate('', xy=(r.tidal_intrusion_limit_km, y), xytext=(const_lim.iloc[0], y),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=0.8, mutation_scale=6,
                                    shrinkA=0, shrinkB=0), zorder=5)
        d_km = r.tidal_intrusion_limit_km - const_lim.iloc[0]
        lti_labels.append(ax.text(0.5 * (r.tidal_intrusion_limit_km + const_lim.iloc[0]), y - 0.05,
                                  f"{d_km:+.1f} km ({r.tidal_intrusion_change_rel * 100:+.0f}%)".replace('-', '−'),
                                  ha='center', va='top', color=col, zorder=6,
                                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5)))
    ax.set_title(f'$Q$ = {q} m\u00b3/s')
    ax.set_xlabel('Along-estuary distance [km]')
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(-0.02, 1.02, f'({chr(ord("b") + c)})', transform=ax.transAxes, ha='right', va='bottom')
    ax.set_xlim(bar_xlim[q])
    ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
ax_prism.set_ylabel(f'Peak ratio {RPEAK} (1 = constant)')
ax_prism.set_yticks([1] + all_pm)
ax_prism.set_ylim(0.5, all_pm[-1] + 0.5)

# (a) tidal prism, all discharges: absolute value on the x-axis (constant run = grey marker +
# dashed line, as in b-d), largest relative change vs. constant as arrow + direct label
for q in q_list:
    s = summary[(summary.discharge == q) & summary.prism_Mm3.notna()].sort_values('pm')
    if s.empty:
        continue
    p_const = P_const.get(q, np.nan)
    if np.isfinite(p_const):
        ax_prism.axvline(p_const, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', zorder=1)
    ax_prism.plot(s.prism_Mm3, s.pm, color=TREND_COLOR, linewidth=0.6, zorder=2)
    for _, r in s.iterrows():
        ax_prism.plot(r.prism_Mm3, r.pm, marker=DISCHARGE_MARKERS[q],
                      color=GREY_CONST if r.n == 0 else PM_COLOR[r.pm], markersize=MARKER_SIZE,
                      markeredgecolor='k', markeredgewidth=0.6, linestyle='', zorder=3)
    top = s.iloc[-1]   # direct label: Q [m3/s] above the R_peak = max marker
    ax_prism.annotate(f'{q}', (top.prism_Mm3, top.pm), xytext=(0, 3.5),
                      textcoords='offset points', ha='center', va='bottom')
    var = s[(s.n > 0) & s.tidal_prism_change_rel.notna()]
    if np.isfinite(p_const) and len(var):   # largest relative reduction: arrow + direct label
        r = var.loc[var.tidal_prism_change_rel.idxmin()]
        y, col = r.pm - LTI_LABEL_DY, PM_COLOR[r.pm]
        ax_prism.annotate('', xy=(r.prism_Mm3, y), xytext=(p_const, y),
                          arrowprops=dict(arrowstyle='-|>', color=col, lw=0.8, mutation_scale=6,
                                          shrinkA=0, shrinkB=0), zorder=5)
        ax_prism.text(0.5 * (r.prism_Mm3 + p_const), y - 0.05,
                      f"{r.tidal_prism_change_rel * 100:+.0f}%".replace('-', '−'),
                      ha='center', va='top', color=col, zorder=6,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
ax_prism.set_xlim(PRISM_XLIM)
ax_prism.xaxis.set_major_locator(MultipleLocator(PRISM_XTICK_STEP))
ax_prism.set_xlabel('Tidal prism [10$^6$ m³]')
ax_prism.set_title('Tidal prism')
ax_prism.grid(True, color='0.9', linewidth=0.5)
ax_prism.text(-0.02, 1.02, '(a)', transform=ax_prism.transAxes, ha='right', va='bottom')

legend_handles = [mlines.Line2D([], [], color=PM_COLOR[all_pm[-1]], linewidth=BAR_LINE_WIDTH,
                                solid_capstyle='butt', label='reach with raised bars')]
legend_handles.append(tuple(mlines.Line2D([], [], color='white', marker=DISCHARGE_MARKERS[q], linestyle='',
                                          markersize=MARKER_SIZE, markeredgecolor='k', markeredgewidth=0.6)
                            for q in q_list))
legend_handles.append(mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                                    label='tidal intrusion limit, constant'))
legend_labels = ['reach with raised bars', 'tidal intrusion limit', 'tidal intrusion limit, constant']
fig.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
           loc='outside lower center', ncol=len(legend_handles), frameon=True)

# shift the "km (%)" labels horizontally so they stay inside their panel (after the final layout)
fig.canvas.draw()
for t in lti_labels:
    ax = t.axes
    bb, ab = t.get_window_extent(), ax.get_window_extent()
    pad = 2   # px
    dx_px = max(ab.x0 + pad - bb.x0, 0) - max(bb.x1 - (ab.x1 - pad), 0)
    if dx_px:
        x_disp, y_disp = ax.transData.transform(t.get_position())
        t.set_position(ax.transData.inverted().transform((x_disp + dx_px, y_disp)))

save_agu_figure(fig, OUTPUT_DIR / f"AGU_raised_bars_tidal_intrusion_prism_allQ_n{PROFILE_N}")
plt.show()

# %%
