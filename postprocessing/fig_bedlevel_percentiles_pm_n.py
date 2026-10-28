"""Bed level p95 / p5 along the estuary vs. R_peak, one column per n_peaks (single snapshot).

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide):
  - Rows    (top -> bottom): p95, p5 bed level per 1 km x-bin.
  - Columns (left -> right): one per n_peaks value (the constant pm1_n0 run is the
                             grey dashed reference in every panel).
  - Colours: R_peak value, Blues palette (light -> dark).

Bed-level population per x-bin: MASK_MODE (default 'union': faces below
CHANNEL_INIT_THRESHOLD at t=0 OR at the snapshot, so channel widening is included);
see functions/F_bedlevel_percentiles.py. The grey band is the +/-2 sigma natural
variability of the noisy repeats of the constant run (plus the constant run itself),
computed on the same population.

Diagnostic version (all snapshots, R_peak- and n_peaks-effect figures, difference /
detrended): plot_sensitivity_pm_n_bedlevel_percentiles.py.
"""
#%% --- Imports ---
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import (
    _parse_pm_n,
    _scenario_label,
    apply_plot_style,
    get_variability_map,
    find_variability_model_folders,
    get_snapshot_matches_by_target_dates,
    sort_scenario_keys,
    group_snapshot_by_scenario,
    stack_metric_arrays,
)
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_bedlevel_percentiles import bedlevel_profiles

#%% --- 1. SETTINGS ---
DISCHARGE = 500
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
base_path = BASE_DIR / f"Q{DISCHARGE}"

TARGET_DATE = np.datetime64('2031-01-01')   # nearest available timestep per run is used

MASK_MODE = 'union'            # 'frozen' | 'union' (see functions/F_bedlevel_percentiles.py)
CHANNEL_INIT_THRESHOLD = 2.2   # channel = bed below this [m]
X_BINS = np.arange(20000, 45000, 1000)   # 1 km bins, 20-44 km
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
y_range = (5000, 10000)

# Row definitions: (metric, row y-label)
ROW_SPECS = [
    ('p95', 'p95 bed level [m]'),
    ('p5', 'p5 bed level [m]'),
]
METRICS = [metric for metric, _ in ROW_SPECS]

XTICK_STEP_KM = 5   # x-axis tick spacing (distance along estuary), in km
YTICK_STEP_M = 2    # y-axis tick spacing (bed level), in m

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Natural variability envelope — noisy repeats of the constant scenario
SHOW_NOISY_ENVELOPE = True
NOISY_BASE_PATH = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\1_RiverDischargeVariability_domain45x15"
    r"\Model_Output\Q500\0_Noise_Q500"
)
NOISY_SUBFOLDERS = [
    '1_Q500_noisy0.9095347',
    '1_Q500_noisy1_rst.9160657',
    '1_Q500_noisy2_rst.9160663',
]

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

RPEAK  = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'

GREY_CONST = "#7f7f7f"
ENV_COLOR = '0.85'       # opaque light grey (no alpha: looks the same on any background)
LINE_WIDTH = 1.0         # scenario lines (plot and legend)
LINE_WIDTH_CONST = 0.9   # constant reference line (plot and legend)

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170   # AGU full-page width
FONT_SIZE = 8           # one size for all text: ticks, labels, titles, legend

# Margins / gaps in inches, so the layout fills the full width at the final print size
LEFT_MARGIN_IN = 0.42    # y tick labels + row y-labels
RIGHT_MARGIN_IN = 0.10   # small buffer
WSPACE_IN = 0.08         # gap between columns (shared y axis per row, no inner tick labels)
HSPACE_IN = 0.10         # gap between rows (shared x axis, no inner tick labels)
PANEL_ASPECT = 0.85      # panel height / width
TOP_MARGIN_IN = 0.48     # figure suptitle + column titles
XLABEL_GAP_IN = 0.27     # bottom of axes -> top of the x-axis label (room for x tick labels)
LEGEND_GAP_IN = 0.45     # bottom of axes -> top of the legend
BOTTOM_MARGIN_IN = 0.72  # x tick labels + x-axis label + legend

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'bedlevel_percentiles_pm_n'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assessment_dir = base_path / 'cached_data'
assessment_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / 'timed-out'


#%% --- 2. HELPER ---
def _snapshot_profiles(ds):
    """{metric: value per x-bin} at TARGET_DATE on the MASK_MODE population (None if no timesteps)."""
    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        return None, None
    _, ts_idx, actual_dt = matches[0]

    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
    bed_now = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

    profs, _ = bedlevel_profiles(bed_now, init_bl, face_x, face_y, width_mask, X_BINS, METRICS,
                                 MASK_MODE, threshold=CHANNEL_INIT_THRESHOLD)
    return profs, actual_dt


#%% --- 3. LOAD MODEL DATA (single snapshot, all runs) ---
VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path, discharge=DISCHARGE, scenarios_to_process=None, analyze_noisy=False,
)

scenario_results = {}  # {folder_str: {metric: array per x-bin}}

for folder in model_folders:
    folder_str = folder.name
    print(f"Processing: {folder_str}")

    run_paths = get_stitched_map_run_paths(
        base_path=base_path, folder_name=folder.name,
        timed_out_dir=timed_out_dir, variability_map=VARIABILITY_MAP, analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [base_path / folder]

    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir, folder_name=folder.name, run_paths=run_paths,
        var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS, append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        print(f"  No cached data for {folder_str}, skipping.")
        continue

    profs, actual_dt = _snapshot_profiles(ds)
    ds.close()
    if profs is None:
        print(f"  No timesteps found for {folder_str}, skipping.")
        continue
    scenario_results[folder_str] = profs
    print(f"  Snapshot {actual_dt}: computed {', '.join(METRICS)} ({MASK_MODE}).")


#%% --- 4. GROUP BY SCENARIO / PARSE pm,n ---
scenario_groups = group_snapshot_by_scenario(scenario_results)

scen_pm_n = {}
for scen_key in sort_scenario_keys(scenario_groups.keys()):
    pm, n = _parse_pm_n(_scenario_label(scen_key, SCENARIO_LABELS))
    if pm is not None:
        scen_pm_n[scen_key] = (pm, n)

baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

pm_by_n = {}   # {n_val: [(pm_val, scen_key), ...]}
for scen_key, (pm, n) in scen_pm_n.items():
    if n == 0:
        continue
    pm_by_n.setdefault(n, []).append((pm, scen_key))
for n in pm_by_n:
    pm_by_n[n].sort()

all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
sorted_n_vals = sorted(pm_by_n.keys())

_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}


def _get_y(scen_key, metric):
    """Mean metric profile across the runs of a scenario."""
    y_stack = stack_metric_arrays(scenario_groups[scen_key], metric)
    return None if y_stack is None else np.nanmean(y_stack, axis=0)


def _fmt(v):
    return str(int(v)) if v == int(v) else str(v)


y_const_by_metric = {m: (_get_y(baseline_scen, m) if baseline_scen else None) for m in METRICS}


#%% --- 5. NATURAL VARIABILITY ENVELOPE (±2σ, per metric) ---
noisy_metric_profiles = {m: [] for m in METRICS}

if SHOW_NOISY_ENVELOPE:
    if not NOISY_BASE_PATH.exists():
        print(f"[WARNING] Noisy base path not found: {NOISY_BASE_PATH}")
    else:
        noisy_cache_dir = NOISY_BASE_PATH / 'cached_data'
        noisy_cache_dir.mkdir(parents=True, exist_ok=True)

        for subfolder in NOISY_SUBFOLDERS:
            noisy_folder = NOISY_BASE_PATH / subfolder
            if not noisy_folder.exists():
                print(f"[WARNING] Noisy subfolder not found: {noisy_folder}")
                continue
            print(f"Loading noisy run: {subfolder}")

            ds_n = load_or_update_map_cache_multi(
                cache_dir=noisy_cache_dir, folder_name=subfolder, run_paths=[noisy_folder],
                var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX,
                append_time=APPEND_TIMESTEPS, append_vars=APPEND_VARIABLES,
                cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
            )
            if ds_n is None:
                print(f"  No cached data for {subfolder}, skipping.")
                continue

            profs_n, _ = _snapshot_profiles(ds_n)
            ds_n.close()
            if profs_n is None:
                print(f"  No timesteps found for {subfolder}, skipping.")
                continue
            for m in METRICS:
                noisy_metric_profiles[m].append(profs_n[m])
            print(f"  {subfolder}: OK")

envelope_by_metric = {}
for m, profs in noisy_metric_profiles.items():
    all_profs = list(profs)
    if not all_profs:
        continue
    if y_const_by_metric.get(m) is not None:
        all_profs.append(y_const_by_metric[m])
    stk = np.vstack(all_profs)
    mean, std = np.nanmean(stk, axis=0), np.nanstd(stk, axis=0)
    envelope_by_metric[m] = (mean - 2 * std, mean + 2 * std)


#%% --- 6. FIGURE / GRIDSPEC LAYOUT ---
n_cols = len(sorted_n_vals)
n_rows = len(ROW_SPECS)

fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
panel_width_in = (fig_width_in - LEFT_MARGIN_IN - RIGHT_MARGIN_IN - (n_cols - 1) * WSPACE_IN) / n_cols
panel_height_in = PANEL_ASPECT * panel_width_in
fig_height_in = TOP_MARGIN_IN + n_rows * panel_height_in + (n_rows - 1) * HSPACE_IN + BOTTOM_MARGIN_IN

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs = gridspec.GridSpec(
    n_rows, n_cols, figure=fig,
    left=LEFT_MARGIN_IN / fig_width_in,
    right=1 - RIGHT_MARGIN_IN / fig_width_in,
    top=1 - TOP_MARGIN_IN / fig_height_in,
    bottom=BOTTOM_MARGIN_IN / fig_height_in,
    wspace=WSPACE_IN / panel_width_in,
    hspace=HSPACE_IN / panel_height_in,
)

axes = np.empty((n_rows, n_cols), dtype=object)
ax_ref_x = None
for r in range(n_rows):
    ax_ref_y = None
    for c in range(n_cols):
        ax = fig.add_subplot(gs[r, c], sharex=ax_ref_x, sharey=ax_ref_y)
        if ax_ref_x is None:
            ax_ref_x = ax
        if ax_ref_y is None:
            ax_ref_y = ax
        axes[r, c] = ax


#%% --- 7. PLOT EACH PANEL ---
for r, (metric, row_label) in enumerate(ROW_SPECS):
    y_const = y_const_by_metric.get(metric)
    env = envelope_by_metric.get(metric)

    for c, n_val in enumerate(sorted_n_vals):
        ax = axes[r, c]

        if SHOW_NOISY_ENVELOPE and env is not None:
            ax.fill_between(X_KM, env[0], env[1], color=ENV_COLOR, linewidth=0, zorder=1.6)  # above grid (1.5)

        if y_const is not None:
            ax.plot(X_KM, y_const, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', zorder=2)

        for pm_val, scen_key in pm_by_n[n_val]:
            y = _get_y(scen_key, metric)
            if y is not None:
                ax.plot(X_KM, y, color=PM_COLOR[pm_val], linewidth=LINE_WIDTH, zorder=3)

        ax.set_xlim(X_BINS[0] / 1000, X_BINS[-1] / 1000)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
        ax.yaxis.set_major_locator(MultipleLocator(YTICK_STEP_M))
        ax.grid(True, color='0.9', linewidth=0.5)

        if r == 0:
            ax.set_title(f'{NPEAKS} = {_fmt(n_val)}')
        if r != n_rows - 1:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel(row_label)
        else:
            ax.tick_params(labelleft=False)

fig.align_ylabels(axes[:, 0])

#%% --- 8. SHARED X-LABEL + LEGEND + SAVE ---
# Centre the x-label and legend on the panels (not on the full figure incl. margins)
x_mid = (LEFT_MARGIN_IN + 0.5 * (fig_width_in - LEFT_MARGIN_IN - RIGHT_MARGIN_IN)) / fig_width_in
fig.supxlabel('distance along estuary [km]', x=x_mid,
              y=(BOTTOM_MARGIN_IN - XLABEL_GAP_IN) / fig_height_in, va='top')

legend_handles = []
if SHOW_NOISY_ENVELOPE and envelope_by_metric:
    legend_handles.append(
        mpatches.Patch(facecolor=ENV_COLOR, label=r'$\pm 2\sigma$ natural variability')
    )
legend_handles.append(
    mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', label='constant')
)
for pm_val in all_pm_vals:
    legend_handles.append(
        mlines.Line2D([], [], color=PM_COLOR[pm_val], linewidth=LINE_WIDTH, label=f'{RPEAK} = {_fmt(pm_val)}')
    )

fig.legend(
    handles=legend_handles, loc='upper center', ncol=len(legend_handles),
    bbox_to_anchor=(x_mid, (BOTTOM_MARGIN_IN - LEGEND_GAP_IN) / fig_height_in),
    frameon=True, handlelength=1.8, handletextpad=0.5, columnspacing=1.2, borderaxespad=0,
)

fig.suptitle(
    f"Bed level percentiles vs. {RPEAK} / {NPEAKS} \u2014 {TARGET_DATE} snapshot, $Q$ = {DISCHARGE} m\u00b3/s",
    x=x_mid, y=1 - 0.04 / fig_height_in, va='top',
)

save_name = f"AGU_bedlevel_percentiles_pm_n_Q{DISCHARGE}_{TARGET_DATE}_{MASK_MODE}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%
