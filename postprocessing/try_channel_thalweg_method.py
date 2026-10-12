"""pm/n sensitivity analysis – channel-depth (thalweg) metrics

Companion to plot_pm_n_sensitivity.py. A percentile over all cells in a 1 km bin mixes channel
depth with channel width. Here the thalweg is taken per grid column (= one cross-section) inside
the frozen t=0 channel footprint and only then summarised per bin.

METRIC:
  'Thalweg'      → median over columns of the mean of the N_DEEPEST deepest cells per column [m]
                   (raw bed elevation: more negative = deeper)
  'Relief'       → p95 bed level (bar) minus Thalweg [m]  (larger = stronger channel-bar relief)
  'Channel area' → median over columns of the cross-sectional area below REF_LEVEL [m²]

Layout identical to plot_pm_n_sensitivity.py:
  A) Effect of R_peak  – one panel per n_peaks
  B) Effect of n_peaks – one panel per R_peak
  each as absolute / difference-from-constant / detrended (change from initial bed).
The last cell compares pooled per-column thalweg distributions of selected scenarios
(e.g. the same-V_sed pair pm3_n5 vs pm5_n1).
"""

#%% IMPORTS
from pathlib import Path
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import (
    _date_to_filename_tag,
    _date_to_label,
    _scenario_label,
    _parse_pm_n,
    get_variability_map,
    find_variability_model_folders,
    get_target_snapshot_dates,
    get_snapshot_matches_by_target_dates,
    sort_scenario_keys,
    group_snapshot_by_scenario,
    stack_metric_arrays,
    apply_plot_style,
)

from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0')


#%% --- CONFIGURATION ---
DISCHARGE = 500

# Metric
METRIC    = 'Thalweg'   # 'Thalweg' | 'Relief' | 'Channel area'
N_DEEPEST = 3           # thalweg per column = mean of the N deepest cells (1 = pure minimum)
DX_COL    = 85          # grid cell size in x [m]; used to identify grid columns
DY        = 50          # grid cell size in y [m]; used for channel area
REF_LEVEL = 0.0         # reference level for channel area [m] (e.g. MSL or MLW)

SHOW_NOISY_ENVELOPE = True
SHOW_DIFFERENCE = True   # difference-from-constant plot
SHOW_DETRENDED  = True   # change relative to the initial bed

SNAPSHOT_TARGET_DATES = None
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))
SNAPSHOT_COUNT = 6

METRIC_INFO = {   # metric → (y-label, name used in titles)
    'Thalweg':      ('thalweg bed level [m]',                       'thalweg bed level'),
    'Relief':       ('bar-channel relief (p95 - thalweg) [m]',      'bar-channel relief'),
    'Channel area': (f'channel area below {REF_LEVEL:g} m [m$^2$]', 'channel area'),
}
assert METRIC in METRIC_INFO, f'METRIC must be one of {list(METRIC_INFO)}'
METRIC_KEYS = ['Thalweg', 'Thalweg p10', 'Thalweg p90', 'Bar p95', 'Relief', 'Channel area']


#%% --- CONSTANTS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f'Model_Output/Q{DISCHARGE}'

CHANNEL_INIT_THRESHOLD = 2.2   # defines the channel footprint from t=0
Y_RANGE   = (5000, 10000)
X_TARGETS = np.arange(20000, 44001, 1000)
DX_BIN    = 1000
X_BINS    = np.arange(X_TARGETS[0], X_TARGETS[-1] + DX_BIN, DX_BIN)
X_CENTERS = (X_BINS[:-1] + X_BINS[1:]) / 2

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

NOISY_BASE_PATH = Path(
    rf"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
    rf"\Model_Output\Q{DISCHARGE}\0_Noise_Q{DISCHARGE}"
)
NOISY_SUBFOLDERS = [
    '1_Q500_noisy0.9095347',
    '1_Q500_noisy1_rst.9160657',
    '1_Q500_noisy2_rst.9160663',
] if DISCHARGE == 500 else []

_ALL_LABELS = {
    '1': 'pm1_n0 (constant)', '2': 'pm2_n1', '3': 'pm3_n5', '4': 'pm3_n1', '5': 'pm5_n1',
    '6': 'pm4_n3', '7': 'pm3_n4', '8': 'pm2_n6', '9': 'pm5_n3', '10': 'pm3_n3',
    '11': 'pm2_n3', '12': 'pm5_n4', '13': 'pm4_n4', '14': 'pm2_n4',
}
_KEEP = {500: None, 1000: ['1', '6', '9', '10', '11'], 250: ['1', '6', '9', '10', '11']}
SCENARIO_LABELS = {k: v for k, v in _ALL_LABELS.items()
                   if _KEEP[DISCHARGE] is None or k in _KEEP[DISCHARGE]}

GREY_CONST = "#7f7f7f"

# Fixed axes dimensions — same across all figure variants so subplots align
AX_W, AX_H = 3.5, 3.0   # axes width / height in inches
_LEFT   = 0.95
_RIGHT  = 0.20
_TOP    = 1.30
_BOT    = 0.65
_WSPACE = 0.10

LINE_WIDTH       = 1.8
LINE_WIDTH_CONST = 1.5

FONTSIZE_TITLE  = 18
FONTSIZE_LABELS = FONTSIZE_TITLE - 4
FONTSIZE_TICKS  = FONTSIZE_LABELS - 2


#%% --- FIGURE STYLE ---
STYLE = 'default'   # 'default'  → white background, black text
                    # 'whitefig' → transparent figure, white axes background, white text

if STYLE == 'whitefig':
    apply_plot_style('whitefig', **{
        'axes.facecolor': 'white',
        'grid.color': '#cccccc',
        'savefig.transparent': False,
    })
else:
    apply_plot_style('default')
_tc = plt.rcParams['text.color']
_tr = plt.rcParams.get('savefig.transparent', False)


#%% --- HELPERS: per-column thalweg inside the frozen footprint ---
def build_column_layout(face_x, bin_mask, dx_col):
    """Frozen layout of one x-bin: face indices sorted by x, plus the start index of each grid column.

    Columns are separated where the sorted x jumps by more than half a cell, so no alignment of the
    grid with the bin edges is needed. Returns None when the bin has no footprint cells.
    """
    idx = np.flatnonzero(bin_mask)
    if idx.size == 0:
        return None
    idx = idx[np.argsort(face_x[idx], kind='stable')]
    starts = np.r_[0, np.flatnonzero(np.diff(face_x[idx]) > 0.5 * dx_col) + 1]
    return idx, starts


def bin_metrics(bl, layouts):
    """Per-bin metrics from one bed-level field.

    Returns (metrics, columns): metrics = {key in METRIC_KEYS: array over bins},
    columns = list over bins of the per-column thalweg values (for pooled distributions).
    """
    n_bins = len(layouts)
    out = {key: np.full(n_bins, np.nan) for key in METRIC_KEYS}
    columns = [np.array([]) for _ in range(n_bins)]

    for k, layout in enumerate(layouts):
        if layout is None:
            continue
        idx, starts = layout
        z = bl[idx]
        valid = ~np.isnan(z)
        if not valid.any():
            continue

        thal, area = [], []
        for c in np.split(z, starts[1:]):           # one array per grid column (cross-section)
            c = c[~np.isnan(c)]
            if c.size == 0:
                continue
            thal.append(np.sort(c)[:N_DEEPEST].mean())                  # thalweg of this column
            area.append(np.clip(REF_LEVEL - c, 0, None).sum() * DY)    # area below REF_LEVEL
        thal = np.array(thal)

        bar = np.percentile(z[valid], 95)                               # bar height, as before
        out['Thalweg'][k] = np.median(thal)
        out['Thalweg p10'][k], out['Thalweg p90'][k] = np.percentile(thal, [10, 90])  # p10 = deep end
        out['Bar p95'][k] = bar
        out['Relief'][k] = bar - out['Thalweg'][k]
        out['Channel area'][k] = np.median(area)
        columns[k] = thal
    return out, columns


def process_run(ds, target_dates):
    """Frozen t=0 footprint → per-bin metrics at t=0 and at every matching snapshot.

    Returns (init_metrics, snaps) with snaps = {snapshot_key: (target_dt, metrics, columns)}.
    """
    face_x, face_y = _get_face_coords(ds)
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
    footprint = (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1]) & (init_bl < CHANNEL_INIT_THRESHOLD)

    layouts = [
        build_column_layout(face_x, footprint & (face_x >= X_BINS[k]) & (face_x < X_BINS[k + 1]), DX_COL)
        for k in range(len(X_BINS) - 1)
    ]
    n_cols  = [len(lay[1]) for lay in layouts if lay is not None]
    n_cells = [lay[0].size / len(lay[1]) for lay in layouts if lay is not None]
    if n_cols:   # sanity check on DX_COL: expect ~1000/DX_COL columns per bin
        print(f"  footprint: median {np.median(n_cols):.0f} columns per bin, "
              f"{np.median(n_cells):.0f} cells per column")

    init_metrics, _ = bin_metrics(init_bl, layouts)

    snaps = {}
    for target_dt, ts_idx, _actual_dt in get_snapshot_matches_by_target_dates(ds.time.values, target_dates):
        bl = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
        metrics, columns = bin_metrics(bl, layouts)
        snaps[f"d{_date_to_filename_tag(target_dt)}"] = (target_dt, metrics, columns)
    return init_metrics, snaps


#%% --- SEARCH FOLDERS ---
base_path = base_directory / config
VARIABILITY_MAP = get_variability_map(DISCHARGE)

model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=None,
    analyze_noisy=False,
)

assessment_dir = base_path / 'cached_data'
assessment_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / 'timed-out'
output_dir = base_path / 'output_plots' / 'plots_pm_n_thalweg'
output_dir.mkdir(parents=True, exist_ok=True)


#%% --- LOAD DATA ---
comparison_results = {}   # {snapshot_key: {folder_str: {metric: per-bin array, 'x_centers': ...}}}
comparison_labels  = {}
initial_profiles   = {}   # {folder_str: {metric: per-bin array at t=0}}
thal_columns       = {}   # {snapshot_key: {folder_str: [per-bin arrays of per-column thalweg]}}

target_snapshot_dates = get_target_snapshot_dates(
    count=SNAPSHOT_COUNT,
    explicit_dates=SNAPSHOT_TARGET_DATES,
    date_range=SNAPSHOT_DATE_RANGE,
)

print("\nTarget hydrodynamic snapshot dates:")
for dt in target_snapshot_dates:
    print(f"  - {_date_to_label(dt)}")

for folder in model_folders:
    folder_str = folder.name
    print(f"\nProcessing: {folder_str}")

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [base_path / folder]

    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=['mesh2d_mor_bl'],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        print(f"  No cached data for {folder_str}, skipping.")
        continue

    init_metrics, snaps = process_run(ds, target_snapshot_dates)
    ds.close()
    if not snaps:
        print(f"  No timesteps found for {folder_str}, skipping.")
        continue

    initial_profiles[folder_str] = init_metrics
    for snapshot_key, (target_dt, metrics, columns) in snaps.items():
        comparison_results.setdefault(snapshot_key, {})[folder_str] = {**metrics, 'x_centers': X_CENTERS}
        comparison_labels[snapshot_key] = _date_to_label(target_dt)
        thal_columns.setdefault(snapshot_key, {})[folder_str] = columns
        print(f"  Snapshot {_date_to_label(target_dt)}: computed {METRIC}.")


#%% --- LOAD NOISY ENVELOPE DATA ---
noisy_profiles = {}   # {snapshot_key: [per-bin METRIC array, one per noisy run]}

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
                cache_dir=noisy_cache_dir,
                folder_name=subfolder,
                run_paths=[noisy_folder],
                var_names=['mesh2d_mor_bl'],
                bbox=CACHE_BBOX,
                append_time=APPEND_TIMESTEPS,
                append_vars=APPEND_VARIABLES,
                cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
            )
            if ds_n is None:
                print(f"  No cached data for {subfolder}, skipping.")
                continue

            _, snaps_n = process_run(ds_n, target_snapshot_dates)
            ds_n.close()
            for snapshot_key, (_, metrics, _) in snaps_n.items():
                noisy_profiles.setdefault(snapshot_key, []).append(metrics[METRIC])
        print(f"Noisy envelope ready for {len(noisy_profiles)} snapshots.")


#%% --- SENSITIVITY PLOTS ---
ylabel_base, title_metric = METRIC_INFO[METRIC]
metric_tag  = METRIC.lower().replace(' ', '_')
_noisy_tag  = 'noisy' if SHOW_NOISY_ENVELOPE else ''
plot_modes  = ['absolute'] + (['difference'] if SHOW_DIFFERENCE else []) + (['detrended'] if SHOW_DETRENDED else [])


def _fmt(v):
    return str(int(v)) if v == int(v) else str(v)


snapshot_keys = list(comparison_results.keys())
for snapshot_key in snapshot_keys:
    snapshot_results = comparison_results[snapshot_key]
    if not snapshot_results:
        continue

    is_last_snapshot = (snapshot_key == snapshot_keys[-1])
    scenario_groups = group_snapshot_by_scenario(snapshot_results)
    all_scen_keys = sort_scenario_keys(scenario_groups.keys())
    snap_label = comparison_labels.get(snapshot_key, snapshot_key)

    # Parse pm, n for each scenario
    scen_pm_n = {}
    for scen_key in all_scen_keys:
        pm, n = _parse_pm_n(_scenario_label(scen_key, SCENARIO_LABELS))
        if pm is not None:
            scen_pm_n[scen_key] = (pm, n)
    baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

    pm_by_n, n_by_pm = {}, {}   # {n: [(pm, scen_key), ...]}, {pm: [(n, scen_key), ...]}
    for scen_key, (pm, n) in scen_pm_n.items():
        if n == 0:
            continue
        pm_by_n.setdefault(n, []).append((pm, scen_key))
        n_by_pm.setdefault(pm, []).append((n, scen_key))
    for grp in (pm_by_n, n_by_pm):
        for key in grp:
            grp[key].sort()
    all_pm_vals = sorted(n_by_pm)
    all_n_vals  = sorted(pm_by_n)

    # Blues for pm, Greens for n (light → dark)
    _n_pm = max(len(all_pm_vals) - 1, 1)
    PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}
    _n_n = max(len(all_n_vals) - 1, 1)
    N_COLOR  = {n: plt.cm.Greens(0.35 + 0.55 * i / _n_n) for i, n in enumerate(all_n_vals)}

    def _get_y(scen_key):
        """Mean over runs of the per-bin metric."""
        stack = stack_metric_arrays(scenario_groups[scen_key], METRIC)
        return None if stack is None else np.nanmean(stack, axis=0)

    def _get_x(scen_key):
        x_data = next((d for _, d in scenario_groups[scen_key] if 'x_centers' in d), None)
        return x_data['x_centers'] / 1000 if x_data else X_TARGETS / 1000

    def _get_init(scen_key):
        """Mean over runs of the initial (t=0) per-bin metric."""
        profs = [initial_profiles[fn][METRIC] for fn, _ in scenario_groups[scen_key] if fn in initial_profiles]
        return np.nanmean(np.vstack(profs), axis=0) if profs else None

    y_const      = _get_y(baseline_scen) if baseline_scen else None
    x_const      = _get_x(baseline_scen) if baseline_scen else None
    y_init_const = _get_init(baseline_scen) if baseline_scen else None
    y_const_det  = (y_const - y_init_const) if (y_const is not None and y_init_const is not None) else None

    # ±2σ envelope from noisy repeats + constant run (absolute space)
    env = None
    if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_profiles:
        stk = np.vstack(noisy_profiles[snapshot_key] + ([y_const] if y_const is not None else []))
        env_m, env_s = np.nanmean(stk, axis=0), np.nanstd(stk, axis=0)
        env = (env_m - 2 * env_s, env_m + 2 * env_s)

    for plot_mode in plot_modes:
        if plot_mode == 'absolute':
            norm_tag, norm_title, ylabel = '', '', ylabel_base
            out_dir = output_dir
        elif plot_mode == 'difference':
            norm_tag, norm_title = '_difference', ' (difference from constant)'
            ylabel = f'{ylabel_base}\n(difference from constant)'
            out_dir = output_dir / 'difference_from_Qconstant'
        else:
            norm_tag, norm_title = '_detrended', ' (change from initial bed)'
            ylabel = f'{ylabel_base}\n(change from initial bed)'
            out_dir = output_dir / 'detrended'
        out_dir.mkdir(parents=True, exist_ok=True)

        def _transform(y, scen_key):
            if plot_mode == 'difference' and y_const is not None:
                return y - y_const
            if plot_mode == 'detrended':
                y_init = _get_init(scen_key)
                return y - y_init if y_init is not None else y
            return y

        # A) effect of R_peak (panel per n_peaks)   B) effect of n_peaks (panel per R_peak)
        for vary in ('pm', 'n'):
            groups   = pm_by_n if vary == 'pm' else n_by_pm   # {panel value: [(line value, scen_key), ...]}
            colors   = PM_COLOR if vary == 'pm' else N_COLOR
            all_vals = all_pm_vals if vary == 'pm' else all_n_vals
            sym_line, sym_panel = ((r'R_{\mathrm{peak}}', r'n_{\mathrm{peaks}}') if vary == 'pm'
                                   else (r'n_{\mathrm{peaks}}', r'R_{\mathrm{peak}}'))
            panel_vals = sorted(groups)
            if not panel_vals:
                continue

            n_panels = len(panel_vals)
            fig_w = _LEFT + n_panels * AX_W + (n_panels - 1) * _WSPACE + _RIGHT
            fig_h = _BOT + AX_H + _TOP
            fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h),
                                     sharey=True, sharex=False, squeeze=False)
            axes = axes[0]

            for ci, (ax, panel_val) in enumerate(zip(axes, panel_vals)):
                # grey dashed constant reference
                if plot_mode == 'absolute' and y_const is not None:
                    ax.plot(x_const, y_const, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', zorder=2)
                elif plot_mode == 'difference':
                    ax.axhline(0.0, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', zorder=2)
                elif plot_mode == 'detrended' and y_const_det is not None:
                    ax.plot(x_const, y_const_det, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', zorder=2)

                # natural variability envelope (not shown in detrended mode)
                if env is not None and plot_mode != 'detrended':
                    e_min, e_max = env
                    if plot_mode == 'difference' and y_const is not None:
                        e_min, e_max = e_min - y_const, e_max - y_const
                    ax.fill_between(X_CENTERS / 1000, e_min, e_max, alpha=0.25, color='0.55', zorder=1)

                for line_val, scen_key in groups[panel_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    ax.plot(_get_x(scen_key), _transform(y, scen_key), color=colors[line_val],
                            linewidth=LINE_WIDTH, zorder=3)

                ax.set_title(f'${sym_panel}$ = {_fmt(panel_val)}',
                             fontsize=FONTSIZE_TITLE, fontweight='bold', pad=5)
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=FONTSIZE_TICKS)
                ax.tick_params(labelsize=FONTSIZE_TICKS)
                if ci == 0:
                    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)

            # shared legend – envelope, constant, then line values small → large
            legend_handles = []
            if env is not None and plot_mode != 'detrended':
                legend_handles.append(mpatches.Patch(facecolor='0.55', alpha=0.4,
                                                     label=r'$\pm 2\sigma$ natural variability'))
            legend_handles.append(mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                                                linestyle='--', label='constant (pm1_n0)'))
            for val in all_vals:
                legend_handles.append(mlines.Line2D([], [], color=colors[val], linewidth=LINE_WIDTH,
                                                    label=f'${sym_line}$ = {_fmt(val)}'))
            fig.legend(handles=legend_handles, fontsize=FONTSIZE_TICKS, loc='lower center',
                       ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.12), frameon=True)
            fig.suptitle(
                f'Effect of ${sym_line}$ on {title_metric}{norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99, color=_tc,
            )
            fig.subplots_adjust(
                left=_LEFT / fig_w, right=1 - _RIGHT / fig_w,
                bottom=_BOT / fig_h, top=1 - _TOP / fig_h,
                wspace=_WSPACE / AX_W,
            )

            fname = f'thalweg_{vary}_effect{norm_tag}_{_noisy_tag}_{snap_label}_{STYLE}_Q{DISCHARGE}_{metric_tag}.png'
            fig.savefig(out_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
            if is_last_snapshot:
                fig.savefig(out_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')


#%% --- POOLED PER-COLUMN THALWEG DISTRIBUTIONS (robust comparison of selected scenarios) ---
# Cross-section by cross-section comparison is dominated by pool/crossover phase differences between runs.
# Pooling the per-column thalweg over several pool-crossover cycles compares the distributions instead.
# Columns within a bin are spatially correlated: report medians and p10/p90, not p-values.
POOL_SNAPSHOT_KEY = list(comparison_results.keys())[-1]
POOL_X_RANGE_KM   = (35, 44)
POOL_SCENARIOS    = ['pm3_n5', 'pm5_n1']   # labels from SCENARIO_LABELS; None → all non-constant scenarios
POOL_DETRENDED    = True                   # subtract each bin's initial thalweg (removes the bed slope)

pool_groups = group_snapshot_by_scenario(comparison_results[POOL_SNAPSHOT_KEY])
pool_labels = {k: _scenario_label(k, SCENARIO_LABELS) for k in pool_groups}
if POOL_SCENARIOS is None:
    POOL_SCENARIOS = [lab for lab in pool_labels.values() if 'constant' not in lab]
bins_sel = np.flatnonzero((X_CENTERS / 1000 >= POOL_X_RANGE_KM[0]) & (X_CENTERS / 1000 <= POOL_X_RANGE_KM[1]))

fig, ax = plt.subplots(figsize=(AX_W + 0.9, AX_H + 0.6))
print(f"\nPer-column thalweg, x = {POOL_X_RANGE_KM[0]}-{POOL_X_RANGE_KM[1]} km, "
      f"snapshot {comparison_labels.get(POOL_SNAPSHOT_KEY, POOL_SNAPSHOT_KEY)}"
      f"{' (change from initial)' if POOL_DETRENDED else ''}:")
for lab in POOL_SCENARIOS:
    scen = next((k for k, v in pool_labels.items() if v == lab), None)
    if scen is None:
        print(f"  {lab}: not found in this snapshot")
        continue
    vals = []
    for fn, _ in pool_groups[scen]:
        init = initial_profiles[fn]['Thalweg']
        for k in bins_sel:
            c = thal_columns[POOL_SNAPSHOT_KEY][fn][k]
            vals.append(c - init[k] if POOL_DETRENDED else c)
    vals = np.concatenate(vals) if vals else np.array([])
    vals = np.sort(vals[~np.isnan(vals)])
    if vals.size == 0:
        continue

    pm, n = _parse_pm_n(lab)
    leg = (f'$R_{{\\mathrm{{peak}}}}$ = {_fmt(pm)}, $n_{{\\mathrm{{peaks}}}}$ = {_fmt(n)}' if pm is not None else lab)
    ax.step(vals, np.arange(1, vals.size + 1) / vals.size, where='post', linewidth=LINE_WIDTH, label=leg)
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    print(f"  {lab:>8}: n = {vals.size:4d} columns,  p10 = {p10:6.2f},  median = {p50:6.2f},  p90 = {p90:6.2f}  [m]")

ax.set_xlabel('per-column thalweg change from initial bed [m]' if POOL_DETRENDED
              else 'per-column thalweg bed level [m]', fontsize=FONTSIZE_TICKS)
ax.set_ylabel('cumulative fraction of columns', fontsize=FONTSIZE_TICKS)
ax.tick_params(labelsize=FONTSIZE_TICKS)
ax.grid(True, alpha=0.22, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=FONTSIZE_TICKS, frameon=False)
fig.savefig(output_dir / f'pooled_thalweg_ecdf_{comparison_labels.get(POOL_SNAPSHOT_KEY, POOL_SNAPSHOT_KEY)}'
                         f'_Q{DISCHARGE}.png', dpi=200, bbox_inches='tight', transparent=_tr)
plt.show()

print("\nDone.")