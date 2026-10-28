"""pm/n sensitivity analysis – p5 / p95 (or mean) of the raw bed level per 1 km x-bin

Diagnostic version: all snapshots, both figure sets, absolute / difference /
detrended. The paper figure (single snapshot, p95 + p5 rows) is made by
fig_bedlevel_percentiles_pm_n.py.

Each metric has its own face population (MASK_MODE), all built from the raw bed:
  'frozen'   faces below CHANNEL_INIT_THRESHOLD at t=0 (the original method; stable,
             but blind to channel widening)
  'union'    faces below CHANNEL_INIT_THRESHOLD at t=0 OR at the snapshot (adds
             widening; same 2.2 m cut-off, so bank slopes barely enter)
  'envelope' faces below water level(x) + WET_MARGIN_M at t=0 OR at the snapshot,
             plus dry stretches narrower than MAX_BAR_WIDTH_M bridged by wet faces
             (keeps mid-channel bars above the limit; needs the water-level table)
See functions/F_bedlevel_percentiles.py. Default: 'union' for both metrics.
Detrended mode uses the same population at t=0 as the reference.

Layout: one subplot per fixed parameter (n or pm), lines per varying parameter.
Two figure sets per snapshot and metric:
  A) Effect of R_peak  – one panel per n_peaks  (colours = R_peak values)
  B) Effect of n_peaks – one panel per R_peak   (colours = n_peaks values)
Each is saved as absolute, difference-from-constant and detrended versions.
"""

#%% IMPORTS
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

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
from functions.F_bedlevel_percentiles import bedlevel_profiles, wet_limits

#%% --- CONFIGURATION ---
DISCHARGE = 500
METRICS = ['p5', 'p95']   # any of 'pNN' (percentile) or 'mean'
MASK_MODE = {'p5': 'union', 'p95': 'union'}   # per metric: 'frozen' | 'union' | 'envelope'
DEFAULT_MASK_MODE = 'union'                    # for metrics not listed above

SHOW_NOISY_ENVELOPE = True
SHOW_DIFFERENCE = False    # difference-from-constant plots
SHOW_DETRENDED  = False    # detrended plots (change relative to the t=0 estuary)
SHOW_FIGURES    = False    # plt.show() per figure; False → only save

SNAPSHOT_TARGET_DATES = None
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))
SNAPSHOT_COUNT = 6

#%% --- CONSTANTS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f'Model_Output/Q{DISCHARGE}'

# Channel populations (see module docstring)
CHANNEL_INIT_THRESHOLD = 2.2   # 'frozen' / 'union': channel = bed below this [m]
WATER_LEVEL_KM_M = [(20, 2.1), (35, 2.7), (44, 4.3)]   # (x [km], water level [m]) at Q=500,
                                                       # linearly interpolated; refine if needed
WET_MARGIN_M = 1.0        # wet = bed < water level + margin (includes bars just above water)
MAX_BAR_WIDTH_M = 400.0   # dry stretches narrower than this, between wet faces, are bars (kept)

x_targets = np.arange(20000, 44001, 1000)
X_BINS = np.arange(x_targets[0], x_targets[-1] + 1000, 1000)
X_CENTERS = (X_BINS[:-1] + X_BINS[1:]) / 2
X_KM = X_CENTERS / 1000
y_range = (5000, 10000)

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

if DISCHARGE == 500:
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

elif DISCHARGE == 1000:
    NOISY_BASE_PATH = Path(r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q1000\0_Noise_Q1000")
    NOISY_SUBFOLDERS = []
    SCENARIO_LABELS = {
        '1':  'pm1_n0 (constant)',
        # '2':  'pm2_n1',
        # '3':  'pm3_n5',
        # '4':  'pm3_n1',
        # '5':  'pm5_n1',
        '6':  'pm4_n3',
        # '7':  'pm3_n4',
        # '8':  'pm2_n6',
        '9':  'pm5_n3',
        '10': 'pm3_n3',
        '11': 'pm2_n3',
        # '12': 'pm5_n4',
        # '13': 'pm4_n4',
        # '14': 'pm2_n4',
    }

elif DISCHARGE == 250:
    NOISY_BASE_PATH = Path(r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q250\0_Noise_Q250")
    NOISY_SUBFOLDERS = []
    SCENARIO_LABELS = {
        '1':  'pm1_n0 (constant)',
        # '2':  'pm2_n1',
        # '3':  'pm3_n5',
        # '4':  'pm3_n1',
        # '5':  'pm5_n1',
        '6':  'pm4_n3',
        # '7':  'pm3_n4',
        # '8':  'pm2_n6',
        '9':  'pm5_n3',
        '10': 'pm3_n3',
        '11': 'pm2_n3',
        # '12': 'pm5_n4',
        # '13': 'pm4_n4',
        # '14': 'pm2_n4',
    }

#%% --- PLOT SETTINGS ---
GREY_CONST = "#7f7f7f"   # constant scenario colour
RPEAK  = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'

# Fixed axes dimensions — same across all figure variants so subplots align
AX_W, AX_H = 3.5, 3.0   # axes width / height in inches (not panel/figure size)
# Margins in inches (space outside the axes area):
_LEFT   = 0.95  # left:   y-label + ticks
_RIGHT  = 0.20  # right:  small buffer
_TOP    = 1.30  # top:    panel title + gap + suptitle
_BOT    = 0.65  # bottom: x-label + ticks
_WSPACE = 0.10  # gap between panels in inches (small; sharey=True)

LINE_WIDTH       = 1.8   # width of scenario lines
LINE_WIDTH_CONST = 1.5   # width of the constant reference line

FONTSIZE_TITLE  = 18                     # figure suptitle and panel titles
FONTSIZE_LABELS = FONTSIZE_TITLE - 4     # axis labels and legend title
FONTSIZE_TICKS  = FONTSIZE_LABELS - 2    # tick labels and legend text

STYLE = 'default'   # 'default'   →  white background, black text
                    # 'whitefig'  →  transparent figure, white axes background, white text

if STYLE == 'whitefig':
    apply_plot_style('whitefig', **{
        'axes.facecolor': 'white',
        'grid.color': '#cccccc',
        'savefig.transparent': False,
    })
else:
    apply_plot_style('default')
_tc = plt.rcParams['text.color']                        # text/title colour
_tr = plt.rcParams.get('savefig.transparent', False)    # transparent flag for savefig

#%% --- HELPERS ---
MODE_OF = {m: MASK_MODE.get(m, DEFAULT_MASK_MODE) for m in METRICS}
WET_LIMIT = wet_limits(X_BINS, WATER_LEVEL_KM_M, WET_MARGIN_M)   # only used by 'envelope'


def _profiles(bed_now, init_bl, face_x, face_y, width_mask):
    """{metric: value per x-bin} using each metric's own population, plus face counts per bin
    (counts refer to the first metric's population)."""
    return bedlevel_profiles(bed_now, init_bl, face_x, face_y, width_mask, X_BINS, METRICS, MODE_OF,
                             threshold=CHANNEL_INIT_THRESHOLD, wet_limit=WET_LIMIT,
                             max_bar_width=MAX_BAR_WIDTH_M)


def _fmt(v):
    return str(int(v)) if v == int(v) else str(v)

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
sensitivity_output_dir = base_path / 'output_plots' / 'plots_pm_n_sensitivity'
sensitivity_output_dir.mkdir(parents=True, exist_ok=True)

#%% --- LOAD DATA ---
comparison_results = {}   # {snapshot_key: {folder: {'<metric> BedLevel': array, ...}}}
comparison_labels  = {}
initial_profiles   = {}   # {folder: {metric: array}} on the t=0 estuary

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

    snapshot_matches = get_snapshot_matches_by_target_dates(ds.time.values, target_snapshot_dates)
    if not snapshot_matches:
        print(f"  No timesteps found for {folder_str}, skipping.")
        ds.close()
        continue

    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])

    # t=0: reference (detrending) profile
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
    initial_profiles[folder_str], _ = _profiles(init_bl, init_bl, face_x, face_y, width_mask)
    print("  Population per metric: " + ", ".join(f"{m}={mode}" for m, mode in MODE_OF.items()))

    for target_dt, ts_idx, actual_dt in snapshot_matches:
        snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
        comparison_results.setdefault(snapshot_key, {})
        comparison_labels[snapshot_key] = _date_to_label(target_dt)

        bedlev_data = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
        profs, counts = _profiles(bedlev_data, init_bl, face_x, face_y, width_mask)

        comparison_results[snapshot_key][folder_str] = {
            **{f'{m} BedLevel': profs[m] for m in METRICS},
            'x_centers': X_CENTERS,
        }
        print(f"  Snapshot {_date_to_label(target_dt)}: computed {', '.join(METRICS)} "
              f"({sum(counts)} estuary faces, fewest in a bin: {min(counts)}).")
    ds.close()

#%% --- LOAD NOISY ENVELOPE DATA ---
noisy_envelope_data = {}   # {metric: {snapshot_key: {'profiles': [array per run]}}}

if SHOW_NOISY_ENVELOPE:
    if not NOISY_BASE_PATH.exists():
        print(f"[WARNING] Noisy base path not found: {NOISY_BASE_PATH}")
    else:
        noisy_cache_dir = NOISY_BASE_PATH / 'cached_data'
        noisy_cache_dir.mkdir(parents=True, exist_ok=True)
        noisy_profiles = {}   # {snapshot_key: [{metric: array} per run]}

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

            snaps_n = get_snapshot_matches_by_target_dates(ds_n.time.values, target_snapshot_dates)
            fx_n, fy_n = _get_face_coords(ds_n)
            wmask_n = (fy_n >= y_range[0]) & (fy_n <= y_range[1])
            init_bl_n = ds_n['mesh2d_mor_bl'].isel(time=0).values.copy()

            for tdt, ts_idx, adt in snaps_n:
                snap_key = f"d{_date_to_filename_tag(tdt)}"
                bl = ds_n['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
                profs, _ = _profiles(bl, init_bl_n, fx_n, fy_n, wmask_n)
                noisy_profiles.setdefault(snap_key, []).append(profs)
                print(f"  {subfolder}: snapshot {_date_to_label(tdt)} OK")

            ds_n.close()

        for snap_key, runs in noisy_profiles.items():
            for m in METRICS:
                noisy_envelope_data.setdefault(m, {})[snap_key] = {'profiles': [r[m] for r in runs]}
        print(f"Noisy envelope ready for {len(noisy_profiles)} snapshots.")

#%% --- SENSITIVITY PLOTS ---
def plot_snapshot(snapshot_key, snapshot_results, metric):
    """All figures (A: R_peak effect, B: n_peaks effect; absolute / difference / detrended)
    for one snapshot and one metric."""
    if not snapshot_results:
        return

    is_last_snapshot = (snapshot_key == list(comparison_results.keys())[-1])
    metric_key = f'{metric} BedLevel'
    noisy_env = noisy_envelope_data.get(metric, {})
    has_noisy = SHOW_NOISY_ENVELOPE and snapshot_key in noisy_env

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

    # Groupings
    pm_by_n = {}   # {n_val: [(pm_val, scen_key), ...]}
    n_by_pm = {}   # {pm_val: [(n_val, scen_key), ...]}
    for scen_key, (pm, n) in scen_pm_n.items():
        if n == 0:
            continue
        pm_by_n.setdefault(n, []).append((pm, scen_key))
        n_by_pm.setdefault(pm, []).append((n, scen_key))
    for group in (pm_by_n, n_by_pm):
        for key in group:
            group[key].sort()

    all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
    all_n_vals  = sorted({n  for pm, n in scen_pm_n.values() if n > 0})

    # Colormaps: Blues for pm (light→dark), Greens for n (light→dark)
    PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / max(len(all_pm_vals) - 1, 1))
                for i, pm in enumerate(all_pm_vals)}
    N_COLOR  = {n: plt.cm.Greens(0.35 + 0.55 * i / max(len(all_n_vals) - 1, 1))
                for i, n in enumerate(all_n_vals)}

    def _get_y(scen_key):
        """Mean metric bed level across runs (raw bed elevation, negative = deeper)."""
        y_stack = stack_metric_arrays(scenario_groups[scen_key], metric_key)
        return None if y_stack is None else np.nanmean(y_stack, axis=0)

    def _get_initial_profile(scen_key):
        """Mean t=0 estuary metric profile across the runs of a scenario."""
        profs = [initial_profiles[fn][metric] for fn, _ in scenario_groups[scen_key]
                 if fn in initial_profiles]
        return np.nanmean(np.vstack(profs), axis=0) if profs else None

    y_const = _get_y(baseline_scen) if baseline_scen else None
    y_init_const = _get_initial_profile(baseline_scen) if baseline_scen else None
    y_const_det = (y_const - y_init_const
                   if y_const is not None and y_init_const is not None else None)

    # ±2σ natural-variability envelope: noisy repeats + the constant run
    if has_noisy:
        nd = noisy_env[snapshot_key]
        stk = np.vstack(list(nd['profiles']) + ([y_const] if y_const is not None else []))
        mean, std = np.nanmean(stk, axis=0), np.nanstd(stk, axis=0)
        nd['env_min'], nd['env_max'] = mean - 2 * std, mean + 2 * std

    plot_modes = ['absolute']
    if SHOW_DIFFERENCE:
        plot_modes.append('difference')
    if SHOW_DETRENDED:
        plot_modes.append('detrended')

    for plot_mode in plot_modes:
        normalise = (plot_mode == 'difference')
        detrended = (plot_mode == 'detrended')
        show_env = has_noisy and not detrended   # envelope is not shown in detrended mode

        if plot_mode == 'absolute':
            norm_tag, norm_title = '', ''
            ylabel   = f'{metric} bed level [m]'
            out_dir  = sensitivity_output_dir
        elif plot_mode == 'difference':
            norm_tag, norm_title = '_difference', ' (difference from constant)'
            ylabel   = f'{metric} bed level \n(difference from constant)  [m]'
            out_dir  = sensitivity_output_dir / 'difference_from_Qconstant'
        else:
            norm_tag, norm_title = '_detrended', ' (change from initial bed)'
            ylabel   = f'{metric} bed level \n(change from initial bed)  [m]'
            out_dir  = sensitivity_output_dir / 'detrended'
        out_dir.mkdir(parents=True, exist_ok=True)

        def _series(scen_key):
            """Line to plot for a scenario in the current mode."""
            y = _get_y(scen_key)
            if y is None:
                return None
            if normalise and y_const is not None:
                return y - y_const
            if detrended:
                y_init = _get_initial_profile(scen_key)
                if y_init is not None:
                    return y - y_init
            return y

        def _draw_reference(ax):
            """Constant reference line and natural-variability envelope."""
            if normalise:
                ax.axhline(0.0, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                           label='constant (pm1_n0)', zorder=2)
            else:
                y_ref = y_const_det if detrended else y_const
                if y_ref is not None:
                    ax.plot(X_KM, y_ref, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)
            if show_env:
                emin, emax = noisy_env[snapshot_key]['env_min'], noisy_env[snapshot_key]['env_max']
                if normalise and y_const is not None:
                    emin, emax = emin - y_const, emax - y_const
                ax.fill_between(X_KM, emin, emax, alpha=0.25, color='0.55', zorder=1,
                                label=r'$\pm 2\sigma$ natural variability')

        def _make_figure(panels, panel_title, color_map, line_label, legend_vals,
                         varied_symbol, kind, legend_title, legend_y):
            """One row of panels. panels: {panel_value: [(line_value, scen_key), ...]}."""
            n_panels = len(panels)
            fig_w = _LEFT + n_panels * AX_W + (n_panels - 1) * _WSPACE + _RIGHT
            fig_h = _BOT + AX_H + _TOP
            fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True, sharex=False)
            if n_panels == 1:
                axes = [axes]

            for ci, panel_val in enumerate(sorted(panels)):
                ax = axes[ci]
                _draw_reference(ax)
                for line_val, scen_key in panels[panel_val]:
                    y = _series(scen_key)
                    if y is None:
                        continue
                    ax.plot(X_KM, y, color=color_map[line_val], linewidth=LINE_WIDTH,
                            label=line_label(line_val), zorder=3)

                ax.set_title(panel_title(panel_val), fontsize=FONTSIZE_TITLE, fontweight='bold', pad=5)
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=FONTSIZE_TICKS)
                ax.tick_params(labelsize=FONTSIZE_TICKS)
                if ci == 0:
                    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)

            # Shared legend – envelope, constant, then the varied parameter small→large
            handles = []
            if show_env:
                handles.append(mpatches.Patch(facecolor='0.55', alpha=0.4,
                                              label=r'$\pm 2\sigma$ natural variability'))
            handles.append(mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                                         linestyle='--', label='constant (pm1_n0)'))
            for v in legend_vals:
                handles.append(mlines.Line2D([], [], color=color_map[v], linewidth=LINE_WIDTH,
                                             linestyle='-', label=line_label(v)))
            fig.legend(handles=handles, title=legend_title, title_fontsize=FONTSIZE_LABELS,
                       fontsize=FONTSIZE_TICKS, loc='lower center', ncol=len(handles),
                       bbox_to_anchor=(0.5, legend_y), frameon=True)
            fig.suptitle(
                f'Effect of {varied_symbol} on {metric} bed level {norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99, color=_tc,
            )
            fig.subplots_adjust(
                left=_LEFT / fig_w, right=1 - _RIGHT / fig_w,
                bottom=_BOT / fig_h, top=1 - _TOP / fig_h,
                wspace=_WSPACE / AX_W,
            )

            noisy_tag = 'noisy' if SHOW_NOISY_ENVELOPE else ''
            fname = f'sensitivity_{kind}_effect{norm_tag}_{noisy_tag}_{snap_label}_{STYLE}_Q{DISCHARGE}_{metric}_{MODE_OF[metric]}.png'
            fig.savefig(out_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
            if is_last_snapshot:
                fig.savefig(out_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
            if SHOW_FIGURES:
                plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

        # ---- Figure A: pm-effect, one panel per n ----
        if pm_by_n:
            _make_figure(
                pm_by_n,
                panel_title=lambda v: f'{NPEAKS} = {_fmt(v)}',
                color_map=PM_COLOR,
                line_label=lambda v: f'{RPEAK} = {_fmt(v)}',
                legend_vals=all_pm_vals,
                varied_symbol=RPEAK, kind='pm', legend_title=None, legend_y=-0.1,
            )

        # ---- Figure B: n-effect, one panel per pm ----
        if n_by_pm:
            _make_figure(
                n_by_pm,
                panel_title=lambda v: f'{RPEAK} = {_fmt(v)}',
                color_map=N_COLOR,
                line_label=lambda v: f'{NPEAKS} = {_fmt(v)}',
                legend_vals=all_n_vals,
                varied_symbol=NPEAKS, kind='n', legend_title='Number of peaks', legend_y=-0.18,
            )


for metric in METRICS:
    for snapshot_key, snapshot_results in comparison_results.items():
        plot_snapshot(snapshot_key, snapshot_results, metric)

print("\nDone.")