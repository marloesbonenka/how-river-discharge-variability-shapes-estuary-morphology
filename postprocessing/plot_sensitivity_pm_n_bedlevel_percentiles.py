"""pm/n sensitivity analysis – bed level metric (percentile or mean)

Toggle COMPUTE_MEAN to switch between:
  False (default) → p{depth_percentile} bed level within frozen t=0 channel mask
  True            → width-averaged mean bed level within frozen t=0 channel mask

Layout: one subplot per fixed parameter (n or pm), lines per varying parameter.
Colors follow the same PALETTE as plot_scenario_lines.py.
Two figure sets per snapshot:
  A) Effect of R_peak  – one panel per n_peaks  (colours = R_peak values)
  B) Effect of n_peaks – one panel per R_peak   (colours = n_peaks values)
Both sets are also saved as normalised / detrended versions.
"""

#%% IMPORTS
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
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
)

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

#%% --- CONFIGURATION ---
DISCHARGE = 1000
depth_percentile = 95

SHOW_NOISY_ENVELOPE = True
SHOW_DIFFERENCE = True   # show difference-from-constant plot
SHOW_DETRENDED  = True   # show detrended plot (change relative to initial bed level)
COMPUTE_MEAN = False  # True → width-averaged mean; False → p{depth_percentile} within frozen channel mask

SNAPSHOT_TARGET_DATES = None
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))
SNAPSHOT_COUNT = 6
#%% --- CONSTANTS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f'Model_Output/Q{DISCHARGE}'

bed_threshold = 6
CHANNEL_INIT_THRESHOLD = 2.2  # defines the channel footprint from t=0
channel_masks = {}  # {folder_str: {bin_idx: boolean array}}

start_date = np.datetime64('2025-01-01')
x_targets = np.arange(20000, 44001, 1000)
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
#%% --- SCENARIO LABELS ---


# Constant scenario colour
GREY_CONST = "#7f7f7f"

# Fixed axes dimensions — same across all figure variants so subplots align
AX_W, AX_H = 3.5, 3.0   # axes width / height in inches (not panel/figure size)
# Margins in inches (space outside the axes area):
_LEFT   = 0.95  # left:   y-label + ticks
_RIGHT  = 0.20  # right:  small buffer
_TOP    = 1.30  # top:    panel title + gap + suptitle
_BOT    = 0.65  # bottom: x-label + ticks
_WSPACE = 0.10  # gap between panels in inches (small; sharey=True)

# --- Line width ---
LINE_WIDTH       = 1.8   # width of scenario lines in the plots
LINE_WIDTH_CONST = 1.5   # width of the constant reference line

# --- Font sizes ---
FONTSIZE_TITLE  = 18   # figure suptitle and panel titles
FONTSIZE_LABELS = FONTSIZE_TITLE - 4    # axis labels and legend title
FONTSIZE_TICKS  = FONTSIZE_LABELS - 2    # tick labels and legend text


#%% --- FIGURE STYLE ---
STYLE = 'default'   # 'default'   →  white background, black text
                    # 'whitefig'  →  transparent figure, white axes background, white text

STYLES = {
    'default': {},
    'whitefig': {
        'figure.facecolor':    'none',
        'axes.facecolor':      'white',
        'axes.edgecolor':      'white',
        'axes.labelcolor':     'white',
        'xtick.color':         'white',
        'ytick.color':         'white',
        'text.color':          'white',
        'grid.color':          '#cccccc',
        'legend.facecolor':    'none',
        'legend.edgecolor':    'white',
        'savefig.transparent': False,
    },
}

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
_tc = plt.rcParams['text.color']                        # convenience: text/title color
_tr = plt.rcParams.get('savefig.transparent', False)    # convenience: transparent flag for savefig


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
comparison_results = {}
comparison_labels  = {}
initial_profiles   = {}  # {folder_str: 1-D array of p{depth_percentile} at t=0}

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

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=['mesh2d_mor_bl'],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
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
    dx = 1000
    x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2

    # Initial (t=0) profile — builds frozen channel mask for all plot modes
    if folder_str not in initial_profiles:
        _init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
        _valid_init = width_mask & (_init_bl < CHANNEL_INIT_THRESHOLD)
        _init_percs = []
        channel_masks[folder_str] = {}

        for _k in range(len(x_bins) - 1):
            _bm = _valid_init & (face_x >= x_bins[_k]) & (face_x < x_bins[_k + 1])
            channel_masks[folder_str][_k] = _bm  # frozen for all timesteps

            if np.any(_bm):
                _vd = _init_bl[_bm]
                _vd = _vd[~np.isnan(_vd)]
                _init_percs.append(
                    np.percentile(_vd, depth_percentile) if len(_vd) > 0 else np.nan
                )
            else:
                _init_percs.append(np.nan)

        initial_profiles[folder_str] = np.array(_init_percs)
        print(f"  Initial profile (t=0) and channel mask computed.")

    for target_dt, ts_idx, actual_dt in snapshot_matches:
            snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
            comparison_results.setdefault(snapshot_key, {})
            comparison_labels[snapshot_key] = _date_to_label(target_dt)

            bedlev_data = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

            bin_metrics = []
            for k in range(len(x_bins) - 1):
                bin_mask = channel_masks[folder_str][k]  # <-- frozen t=0 channel mask
                if np.any(bin_mask):
                    bin_bedlevs = bedlev_data[bin_mask]
                    bin_bedlevs = bin_bedlevs[~np.isnan(bin_bedlevs)]
                    if len(bin_bedlevs) > 0:
                        val = np.mean(bin_bedlevs) if COMPUTE_MEAN else np.percentile(bin_bedlevs, depth_percentile)
                    else:
                        val = np.nan
                    bin_metrics.append(val)
                else:
                    bin_metrics.append(np.nan)

            _metric_key = 'BL' if COMPUTE_MEAN else f'p{depth_percentile} BedLevel'
            comparison_results[snapshot_key][folder_str] = {
                _metric_key: np.array(bin_metrics),
                'x_centers': x_centers,
            }
            _metric_label = 'mean' if COMPUTE_MEAN else f'p{depth_percentile}'
            print(f"  Snapshot {_date_to_label(target_dt)}: computed {_metric_label}.")
    ds.close()


#%% --- LOAD NOISY ENVELOPE DATA ---
noisy_envelope_data = {}  # populated only when SHOW_NOISY_ENVELOPE is True

if SHOW_NOISY_ENVELOPE:
    if not NOISY_BASE_PATH.exists():
        print(f"[WARNING] Noisy base path not found: {NOISY_BASE_PATH}")
    else:
        _dx = 1000
        _x_bins = np.arange(x_targets[0], x_targets[-1] + _dx, _dx)
        _x_centers = (_x_bins[:-1] + _x_bins[1:]) / 2

        _noisy_cache_dir = NOISY_BASE_PATH / 'cached_data'
        _noisy_cache_dir.mkdir(parents=True, exist_ok=True)

        _noisy_profiles      = {}   # {snapshot_key: [1-D array per run]}
        _noisy_init_profiles = []   # initial (t=0) profiles from noisy runs (for detrending)

        for _subfolder in NOISY_SUBFOLDERS:
            _noisy_folder = NOISY_BASE_PATH / _subfolder
            if not _noisy_folder.exists():
                print(f"[WARNING] Noisy subfolder not found: {_noisy_folder}")
                continue
            print(f"Loading noisy run: {_subfolder}")

            _ds_n = load_or_update_map_cache_multi(
                cache_dir=_noisy_cache_dir,
                folder_name=_subfolder,
                run_paths=[_noisy_folder],
                var_names=['mesh2d_mor_bl'],
                bbox=CACHE_BBOX,
                append_time=APPEND_TIMESTEPS,
                append_vars=APPEND_VARIABLES,
                cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
            )
            if _ds_n is None:
                print(f"  No cached data for {_subfolder}, skipping.")
                continue

            _snaps_n = get_snapshot_matches_by_target_dates(
                _ds_n.time.values, target_snapshot_dates
            )
            _fx_n, _fy_n = _get_face_coords(_ds_n)
            _wmask_n = (_fy_n >= y_range[0]) & (_fy_n <= y_range[1])

            # Build frozen t=0 channel mask using CHANNEL_INIT_THRESHOLD
            _init_bl_n = _ds_n['mesh2d_mor_bl'].isel(time=0).values.copy()
            _valid_init_n = _wmask_n & (_init_bl_n < CHANNEL_INIT_THRESHOLD)
            _noisy_channel_masks = {}
            for _ki in range(len(_x_bins) - 1):
                _bmi = _valid_init_n & (_fx_n >= _x_bins[_ki]) & (_fx_n < _x_bins[_ki + 1])
                _noisy_channel_masks[_ki] = _bmi

            # Initial (t=0) profile for detrended normalization
            if SHOW_DETRENDED:
                _init_percs_n = []
                for _ki in range(len(_x_bins) - 1):
                    _bmi = _noisy_channel_masks[_ki]
                    if np.any(_bmi):
                        _vdi = _init_bl_n[_bmi]
                        _vdi = _vdi[~np.isnan(_vdi)]
                        _init_percs_n.append(
                            (np.mean(_vdi) if COMPUTE_MEAN else np.percentile(_vdi, depth_percentile))
                            if len(_vdi) > 0 else np.nan
                        )
                    else:
                        _init_percs_n.append(np.nan)
                _noisy_init_profiles.append(np.array(_init_percs_n))

            for _tdt, _ts_idx, _adt in _snaps_n:
                _snap_key = f"d{_date_to_filename_tag(_tdt)}"
                _bl = _ds_n['mesh2d_mor_bl'].isel(time=_ts_idx).values.copy()

                _p_bedlevs = []
                for _k in range(len(_x_bins) - 1):
                    _bm = _noisy_channel_masks[_k]  # <-- frozen t=0 channel mask
                    if np.any(_bm):
                        _vd = _bl[_bm]
                        _vd = _vd[~np.isnan(_vd)]
                        _p_bedlevs.append(
                            (np.mean(_vd) if COMPUTE_MEAN else np.percentile(_vd, depth_percentile))
                            if len(_vd) > 0 else np.nan
                        )
                    else:
                        _p_bedlevs.append(np.nan)

                _noisy_profiles.setdefault(_snap_key, []).append(np.array(_p_bedlevs))
                print(f"  {_subfolder}: snapshot {_date_to_label(_tdt)} OK")

            _ds_n.close()

        for _snap_key, _profs in _noisy_profiles.items():
            if _profs:
                noisy_envelope_data[_snap_key] = {
                    'profiles': [_p for _p in _profs],
                    'x_km':     _x_centers / 1000,
                }
                if SHOW_DETRENDED and _noisy_init_profiles:
                    noisy_envelope_data[_snap_key]['initial_profile'] = np.nanmean(
                        np.vstack(_noisy_init_profiles), axis=0
                    )
        print(f"Noisy envelope ready for {len(noisy_envelope_data)} snapshots.")


#%% --- SENSITIVITY PLOTS ---
for snapshot_key, snapshot_results in comparison_results.items():
    if not snapshot_results:
        continue

    is_last_snapshot = (snapshot_key == list(comparison_results.keys())[-1])

    scenario_groups = group_snapshot_by_scenario(snapshot_results)
    all_scen_keys = sort_scenario_keys(scenario_groups.keys())
    snap_label = comparison_labels.get(snapshot_key, snapshot_key)

    # Parse pm, n for each scenario
    scen_pm_n = {}
    for scen_key in all_scen_keys:
        label = _scenario_label(scen_key, SCENARIO_LABELS)
        pm, n = _parse_pm_n(label)
        if pm is not None:
            scen_pm_n[scen_key] = (pm, n)

    baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

    # Build groupings
    pm_by_n = {}   # {n_val: [(pm_val, scen_key), ...]}
    n_by_pm = {}   # {pm_val: [(n_val, scen_key), ...]}
    for scen_key, (pm, n) in scen_pm_n.items():
        if n == 0:
            continue
        pm_by_n.setdefault(n, []).append((pm, scen_key))
        n_by_pm.setdefault(pm, []).append((n, scen_key))
    for n in pm_by_n:
        pm_by_n[n].sort()
    for pm in n_by_pm:
        n_by_pm[pm].sort()

    all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
    all_n_vals  = sorted({n  for pm, n in scen_pm_n.values() if n > 0})

    # Colormaps: Blues for pm (light→dark), Greens for n (light→dark)
    _n_pm = max(len(all_pm_vals) - 1, 1)
    PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}
    _n_n = max(len(all_n_vals) - 1, 1)
    N_COLOR  = {n:  plt.cm.Greens(0.35 + 0.55 * i / _n_n) for i, n  in enumerate(all_n_vals)}

    def _get_y(scen_key):
        """Mean metric bed level across runs (raw bed elevation, negative = deeper)."""
        _mk = 'BL' if COMPUTE_MEAN else f'p{depth_percentile} BedLevel'
        y_stack = stack_metric_arrays(scenario_groups[scen_key], _mk)
        if y_stack is None:
            return None
        return np.nanmean(y_stack, axis=0)

    def _get_x(scen_key):
        x_data = next((d for _, d in scenario_groups[scen_key] if 'x_centers' in d), None)
        return x_data['x_centers'] / 1000 if x_data else x_targets / 1000

    def _get_initial_profile(scen_key):
        """Mean initial (t=0) p{depth_percentile} profile across runs in a scenario."""
        profs = [initial_profiles[fn] for fn, _ in scenario_groups[scen_key]
                 if fn in initial_profiles]
        if not profs:
            return None
        return np.nanmean(np.vstack(profs), axis=0)

    y_const = _get_y(baseline_scen) if baseline_scen else None
    x_const = _get_x(baseline_scen) if baseline_scen else None

    # Precompute detrended references (needed before envelope computation)
    y_init_const = _get_initial_profile(baseline_scen) if baseline_scen else None
    y_const_det  = (y_const - y_init_const
                    if y_const is not None and y_init_const is not None else None)
    _noisy_init  = (noisy_envelope_data[snapshot_key].get('initial_profile')
                    if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data else None)

    # Finalise ±2σ envelope: noisy repeats + constant run
    if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data:
        _nd = noisy_envelope_data[snapshot_key]
        if 'profiles' in _nd:
            # Absolute envelope (used for absolute and difference plot modes)
            _all_profs = list(_nd['profiles'])
            if y_const is not None:
                _all_profs.append(y_const)
            _stk = np.vstack(_all_profs)
            _m   = np.nanmean(_stk, axis=0)
            _s   = np.nanstd(_stk, axis=0)
            _nd['env_min'] = _m - 2 * _s
            _nd['env_max'] = _m + 2 * _s

            # Detrended envelope: built in detrended space so that y_const_det
            # (detrended by its own initial profile) is correctly encompassed,
            # even when the noisy model and sensitivity model have different
            # initial bed levels.
            _all_det = []
            for _p in _nd['profiles']:
                _all_det.append(_p - _noisy_init if _noisy_init is not None else _p)
            if y_const_det is not None:
                _all_det.append(y_const_det)
            if _all_det:
                _det_stk = np.vstack(_all_det)
                _det_m = np.nanmean(_det_stk, axis=0)
                _det_s = np.nanstd(_det_stk, axis=0)
                _nd['env_min_det'] = _det_m - 2 * _det_s
                _nd['env_max_det'] = _det_m + 2 * _det_s

    _plot_modes = ['absolute']
    if SHOW_DIFFERENCE:
        _plot_modes.append('difference')
    if SHOW_DETRENDED:
        _plot_modes.append('detrended')

    for plot_mode in _plot_modes:
        normalise = (plot_mode == 'difference')
        detrended = (plot_mode == 'detrended')

        _metric_desc = 'mean' if COMPUTE_MEAN else f'p{depth_percentile}'
        if plot_mode == 'absolute':
            norm_tag   = ''
            norm_title = ''
            ylabel     = f'{_metric_desc} bed level [m]'
            _out_dir   = sensitivity_output_dir
        elif plot_mode == 'difference':
            norm_tag   = '_difference'
            norm_title = ' (difference from constant)'
            ylabel     = f'{_metric_desc} bed level \n(difference from constant)  [m]'
            _out_dir   = sensitivity_output_dir / 'difference_from_Qconstant'
        else:  # detrended
            norm_tag   = '_detrended'
            norm_title = ' (change from initial bed)'
            ylabel     = f'{_metric_desc} bed level \n(change from initial bed)  [m]'
            _out_dir   = sensitivity_output_dir / 'detrended'
        _out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Figure A: pm-effect, one panel per n ----
        sorted_n_vals = sorted(pm_by_n.keys())
        if sorted_n_vals:
            n_panels = len(sorted_n_vals)
            _fig_w = _LEFT + n_panels * AX_W + (n_panels - 1) * _WSPACE + _RIGHT
            _fig_h = _BOT + AX_H + _TOP
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(_fig_w, _fig_h),
                sharey=True, sharex=False,
            )
            if n_panels == 1:
                axes = [axes]

            for ci, n_val in enumerate(sorted_n_vals):
                ax = axes[ci]

                # Grey dashed constant reference
                if not normalise and not detrended and y_const is not None:
                    ax.plot(x_const, y_const, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)
                if normalise:
                    ax.axhline(0.0, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                               label='constant (pm1_n0)', zorder=2)
                if detrended and y_const_det is not None:
                    ax.plot(x_const, y_const_det, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)

                # Natural variability envelope (not shown in detrended mode)
                if SHOW_NOISY_ENVELOPE and not detrended and snapshot_key in noisy_envelope_data:
                    _env = noisy_envelope_data[snapshot_key]
                    _emin = _env['env_min'].copy()
                    _emax = _env['env_max'].copy()
                    if normalise and y_const is not None:
                        _emin = _emin - y_const
                        _emax = _emax - y_const
                    ax.fill_between(
                        _env['x_km'], _emin, _emax,
                        alpha=0.25, color='0.55', zorder=1,
                        label=r'$\pm 2\sigma$ natural variability',
                    )

                for pm_val, scen_key in pm_by_n[n_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    if normalise and y_const is not None:
                        y = y - y_const
                    elif detrended:
                        _y_init = _get_initial_profile(scen_key)
                        if _y_init is not None:
                            y = y - _y_init
                    x = _get_x(scen_key)
                    pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                    ax.plot(x, y, color=PM_COLOR[pm_val], linewidth=LINE_WIDTH,
                            label=f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}', zorder=3)

                ax.set_title(
                    f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}',
                    fontsize=FONTSIZE_TITLE, fontweight='bold', pad=5,
                )
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=FONTSIZE_TICKS)
                ax.tick_params(labelsize=FONTSIZE_TICKS)
                if ci == 0:
                    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)
                    ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)

            # Shared legend – constant first, then pm values sorted small→large
            legend_handles = []
            if SHOW_NOISY_ENVELOPE and not detrended and snapshot_key in noisy_envelope_data:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor='0.55', alpha=0.4,
                        label=r'$\pm 2\sigma$ natural variability',
                    )
                )
            legend_handles.append(
                mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                              label='constant (pm1_n0)')
            )
            for pm_val in sorted(all_pm_vals):
                pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                legend_handles.append(
                    mlines.Line2D([], [], color=PM_COLOR[pm_val], linewidth=LINE_WIDTH,
                                  linestyle='-', label=f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}')
                )
            fig.legend(
                handles=legend_handles,
                title_fontsize=FONTSIZE_LABELS, fontsize=FONTSIZE_TICKS, loc='lower center',
                ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.1), frameon=True,
            )
            fig.suptitle(
                f'Effect of $R_{{\\mathrm{{peak}}}}$ on {_metric_desc} bed level {norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99, color=_tc,
            )
            fig.subplots_adjust(
                left=_LEFT / _fig_w,
                right=1 - _RIGHT / _fig_w,
                bottom=_BOT / _fig_h,
                top=1 - _TOP / _fig_h,
                wspace=_WSPACE / AX_W,
            )
            _noisy_tag = 'noisy' if SHOW_NOISY_ENVELOPE else ''
            fname = f'sensitivity_pm_effect{norm_tag}_{_noisy_tag}_{snap_label}_{STYLE}_Q{DISCHARGE}_{_metric_desc}.png'
            fig.savefig(_out_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
            if is_last_snapshot:
                fig.savefig(_out_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

        # ---- Figure B: n-effect, one panel per pm ----
        sorted_pm_vals = sorted(n_by_pm.keys())
        if sorted_pm_vals:
            pm_panels = len(sorted_pm_vals)
            _fig_w = _LEFT + pm_panels * AX_W + (pm_panels - 1) * _WSPACE + _RIGHT
            _fig_h = _BOT + AX_H + _TOP
            fig, axes = plt.subplots(
                1, pm_panels,
                figsize=(_fig_w, _fig_h),
                sharey=True, sharex=False,
            )
            if pm_panels == 1:
                axes = [axes]

            for ci, pm_val in enumerate(sorted_pm_vals):
                ax = axes[ci]

                if not normalise and not detrended and y_const is not None:
                    ax.plot(x_const, y_const, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)
                if normalise:
                    ax.axhline(0.0, color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                               label='constant (pm1_n0)', zorder=2)
                if detrended and y_const_det is not None:
                    ax.plot(x_const, y_const_det, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)

                # Natural variability envelope (not shown in detrended mode)
                if SHOW_NOISY_ENVELOPE and not detrended and snapshot_key in noisy_envelope_data:
                    _env = noisy_envelope_data[snapshot_key]
                    _emin = _env['env_min'].copy()
                    _emax = _env['env_max'].copy()
                    if normalise and y_const is not None:
                        _emin = _emin - y_const
                        _emax = _emax - y_const
                    ax.fill_between(
                        _env['x_km'], _emin, _emax,
                        alpha=0.25, color='0.55', zorder=1,
                        label=r'$\pm 2\sigma$ natural variability',
                    )

                for n_val, scen_key in n_by_pm[pm_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    if normalise and y_const is not None:
                        y = y - y_const
                    elif detrended:
                        _y_init = _get_initial_profile(scen_key)
                        if _y_init is not None:
                            y = y - _y_init
                    x = _get_x(scen_key)
                    ax.plot(x, y, color=N_COLOR[n_val], linewidth=LINE_WIDTH,
                            label=f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}', zorder=3)

                pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                ax.set_title(
                    f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}',
                    fontsize=FONTSIZE_TITLE, fontweight='bold', pad=5,
                )
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=FONTSIZE_TICKS)
                ax.tick_params(labelsize=FONTSIZE_TICKS)
                if ci == 0:
                    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)
                    ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)

            # Shared legend – constant first, then n values sorted small→large
            legend_handles = []
            if SHOW_NOISY_ENVELOPE and not detrended and snapshot_key in noisy_envelope_data:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor='0.55', alpha=0.4,
                        label=r'$\pm 2\sigma$ natural variability',
                    )
                )
            legend_handles.append(
                mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                              label='constant (pm1_n0)')
            )
            for n_val in sorted(all_n_vals):
                legend_handles.append(
                    mlines.Line2D([], [], color=N_COLOR[n_val], linewidth=LINE_WIDTH,
                                  linestyle='-', label=f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}')
                )
            fig.legend(
                handles=legend_handles, title='Number of peaks',
                title_fontsize=FONTSIZE_LABELS, fontsize=FONTSIZE_TICKS, loc='lower center',
                ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.18), frameon=True,
            )
            fig.suptitle(
                f'Effect of $n_{{\\mathrm{{peaks}}}}$ on {_metric_desc} bed level {norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99, color=_tc,
            )
            fig.subplots_adjust(
                left=_LEFT / _fig_w,
                right=1 - _RIGHT / _fig_w,
                bottom=_BOT / _fig_h,
                top=1 - _TOP / _fig_h,
                wspace=_WSPACE / AX_W,
            )
            _noisy_tag = 'noisy' if SHOW_NOISY_ENVELOPE else ''
            fname = f'sensitivity_n_effect{norm_tag}_{_noisy_tag}_{snap_label}_{STYLE}_Q{DISCHARGE}_{_metric_desc}.png'
            fig.savefig(_out_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
            if is_last_snapshot:
                fig.savefig(_out_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

print("\nDone.")
