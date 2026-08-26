"""Compute and compare three hydrodynamic response metrics across all discharge
variability scenarios (Q = 250 / 500 / 1000 m3/s, runs 01, 06, 09, 10, 11):

  1. Intertidal area [km2]    -- MAP wet/dry classification (last 12 h window)
  2. Max tidal intrusion [km] -- MAP rolling max of cross-sec. mean velocity (last 1 day)
  3. Tidal prism [Mm3]        -- HIS cross-section discharge at mouth (last 1 day)

Each metric is plotted vs discharge peak amplitude (R_peak / pm), coloured by
mean discharge. A vertical dashed line marks the constant (n=0) reference value
for each discharge. Set SHOW_TRENDLINE = True to overlay a linear trend per
discharge group per plot.
"""

# %% Imports
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths, get_stitched_his_paths

# =============================================================================
# %% --- SHARED CONFIGURATION ---
# =============================================================================
NORMALIZE       = False   # normalize each metric to the n=0 constant run per discharge
SHOW_TRENDLINE  = False   # tag: set True to overlay a linear trend line per discharge per plot

BASE_DIR           = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
DISCHARGES         = [250, 500, 1000]
RUN_IDS_TO_INCLUDE = {1, 6, 9, 10, 11}

# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}
_FOLDER_RE = re.compile(r'^dhr_(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\d+$')

# MAP cache settings (shared by intertidal area and tidal intrusion)
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG  = None

# Colour and marker per discharge value
DISCHARGE_COLORS   = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}
DISCHARGE_MARKERS  = {250: '^',       500: 'o',       1000: 's'}       # triangle, circle, square

# Combined output directory
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'hydro_tidalmetrics_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure typography — AGU standards: Calibri, minimum 8 pt
FONTSIZE_TITLE  = 10
FONTSIZE_LABELS = 9
FONTSIZE_TICKS  = 8

plt.rcParams.update(plt.rcParamsDefault)
_tr = plt.rcParams.get('savefig.transparent', False)

# =============================================================================
# --- INTERTIDAL AREA (MAP) ---
# =============================================================================
IA_LOAD_VARS     = ['mesh2d_waterdepth', 'mesh2d_flowelem_ba']
IA_FACE_X_VAR    = 'mesh2d_face_x'
IA_WET_THRESHOLD = 0.0001   # [m]  Epshu in Delft3D-FM
IA_WINDOW_HOURS  = 12.0     # [h]  ~ 1 tidal cycle, taken from end of run
IA_X_MIN         = 20000.0  # [m]  restrict to tidal zone
IA_X_MAX         = 45000.0  # [m]

# =============================================================================
# --- MAX TIDAL INTRUSION (MAP) ---
# =============================================================================
LTI_VAR_NAME            = 'mesh2d_ucx'
LTI_LOAD_VARS           = [LTI_VAR_NAME]
LTI_FLOOD_VEL_THRESHOLD = 0.00001  # [m/s]
LTI_N_DAYS              = 1
LTI_X_BIN_WIDTH         = 100      # [m]
LTI_X_MIN               = 19000    # [m]
LTI_X_MAX               = 45000    # [m]

# =============================================================================
# --- TIDAL PRISM (HIS) ---
# =============================================================================
TP_MOUTH_KEYWORDS    = ['km20', 'mouth']
TP_UPSTREAM_KEYWORDS = ['km44', 'upstream']
TP_VARIABLE          = 'cross_section_discharge'
TP_SOURCE            = 'cross_section'
TP_SUBTRACT_RIVER    = True
TP_NEGATE            = True      # positive flood / negative ebb
TP_WINDOW_DAYS       = 1
TP_PRISM_UNIT        = 'Mm\u00b3'
TP_PRISM_SCALE       = 1e6


# =============================================================================
# %% --- SHARED HELPERS ---
# =============================================================================

def discover_scenario_folders(dhr_base, discharge):
    """Find dhr_XX_Qr{discharge}_pm{pm}_n{n}[_mean].{runid} folders,
    restricted to RUN_IDS_TO_INCLUDE. Returns list of
    (folder_path, run_id, pm_val, n_val)."""
    results = []
    if not dhr_base.exists():
        return results
    for folder in sorted(dhr_base.iterdir()):
        if not folder.is_dir():
            continue
        m = _FOLDER_RE.match(folder.name)
        if not m:
            continue
        run_id  = int(m.group(1))
        q_val   = int(m.group(2))
        pm_val  = int(m.group(3))
        n_val   = int(m.group(4))
        if run_id not in RUN_IDS_TO_INCLUDE or q_val != discharge:
            continue
        results.append((folder, run_id, pm_val, n_val))
    return results


# =============================================================================
# %% --- INTERTIDAL AREA HELPERS ---
# =============================================================================

def get_last_n_hours_window(time_values, n_hours):
    """Boolean mask selecting the last n_hours of a datetime64 array."""
    t_end   = time_values[-1]
    t_start = t_end - np.timedelta64(int(n_hours * 3600), 's')
    return time_values >= t_start


def compute_intertidal_area(ds_window):
    """Wet/dry classification over the time window, restricted to the tidal
    zone in x. Returns (area_m2, n_intertidal_faces, n_faces)."""
    depth_vals  = ds_window['mesh2d_waterdepth'].values   # (n_window, nFaces)
    wet_mask_t  = depth_vals > IA_WET_THRESHOLD

    always_wet  = wet_mask_t.all(axis=0)
    always_dry  = (~wet_mask_t).all(axis=0)
    intertidal  = ~always_wet & ~always_dry

    x_vals = (ds_window.coords[IA_FACE_X_VAR].values
              if IA_FACE_X_VAR in ds_window.coords
              else ds_window[IA_FACE_X_VAR].values)
    in_zone    = (x_vals >= IA_X_MIN) & (x_vals <= IA_X_MAX)
    intertidal = intertidal & in_zone

    ba_da  = ds_window['mesh2d_flowelem_ba']
    ba_vals = ba_da.isel(time=0).values if 'time' in ba_da.dims else ba_da.values

    area_m2 = float(np.nansum(ba_vals[intertidal]))
    return area_m2, int(intertidal.sum()), wet_mask_t.shape[1]


# =============================================================================
# %% --- TIDAL INTRUSION HELPERS ---
# =============================================================================

def get_last_n_days_window_ds(ds, n_days):
    """(t_start, t_end) covering the last n_days of an xarray dataset."""
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    return t_start, t_end


def compute_xprofile_mean(data_t, face_x, var_name, x_edges):
    """Cross-sectional mean of the signed variable at each x-bin (one timestep)."""
    vals    = data_t[var_name].values
    bin_idx = np.digitize(face_x, x_edges) - 1
    n_bins  = len(x_edges) - 1
    profile = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            profile[b] = np.nanmean(vals[mask])
    return profile


def build_ubar_matrix(ds, face_x, var_name, x_edges):
    """Ubar(x, t): cross-sectional mean signed velocity at every x-bin and
    every timestep. Shape: (n_time, n_xbins)."""
    n_time = len(ds.time)
    n_bins = len(x_edges) - 1
    Ubar   = np.full((n_time, n_bins), np.nan)
    for idx in range(n_time):
        data_t      = ds.isel(time=idx)
        Ubar[idx, :] = compute_xprofile_mean(data_t, face_x, var_name, x_edges)
    return Ubar


def front_from_profile(profile_1d, x_centers, threshold):
    """First x where profile drops below threshold moving landward (increasing x)."""
    active = profile_1d > threshold
    if not active[0]:
        return np.nan
    for i in range(len(profile_1d)):
        if not active[i]:
            return x_centers[i]
    return x_centers[-1]


# =============================================================================
# %% --- TIDAL PRISM HELPERS ---
# =============================================================================

def get_last_n_days_window_paths(his_paths, n_days):
    """(t_start, t_end) from the last n_days of the HIS file."""
    ds      = xr.open_mfdataset(his_paths, coords='minimal', compat='override')
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    ds.close()
    return t_start, t_end


def load_his_signal(his_paths, t_start, t_end):
    """Load river-corrected, sign-corrected cross-section discharge at the
    estuary mouth. Mirrors load_his_data() in compute_tidal_prism.py."""
    SOURCE_DIM = {'cross_section': 'cross_section_name', 'station': 'station_name'}
    dim        = SOURCE_DIM[TP_SOURCE]

    ds_his       = xr.open_mfdataset(his_paths, coords='minimal', compat='override')
    ds_his_slice = ds_his.sel(time=slice(t_start, t_end))

    loc_names = [
        name.decode('utf-8').strip() if isinstance(name, bytes) else str(name).strip()
        for name in ds_his_slice[dim].values
    ]

    try:
        idx_mouth = next(i for i, name in enumerate(loc_names)
                         if any(k in name for k in TP_MOUTH_KEYWORDS))
    except StopIteration:
        print('    [WARNING] Mouth keywords not found. Defaulting to index 0.')
        idx_mouth = 0

    idx_river = None
    if TP_SUBTRACT_RIVER and TP_SOURCE == 'cross_section':
        try:
            idx_river = next(i for i, name in enumerate(loc_names)
                             if any(k in name for k in TP_UPSTREAM_KEYWORDS))
        except StopIteration:
            print('    [WARNING] Upstream keywords not found. No river correction applied.')

    time_values  = ds_his_slice.time.values
    time_seconds = (time_values - time_values[0]) / np.timedelta64(1, 's')
    data_mouth   = ds_his_slice[TP_VARIABLE].values[:, idx_mouth]
    sign         = -1 if TP_NEGATE else 1

    if TP_SUBTRACT_RIVER and idx_river is not None:
        data_river = ds_his_slice[TP_VARIABLE].values[:, idx_river]
        signal     = sign * (data_mouth - data_river)
    else:
        signal = sign * data_mouth

    ds_his.close()
    return time_seconds.astype(float), signal


def compute_tidal_prism(time_seconds, signal):
    """Tidal prism = sum(|signal|) * dt / 2, scaled to TP_PRISM_UNIT."""
    dt    = np.median(np.diff(time_seconds))
    prism = np.sum(np.abs(signal)) * dt / 2 / TP_PRISM_SCALE
    return prism


# =============================================================================
# %% --- COMPUTE: INTERTIDAL AREA ---
# =============================================================================
print('\n' + '#'*60)
print('  COMPUTING: Intertidal Area')
print('#'*60)

ia_datadict = {}

for discharge in DISCHARGES:
    dhr_base         = BASE_DIR / f'Q{discharge}' / 'detailed-hydro-run'
    scenario_folders = discover_scenario_folders(dhr_base, discharge)

    if not scenario_folders:
        print(f'[SKIP] No matching folders in: {dhr_base}')
        continue

    print(f"\n{'='*60}\nQ = {discharge} m3/s: {len(scenario_folders)} scenario(s)\n{'='*60}")

    cache_dir = dhr_base / 'cached_data'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)

    for folder_path, run_id, pm_val, n_val in scenario_folders:
        label = f'Qr{discharge}_dhr{run_id:02d}_pm{pm_val}_n{n_val}'
        print(f'  Processing: {folder_path.name}  ->  {label}')

        run_paths = get_stitched_map_run_paths(
            base_path=dhr_base, folder_name=folder_path.name,
            timed_out_dir=None, variability_map=None, analyze_noisy=False,
        ) or [folder_path]

        ds = load_or_update_map_cache_multi(
            cache_dir=cache_dir, folder_name=folder_path.name, run_paths=run_paths,
            var_names=IA_LOAD_VARS, bbox=CACHE_BBOX,
            append_time=True, append_vars=True, cache_tag=cache_tag,
        )

        if ds is None or 'time' not in ds.dims \
                or 'mesh2d_waterdepth' not in ds or 'mesh2d_flowelem_ba' not in ds:
            print(f'    [SKIP] missing data for {label}')
            if ds is not None:
                ds.close()
            continue

        if IA_FACE_X_VAR not in ds.coords and IA_FACE_X_VAR not in ds:
            print(f'    [SKIP] missing {IA_FACE_X_VAR} for {label}')
            ds.close()
            continue

        time_values  = np.asarray(ds.time.values).astype('datetime64[ns]')
        window_mask  = get_last_n_hours_window(time_values, IA_WINDOW_HOURS)
        n_window     = int(window_mask.sum())

        if n_window < 2:
            print(f'    [SKIP] only {n_window} timestep(s) in last {IA_WINDOW_HOURS}h window')
            ds.close()
            continue

        actual_h = (time_values[window_mask][-1] - time_values[window_mask][0]) / np.timedelta64(1, 'h')
        if actual_h < 11.0:
            print(f'    [WARNING] window is {actual_h:.1f}h -- shorter than one tidal cycle')

        ds_window = ds.isel(time=np.where(window_mask)[0])
        area_m2, n_inter, n_faces = compute_intertidal_area(ds_window)
        ds.close()

        area_km2 = area_m2 / 1e6
        print(f'    Intertidal area = {area_m2:,.0f} m2  ({area_km2:.4f} km2)  '
              f'[{n_inter}/{n_faces} faces]')

        ia_datadict[label] = {
            'label':               label,
            'discharge':           discharge,
            'run_id':              run_id,
            'pm':                  pm_val,
            'n':                   n_val,
            'intertidal_area_m2':  area_m2,
            'intertidal_area_km2': area_km2,
        }

print(f'\nCollected intertidal-area results for {len(ia_datadict)} runs.')
ia_summary = pd.DataFrame(ia_datadict.values())
ia_summary.to_csv(OUTPUT_DIR / 'intertidal_area_summary.csv', index=False)


# =============================================================================
# %% --- COMPUTE: MAX TIDAL INTRUSION ---
# =============================================================================
print('\n' + '#'*60)
print('  COMPUTING: Max Tidal Intrusion')
print('#'*60)

lti_x_edges   = np.arange(LTI_X_MIN, LTI_X_MAX + LTI_X_BIN_WIDTH, LTI_X_BIN_WIDTH)
lti_x_centers = 0.5 * (lti_x_edges[:-1] + lti_x_edges[1:])

lti_datadict = {}

for discharge in DISCHARGES:
    dhr_base         = BASE_DIR / f'Q{discharge}' / 'detailed-hydro-run'
    scenario_folders = discover_scenario_folders(dhr_base, discharge)

    if not scenario_folders:
        print(f'[SKIP] No matching folders in: {dhr_base}')
        continue

    print(f"\n{'='*60}\nQ = {discharge} m3/s: {len(scenario_folders)} scenario(s)\n{'='*60}")

    cache_dir = dhr_base / 'cached_data'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)

    for folder_path, run_id, pm_val, n_val in scenario_folders:
        label = f'Qr{discharge}_dhr{run_id:02d}_pm{pm_val}_n{n_val}'
        print(f'  Processing: {folder_path.name}  ->  {label}')

        run_paths = get_stitched_map_run_paths(
            base_path=dhr_base, folder_name=folder_path.name,
            timed_out_dir=None, variability_map=None, analyze_noisy=False,
        ) or [folder_path]

        ds = load_or_update_map_cache_multi(
            cache_dir=cache_dir, folder_name=folder_path.name, run_paths=run_paths,
            var_names=LTI_LOAD_VARS, bbox=CACHE_BBOX,
            append_time=True, append_vars=True, cache_tag=cache_tag,
        )

        if ds is None or 'time' not in ds.dims or LTI_VAR_NAME not in ds:
            print(f'    [SKIP] missing data for {label}')
            if ds is not None:
                ds.close()
            continue

        t_start, t_end = get_last_n_days_window_ds(ds, LTI_N_DAYS)
        ds_win = ds.sel(time=slice(t_start, t_end))

        if len(ds_win.time) == 0:
            print(f'    [SKIP] no timesteps after slicing to last {LTI_N_DAYS} day(s)')
            ds.close()
            continue

        face_x      = ds_win.grid.face_coordinates[:, 0]
        time_values = np.asarray(ds_win.time.values).astype('datetime64[ns]')

        Ubar           = build_ubar_matrix(ds_win, face_x, LTI_VAR_NAME, lti_x_edges)
        fronts_instant = np.full(len(time_values), np.nan)
        for idx in range(len(time_values)):
            fronts_instant[idx] = front_from_profile(
                Ubar[idx, :], lti_x_centers, LTI_FLOOD_VEL_THRESHOLD)

        ds.close()

        # Save per-run time series
        df_front = pd.DataFrame({'time': time_values, 'front_x_instant': fronts_instant})
        df_front.to_csv(OUTPUT_DIR / f'front_x_{label}.csv', index=False)

        valid = fronts_instant[~np.isnan(fronts_instant)]
        if len(valid) == 0:
            print('    Done. No valid front_x found -- check LTI_FLOOD_VEL_THRESHOLD.')
            continue

        lti_max = float(valid.max())
        print(f'    LTI_max = {lti_max:.0f} m  ({lti_max / 1000:.2f} km)')

        lti_datadict[label] = {
            'label':              label,
            'discharge':          discharge,
            'run_id':             run_id,
            'pm':                 pm_val,
            'n':                  n_val,
            'LTI_instant_max_m':  lti_max,
            'LTI_instant_max_km': lti_max / 1000,
        }

print(f'\nCollected LTI results for {len(lti_datadict)} runs.')
lti_summary = pd.DataFrame(lti_datadict.values())
lti_summary.to_csv(OUTPUT_DIR / 'LTI_summary.csv', index=False)


# =============================================================================
# %% --- COMPUTE: TIDAL PRISM ---
# =============================================================================
print('\n' + '#'*60)
print('  COMPUTING: Tidal Prism')
print('#'*60)

tp_datadict = {}

for discharge in DISCHARGES:
    dhr_base         = BASE_DIR / f'Q{discharge}' / 'detailed-hydro-run'
    scenario_folders = discover_scenario_folders(dhr_base, discharge)

    if not scenario_folders:
        print(f'[SKIP] No matching folders in: {dhr_base}')
        continue

    print(f"\n{'='*60}\nQ = {discharge} m3/s: {len(scenario_folders)} scenario(s)\n{'='*60}")

    for folder_path, run_id, pm_val, n_val in scenario_folders:
        label = f'Qr{discharge}_dhr{run_id:02d}_pm{pm_val}_n{n_val}'
        print(f'  Processing: {folder_path.name}  ->  {label}')

        his_paths = get_stitched_his_paths(
            base_path=dhr_base, folder_name=folder_path.name,
            timed_out_dir=None, variability_map=None, analyze_noisy=False,
        )
        if not his_paths:
            his_paths = list(folder_path.glob('*_his.nc'))
        if not his_paths:
            print('    [SKIP] No HIS data found.')
            continue

        t_start, t_end = get_last_n_days_window_paths(his_paths, TP_WINDOW_DAYS)
        time_s, signal = load_his_signal(his_paths, t_start, t_end)
        prism          = compute_tidal_prism(time_s, signal)

        print(f'    Tidal prism = {prism:.3f} {TP_PRISM_UNIT}')

        tp_datadict[label] = {
            'label':     label,
            'discharge': discharge,
            'run_id':    run_id,
            'pm':        pm_val,
            'n':         n_val,
            'prism_Mm3': prism,
        }

print(f'\nCollected tidal-prism results for {len(tp_datadict)} runs.')
tp_summary = pd.DataFrame(tp_datadict.values())
tp_summary.to_csv(OUTPUT_DIR / 'tidal_prism_summary.csv', index=False)


# =============================================================================
# %% --- PLOT HELPER ---
# =============================================================================

def plot_metric(ax, datadict, metric_key, xlabel, normalize=False, add_legend=True,
                annotate=True, fontsize_labels=None, fontsize_ticks=None):
    """Scatter plot of metric vs pm, coloured+marked by discharge. Adds a
    vertical dashed line for the constant (n=0) scenario per discharge, and
    an optional linear trend line if SHOW_TRENDLINE is True.

    Parameters
    ----------
    add_legend : bool
        Draw a legend on this axes.
    annotate : bool
        Annotate each point with its n-value.
    fontsize_labels : int or None
        Override FONTSIZE_LABELS (e.g. for AGU-sized subplot).
    fontsize_ticks : int or None
        Override FONTSIZE_TICKS (e.g. for AGU-sized subplot).
    """
    fs_labels = fontsize_labels if fontsize_labels is not None else FONTSIZE_LABELS
    fs_ticks  = fontsize_ticks  if fontsize_ticks  is not None else FONTSIZE_TICKS
    if not datadict:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=fs_ticks)
        return

    present_discharges = sorted({v['discharge'] for v in datadict.values()})

    for discharge in present_discharges:
        color  = DISCHARGE_COLORS.get(discharge, 'tab:gray')
        marker = DISCHARGE_MARKERS.get(discharge, 'o')
        subset = {k: v for k, v in datadict.items() if v['discharge'] == discharge}

        pm_arr     = np.array([v['pm']         for v in subset.values()], dtype=float)
        metric_arr = np.array([v[metric_key]   for v in subset.values()], dtype=float)
        n_arr      = np.array([v['n']          for v in subset.values()], dtype=float)

        # Optional normalisation to n=0 constant run
        if normalize:
            const_vals = metric_arr[n_arr == 0]
            if len(const_vals) > 0:
                metric_arr = metric_arr / const_vals[0]

        # Dashed vertical line at the constant (n=0) scenario value (no per-line legend entry)
        const_mask = n_arr == 0
        if const_mask.any():
            const_x = metric_arr[const_mask][0]
            ax.axvline(
                const_x, color=color, linestyle='--', linewidth=1.4,
                alpha=0.75, zorder=2,
                label='_nolegend_',
            )

        label_q = f'Q\u200a=\u200a{discharge} m\u00b3/s'
        ax.scatter(metric_arr, pm_arr, color=color, marker=marker,
                   zorder=4, s=30, label=label_q)

        if annotate:
            for x_, y_, n_ in zip(metric_arr, pm_arr, n_arr):
                ax.annotate(
                    f'n{int(n_)}', (x_, y_), textcoords='offset points',
                    xytext=(4, 4), fontsize=fs_ticks, color=color,
                )

        # Optional linear trend line per discharge
        if SHOW_TRENDLINE and len(metric_arr) >= 2:
            slope, intercept, *_ = stats.linregress(metric_arr, pm_arr)
            x_fit = np.linspace(metric_arr.min(), metric_arr.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(
                x_fit, y_fit, color=color, linestyle=':', linewidth=1.8,
                alpha=0.85, zorder=3,
            )

    # Single legend entry for all dashed constant-discharge lines
    ax.plot([], [], color='gray', linestyle='--', linewidth=1.4, alpha=0.75,
            label='constant discharge (n\u200a=\u200a0)')

    ax.set_xlabel(xlabel, fontsize=fs_labels)
    ax.set_ylabel('discharge amplitude $R_{\\mathrm{peak}}$', fontsize=fs_labels)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs_ticks)
    if add_legend:
        ax.legend(fontsize=fs_ticks - 1, loc='center left', bbox_to_anchor=(1.0, 0.5))


# =============================================================================
# %% --- PLOT: INTERTIDAL AREA ---
# =============================================================================
if ia_datadict:
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_key = 'intertidal_area_km2'
    if NORMALIZE:
        xlabel = 'normalized intertidal area [-]'
        fname  = 'intertidal_area_vs_peak_amplitude_allQ_normalized'
    else:
        xlabel = 'intertidal area [km\u00b2]'
        fname  = 'intertidal_area_vs_peak_amplitude_allQ'

    plot_metric(ax, ia_datadict, metric_key, xlabel, normalize=NORMALIZE)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight', transparent=_tr)
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / f"{fname}.png"}')
else:
    print('[SKIP PLOT] No intertidal area data.')


# =============================================================================
# %% --- PLOT: MAX TIDAL INTRUSION ---
# =============================================================================
if lti_datadict:
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_key = 'LTI_instant_max_km'
    if NORMALIZE:
        xlabel = 'normalized x-location of max tidal intrusion [-]'
        fname  = 'LTI_vs_peak_amplitude_allQ_normalized'
    else:
        xlabel = 'x-location of max tidal intrusion [km]'
        fname  = 'LTI_vs_peak_amplitude_allQ'

    plot_metric(ax, lti_datadict, metric_key, xlabel, normalize=NORMALIZE)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight', transparent=_tr)
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / f"{fname}.png"}')
else:
    print('[SKIP PLOT] No tidal intrusion data.')


# =============================================================================
# %% --- PLOT: TIDAL PRISM ---
# =============================================================================
if tp_datadict:
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_key = 'prism_Mm3'
    if NORMALIZE:
        xlabel = 'normalized tidal prism [-]'
        fname  = 'tidal_prism_vs_peak_amplitude_allQ_normalized'
    else:
        xlabel = f'tidal prism [{TP_PRISM_UNIT}]'
        fname  = 'tidal_prism_vs_peak_amplitude_allQ'

    plot_metric(ax, tp_datadict, metric_key, xlabel, normalize=NORMALIZE)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight', transparent=_tr)
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / f"{fname}.png"}')
else:
    print('[SKIP PLOT] No tidal prism data.')


print(f'\nAll outputs written to: {OUTPUT_DIR.resolve()}')


# =============================================================================
# %% --- PLOT: COMBINED 3-PANEL SUBPLOT FIGURE ---
# =============================================================================
# One row, three columns, shared y-axis, single legend.

_AGU_RC = {
    'font.family':     'sans-serif',
    'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
    'font.size':        FONTSIZE_TICKS,
    'axes.labelsize':   FONTSIZE_LABELS,
    'xtick.labelsize':  FONTSIZE_TICKS,
    'ytick.labelsize':  FONTSIZE_TICKS,
    'legend.fontsize':  FONTSIZE_TICKS,
    'axes.titlesize':   FONTSIZE_LABELS,
}

_panels = [
    (
        ia_datadict,
        'intertidal_area_km2',
        'intertidal area [km\u00b2]' if not NORMALIZE else 'norm. intertidal area [-]',
    ),
    (
        lti_datadict,
        'LTI_instant_max_km',
        'max tidal intrusion [km]' if not NORMALIZE else 'norm. max tidal intrusion [-]',
    ),
    (
        tp_datadict,
        'prism_Mm3',
        f'tidal prism [{TP_PRISM_UNIT}]' if not NORMALIZE else 'norm. tidal prism [-]',
    ),
]

import matplotlib as _mpl
if any(d for d, *_ in _panels):
    with _mpl.rc_context(_AGU_RC):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(7.5, 2.5),   # AGU full-page width = 7.5 in
            sharey=True,
            constrained_layout=True,
        )

        for ax, (datadict, metric_key, xlabel) in zip(axes, _panels):
            plot_metric(
                ax, datadict, metric_key, xlabel,
                normalize=NORMALIZE,
                add_legend=False,
                annotate=True,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Remove duplicate y-label on panels 2 and 3
        for ax in axes[1:]:
            ax.set_ylabel('')

        # Single legend assembled from the rightmost panel's handles
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            fontsize=FONTSIZE_TICKS,
            loc='upper right',
            bbox_to_anchor=(0, 0.5),
            frameon=True,
        )

        fname_sub = 'hydro_metrics_combined_subplot' + ('_normalized' if NORMALIZE else '')
        fig.savefig(OUTPUT_DIR / f'{fname_sub}.png', dpi=300, bbox_inches='tight', transparent=_tr)
        fig.savefig(OUTPUT_DIR / f'{fname_sub}.pdf', bbox_inches='tight', transparent=_tr)
        plt.show()
        plt.close(fig)
        print(f'Saved: {OUTPUT_DIR / f"{fname_sub}.png"}')
else:
    print('[SKIP PLOT] No data available for combined subplot figure.')

# %%
