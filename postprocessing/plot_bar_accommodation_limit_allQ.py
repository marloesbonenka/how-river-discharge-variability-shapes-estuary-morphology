# %% Where does discharge variability stop raising the bars? Bar-signal vs accommodation limit
"""
In a river model, extra sediment supply at a fixed base level aggrades the bed all the
way upstream, without a downstream limit. In the estuary the tidal water levels cap how
high bars can build, so the bar response to discharge variability should end somewhere
along the estuary. This script measures where, for all mean discharges (Q = 250 / 500 /
1000 m3/s) and all peak amplitudes R_peak (= pm).

Two lengths per variable-discharge scenario (each compared with the constant scenario,
n = 0, of the same Q):

x_B  bar-signal limit: seaward end of the reach where
         dp95(x) = p95_var(x) - p95_const(x)
     exceeds the natural variability of the constant run.
x_A  accommodation limit: seaward end of the reach where the accommodation ceiling
         dWL(x) = WL_peak,var(x) - WL_const(x)
     exceeds WL_NOISE_FLOOR_M (X_A_FROM = 'ceiling'). With X_A_FROM = 'accommodation'
     it uses dA(x) = A_var(x) - A_const(x) with A = WL - p95 bed level.

Hypothesis: peak flows lift the water level above the tidal high-water envelope landward
of x_A and give the river extra accommodation space; seaward of x_A the tide sets the
ceiling and variability cannot add sediment storage, so x_B ~ x_A.
  R_peak should move x_A and x_B seaward (higher peaks, higher peak water levels).
  V (sediment supply through km 45) should raise the amplitude of dp95 landward but
  not move x_B on its own. At Q = 500 runs with the same R_peak and different n_peaks
  (different V) test this directly.

Data
----
Detailed-hydro runs (dhr_*, hydrodynamics on the FINAL bed of each morphodynamic run):
  WL     p{WL_PERCENTILE} of the water level (mesh2d_s1) over (time x wet faces) per 1-km
         bin, last WL_WINDOW_H of the run (the Fig. 8 accommodation-space quantity).
         Constant run at Qmean = tidal ceiling; '_peak' run of a scenario (Q = R_peak x
         Qmean) = ceiling at peak flow; '_mean' run = effect of the morphology alone.
  bars   p{BAR_PERCENTILE} bed level (mesh2d_mor_bl, first output = final morphology)
         per bin, inside the strip Y_RANGE_M and the initial estuary footprint.
Morphodynamic runs (only for quantities that need the whole run):
  noise  temporal std of the constant run's p95 profile over its last EVAL_YEARS
         (map snapshots), the natural variability of the bars.
  V      cumulative bedload through the km-45 cross-section over the whole run (HIS),
         shown as V / V_const of the same Q.
L_TI     optional, from LTI_summary.csv of plot_map_hydro_tidalmetrics_comparison_allQ.py.

Scenarios without a peak-flow dhr run get x_B but no x_A.

Outputs (OUTPUT_DIR)
--------------------
bar_accommodation_profiles   rows: dWL | dp95 | dA, columns: Q; R_peak runs with
                             n = PROFILE_N, natural variability shaded, x_A / x_B marked
bar_accommodation_limits     (a) x_A, x_B (and L_TI) vs R_peak, (b) x_B vs x_A,
                             (c) x_B vs V/V_const, (d) landward dp95 amplitude vs V/V_const
bar_accommodation_summary.csv, bar_accommodation_profiles.csv
"""

# %% Imports
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from functions.F_general import (
    apply_plot_style,
    get_last_n_hours_window,
    get_variability_map,
    parse_restart_folder_label,
    strip_top_right_spines,
)
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_his_paths, get_stitched_map_run_paths, load_and_cache_scenario

warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

# =============================================================================
# %% --- CONFIGURATION ---
# =============================================================================
BASE_DIR   = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
DISCHARGES = [250, 500, 1000]
RUN_IDS_TO_INCLUDE = None     # None = every run found; {1, 6, 9, 10, 11} = the set common to all Q
PROFILE_N  = 3                # profile figure: variable runs with this n only, so R_peak is the only change

# Morphodynamic runs: {run_id}_Qr{Q}_pm{pm}_n{n}.{jobid}  (after stripping an rst prefix)
MORPH_FOLDER_RE = re.compile(r'^(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)\.\d+$')
# Detailed-hydro runs: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_{condition}][.{jobid}]
DHR_FOLDER_RE = re.compile(r'^dhr_(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)(?:_([a-z]+))?(?:\.\d+)?$')
DHR_CONDITION = {None: 'mean', 'mean': 'mean', 'meanflow': 'mean',
                 'peak': 'peak', 'peakflow': 'peak', 'lowflow': 'low'}
# dhr folders to ignore, e.g. a peak run forced with the wrong peak discharge
DHR_EXCLUDE = {
    'dhr_03_Qr500_pm3_n5_peak.964907',   # forced with the 5x peak (2500 m3/s), not 3 x Qmean
}

# Water level (dhr MAP)
WL_VAR         = 'mesh2d_s1'
WL_WINDOW_H    = 24.0         # last hours of the dhr run (~2 tidal cycles, after spin-up)
WL_PERCENTILE  = 95
WET_DEPTH_M    = 0.05         # wet face: water level - bed level > this

# Bars
BED_VAR             = 'mesh2d_mor_bl'
CACHE_BBOX          = [1, 1, 45000, 15000]
CACHE_TAG           = None
X_RANGE_M           = (20000.0, 45000.0)
BIN_M               = 1000.0
Y_RANGE_M           = (5000.0, 10000.0)
ESTUARY_INIT_BL_MAX = 2.2     # initial estuary footprint (t = 0 of the morphodynamic run); None = whole strip
BAR_PERCENTILE      = 95
EVAL_YEARS          = 3       # window of the constant morphodynamic run for the bar noise
HYDRO_YEAR_DAYS     = 365.25

# Supply (morphodynamic HIS)
SED_VAR  = 'cross_section_bedload_sediment_transport'   # cumulative
RIVER_KM = 45

# Detection of the limits
K_SIGMA           = 2.0       # bar threshold = K_SIGMA x natural variability of the constant run
BED_NOISE_FLOOR_M = 0.05      # minimum threshold for dp95 [m]
WL_NOISE_FLOOR_M  = 0.05      # threshold for dWL [m] (one dhr run per scenario: no variability estimate)
MAX_GAP_BINS      = 1         # bins below threshold tolerated inside the signal reach
X_A_FROM          = 'ceiling' # 'ceiling' (dWL) | 'accommodation' (dA)
UPSTREAM_REACH_KM = (35.0, 45.0)   # reach of the dp95 amplitude in panel (d)

# Optional L_TI overlay (CSV of plot_map_hydro_tidalmetrics_comparison_allQ.py)
SHOW_LTI = True
LTI_CSV  = BASE_DIR / 'output_plots_combined' / 'hydro_tidalmetrics_comparison' / 'LTI_summary.csv'

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'bar_accommodation_limit'
RUN_CACHE_DIR = OUTPUT_DIR / 'run_cache'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MASK_TAG = f"bl{ESTUARY_INIT_BL_MAX if ESTUARY_INIT_BL_MAX is not None else 'all'}_dx{BIN_M:g}"
DHR_TAG = f"p{WL_PERCENTILE}wl{WL_WINDOW_H:g}h_p{BAR_PERCENTILE}_{_MASK_TAG}"
NOISE_TAG = f"eval{EVAL_YEARS:g}y_p{BAR_PERCENTILE}_{_MASK_TAG}"

# Style (colours/markers per Q as in the tidal-metrics figure; R_peak in Blues as in the sensitivity figures)
DISCHARGE_COLORS  = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}
DISCHARGE_MARKERS = {250: '^', 500: 'o', 1000: 's'}
GREY_CONST = '#7f7f7f'
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
DPI = 300
SHOW_FIGURES = True

apply_plot_style('AGU', font_size=8)

X_EDGES = np.arange(X_RANGE_M[0], X_RANGE_M[1] + 0.5 * BIN_M, BIN_M)
X_CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
X_KM = X_CENTERS / 1000.0


# =============================================================================
# %% --- HELPERS: discovery and loading ---
# =============================================================================
def discover_morph_runs(q_dir, discharge):
    """{run_id: (folder, pm, n)} for the morphodynamic runs of one discharge."""
    runs = {}
    for folder in sorted(q_dir.iterdir()):
        m = MORPH_FOLDER_RE.match(parse_restart_folder_label(folder.name)[1]) if folder.is_dir() else None
        if not m:
            continue
        run_id, q_val, pm, n = (int(g) for g in m.groups())
        if q_val == discharge and (RUN_IDS_TO_INCLUDE is None or run_id in RUN_IDS_TO_INCLUDE):
            runs[run_id] = (folder, pm, n)
    return runs


def discover_dhr_runs(dhr_dir, discharge):
    """{(run_id, condition): (folder, pm, n)}; the first folder wins for duplicates."""
    runs = {}
    for folder in sorted(dhr_dir.iterdir()):
        m = DHR_FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not m or folder.name in DHR_EXCLUDE:
            continue
        run_id, q_val, pm, n = (int(g) for g in m.groups()[:4])
        condition = DHR_CONDITION.get(m.group(5))
        if (q_val != discharge or condition is None
                or (RUN_IDS_TO_INCLUDE is not None and run_id not in RUN_IDS_TO_INCLUDE)):
            continue
        runs.setdefault((run_id, condition), (folder, pm, n))
    return runs


def open_map_cache(base_dir, folder, var_names, variability_map=None):
    run_paths = get_stitched_map_run_paths(
        base_path=base_dir, folder_name=folder.name, timed_out_dir=base_dir / 'timed-out',
        variability_map=variability_map, analyze_noisy=False,
    ) or [folder]
    ds = load_or_update_map_cache_multi(
        cache_dir=base_dir / 'cached_data', folder_name=folder.name, run_paths=run_paths,
        var_names=var_names, bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None or any(v not in ds for v in var_names):
        if ds is not None:
            ds.close()
        raise FileNotFoundError(f"missing {var_names} for {folder.name}")
    return ds


def bin_percentile(values, face_x, mask, q):
    """Percentile per x-bin of values (nfaces,) or (nt, nfaces), NaN ignored."""
    values = np.atleast_2d(values)
    fbin = np.digitize(face_x, X_EDGES) - 1
    out = np.full(X_CENTERS.size, np.nan)
    for b in range(X_CENTERS.size):
        vals = values[:, mask & (fbin == b)].ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[b] = np.percentile(vals, q)
    return out


def constant_reference(q_dir, folder):
    """From the constant MORPHODYNAMIC run: face x, estuary mask (initial footprint) and the
    p95 bar profile of every map snapshot in the last EVAL_YEARS (for the natural variability)."""
    cache = RUN_CACHE_DIR / f"ref_{folder.name}_{NOISE_TAG}.npz"
    if cache.exists():
        return dict(np.load(cache))
    ds = open_map_cache(q_dir, folder, [BED_VAR], get_variability_map(int(q_dir.name[1:])))
    try:
        times = np.asarray(ds.time.values).astype('datetime64[ns]')
        order = np.argsort(times)                        # cached outputs are not stored in time order
        t_start = times[order[-1]] - np.timedelta64(int(EVAL_YEARS * HYDRO_YEAR_DAYS * 86400), 's')
        sel = np.sort(order[times[order] >= t_start])
        face_x, face_y = (np.asarray(c, dtype=float) for c in _get_face_coords(ds))
        mask = (face_y >= Y_RANGE_M[0]) & (face_y <= Y_RANGE_M[1])
        if ESTUARY_INIT_BL_MAX is not None:
            mask &= ds[BED_VAR].isel(time=int(order[0])).values < ESTUARY_INIT_BL_MAX
        bed = ds[BED_VAR].isel(time=sel).values
    finally:
        ds.close()
    p_bar = np.array([bin_percentile(b, face_x, mask, BAR_PERCENTILE) for b in bed])
    out = {'face_x': face_x, 'mask': mask, 'p_bar': p_bar}
    np.savez(cache, **out)
    return out


def dhr_profiles(dhr_dir, folder, ref):
    """From one detailed-hydro run: p{WL_PERCENTILE} water level over the last WL_WINDOW_H
    (wet faces of the strip) and p{BAR_PERCENTILE} bed level of the first output."""
    cache = RUN_CACHE_DIR / f"{folder.name}_{DHR_TAG}.npz"
    if cache.exists():
        return dict(np.load(cache))
    ds = open_map_cache(dhr_dir, folder, [WL_VAR, BED_VAR])
    try:
        face_x, face_y = (np.asarray(c, dtype=float) for c in _get_face_coords(ds))
        if face_x.size != ref['face_x'].size or not np.allclose(face_x, ref['face_x']):
            raise ValueError(f"{folder.name}: faces differ from the morphodynamic run, "
                             "cannot apply the initial estuary footprint")
        times = np.asarray(ds.time.values).astype('datetime64[ns]')
        order = np.argsort(times)
        win = order[get_last_n_hours_window(times[order], WL_WINDOW_H)]
        win = np.sort(win)
        s1 = ds[WL_VAR].isel(time=win).values
        bl = ds[BED_VAR].isel(time=win).values
        bed0 = ds[BED_VAR].isel(time=int(order[0])).values
    finally:
        ds.close()
    strip = (face_y >= Y_RANGE_M[0]) & (face_y <= Y_RANGE_M[1])
    wl = np.where(s1 - bl > WET_DEPTH_M, s1, np.nan)
    out = {'wl': bin_percentile(wl, face_x, strip, WL_PERCENTILE),
           'bar': bin_percentile(bed0, face_x, ref['mask'], BAR_PERCENTILE),
           'window_h': float((times[win[-1]] - times[win[0]]) / np.timedelta64(1, 'h'))}
    np.savez(cache, **out)
    return out


def upstream_supply(q_dir, folder):
    """Cumulative bedload through the km-RIVER_KM cross-section over the whole morphodynamic run."""
    cache = RUN_CACHE_DIR / f"V_{folder.name}_km{RIVER_KM}.npz"
    if cache.exists():
        return float(np.load(cache)['V'])
    his_paths = get_stitched_his_paths(base_path=q_dir, folder_name=folder.name,
                                       timed_out_dir=q_dir / 'timed-out')
    if not his_paths:
        raise FileNotFoundError(f"no HIS output for {folder.name}")
    parts = folder.name.split('_')      # same cache name as plot_his_upstream_cumulative_sediment_km44.py
    his_cache = q_dir / 'cached_data' / f"hisoutput_{int(parts[0])}_{'_'.join(parts[1:])}.nc"
    _, data = load_and_cache_scenario(scenario_dir=folder, his_file_paths=his_paths,
                                      cache_file=his_cache, boxes=[], var_name=SED_VAR)
    km = np.asarray(data['km_positions'], dtype=float)
    series = np.asarray(data[SED_VAR], dtype=float)[:, int(np.argmin(np.abs(km - RIVER_KM)))]
    v = float(series[-1] - series[0])
    np.savez(cache, V=v)
    return v


def signal_limit(exceeds, max_gap=MAX_GAP_BINS):
    """Seaward end [km] of the signal reach, walking from the river towards the sea.

    The reach starts at the most landward bin that exceeds the threshold and ends where
    more than max_gap consecutive bins do not. Returns the seaward edge of its last bin,
    or NaN if no bin exceeds the threshold.
    """
    last, gap = None, 0
    for i in range(exceeds.size - 1, -1, -1):
        if exceeds[i]:
            last, gap = i, 0
        elif last is not None:
            gap += 1
            if gap > max_gap:
                break
    return np.nan if last is None else X_EDGES[last] / 1000.0


# =============================================================================
# %% --- COMPUTE ---
# =============================================================================
NAN_PROFILE = np.full(X_CENTERS.size, np.nan)
summary_rows, profile_rows = [], []
profiles = {}        # Q -> {'thr_*': ..., 'runs': [...]}

for discharge in DISCHARGES:
    q_dir = BASE_DIR / f'Q{discharge}'
    dhr_dir = q_dir / 'detailed-hydro-run'
    morph = discover_morph_runs(q_dir, discharge)
    dhr = discover_dhr_runs(dhr_dir, discharge)
    print(f"\n{'=' * 60}\nQ = {discharge} m3/s: {len(morph)} morphodynamic, {len(dhr)} detailed-hydro run(s)")
    for (rid, cond), (f, _, _) in sorted(dhr.items()):
        print(f"  dhr {rid:02d} {cond:5s} {f.name}")

    const_id = next((rid for rid, (_, _, n) in morph.items() if n == 0), None)
    if const_id is None or (const_id, 'mean') not in dhr:
        print(f"[SKIP] Q = {discharge}: no constant morphodynamic and/or detailed-hydro run")
        continue
    try:
        ref = constant_reference(q_dir, morph[const_id][0])
        const = dhr_profiles(dhr_dir, dhr[(const_id, 'mean')][0], ref)
        v_const = upstream_supply(q_dir, morph[const_id][0])
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[SKIP] Q = {discharge}: {exc}")
        continue

    thr_b = np.maximum(K_SIGMA * np.nanstd(ref['p_bar'], axis=0), BED_NOISE_FLOOR_M)
    thr_w = np.full(X_CENTERS.size, WL_NOISE_FLOOR_M)
    thr_a = np.hypot(thr_w, thr_b)
    acc_c = const['wl'] - const['bar']
    profiles[discharge] = {'thr_b': thr_b, 'thr_w': thr_w, 'thr_a': thr_a, 'runs': []}
    reach = (X_KM >= UPSTREAM_REACH_KM[0]) & (X_KM <= UPSTREAM_REACH_KM[1])

    for rid, (folder, pm, n) in sorted(morph.items(), key=lambda kv: (kv[1][1], kv[1][2])):
        if n == 0 or not ((rid, 'mean') in dhr or (rid, 'peak') in dhr):
            continue
        try:
            mean = dhr_profiles(dhr_dir, dhr[(rid, 'mean')][0], ref) if (rid, 'mean') in dhr else None
            peak = dhr_profiles(dhr_dir, dhr[(rid, 'peak')][0], ref) if (rid, 'peak') in dhr else None
            v = upstream_supply(q_dir, folder)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  [SKIP] {folder.name}: {exc}")
            continue

        bar_v = (mean or peak)['bar']                    # same final bed in both runs
        d_bar = bar_v - const['bar']
        d_wl_peak = peak['wl'] - const['wl'] if peak is not None else NAN_PROFILE
        d_wl_mean = mean['wl'] - const['wl'] if mean is not None else NAN_PROFILE
        d_acc = (peak['wl'] - bar_v) - acc_c if peak is not None else NAN_PROFILE
        x_b = signal_limit(np.nan_to_num(d_bar) > thr_b)
        if peak is None:
            x_a = np.nan
        elif X_A_FROM == 'ceiling':
            x_a = signal_limit(np.nan_to_num(d_wl_peak) > thr_w)
        else:
            x_a = signal_limit(np.abs(np.nan_to_num(d_acc)) > thr_a)

        profiles[discharge]['runs'].append(dict(pm=pm, n=n, d_bar=d_bar, d_wl_peak=d_wl_peak,
                                                d_wl_mean=d_wl_mean, d_acc=d_acc, x_A=x_a, x_B=x_b))
        summary_rows.append({
            'discharge': discharge, 'run_id': rid, 'pm': pm, 'n': n, 'folder': folder.name,
            'has_peak_run': peak is not None, 'V': v, 'V_const': v_const,
            'V_ratio': v / v_const if v_const else np.nan,
            'x_A_km': x_a, 'x_B_km': x_b, 'x_B_minus_x_A_km': x_b - x_a,
            'dp95_upstream_m': float(np.nanmean(d_bar[reach])),
            'dWL_peak_upstream_m': float(np.nanmean(d_wl_peak[reach])),
            'dWL_mean_upstream_m': float(np.nanmean(d_wl_mean[reach])),
        })
        for i, xk in enumerate(X_KM):
            profile_rows.append({
                'discharge': discharge, 'pm': pm, 'n': n, 'x_km': xk,
                'dWL_peak_m': d_wl_peak[i], 'dWL_mean_m': d_wl_mean[i], 'thr_WL_m': thr_w[i],
                'dp95_m': d_bar[i], 'thr_p95_m': thr_b[i], 'dA_m': d_acc[i], 'thr_A_m': thr_a[i],
            })

summary = pd.DataFrame(summary_rows)
if SHOW_LTI and LTI_CSV.exists() and not summary.empty:
    lti = pd.read_csv(LTI_CSV)[['discharge', 'pm', 'n', 'LTI_instant_max_km']]
    summary = summary.merge(lti, on=['discharge', 'pm', 'n'], how='left')
summary.to_csv(OUTPUT_DIR / 'bar_accommodation_summary.csv', index=False)
pd.DataFrame(profile_rows).to_csv(OUTPUT_DIR / 'bar_accommodation_profiles.csv', index=False)

if not summary.empty:
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print('\n' + summary[['discharge', 'pm', 'n', 'has_peak_run', 'V_ratio', 'x_A_km', 'x_B_km',
                              'dp95_upstream_m', 'dWL_peak_upstream_m', 'dWL_mean_upstream_m']]
              .round(3).to_string(index=False))


# =============================================================================
# %% --- PLOT HELPERS ---
# =============================================================================
all_pm = sorted({r['pm'] for p in profiles.values() for r in p['runs']})
_n_pm = max(len(all_pm) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm)}


def save_figure(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f'{name}.{ext}', dpi=DPI, bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / name}.png / .pdf')


def mark_limit(ax, x_km, color, filled=True):
    """Triangle on the top axis edge at a limit position."""
    if np.isfinite(x_km):
        ax.plot(x_km, 1.0, marker='v', ms=5, color=color, mfc=color if filled else 'white',
                mew=0.8, transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)


def label_point(ax, x, y, r, color):
    if np.isfinite(x) and np.isfinite(y):
        ax.annotate(f"{int(r['pm'])}/{int(r['n'])}", (x, y), textcoords='offset points',
                    xytext=(3, 3), fontsize=6, color=color)


# =============================================================================
# %% --- FIGURE 1: profiles of the three differences, per Q ---
# =============================================================================
row_spec = [
    ('d_wl_peak', 'thr_w', 'ceiling change\n$\\Delta WL_{p%d}$ [m]' % WL_PERCENTILE, 'x_A'),
    ('d_bar',     'thr_b', f'bar change\n$\\Delta p_{{{BAR_PERCENTILE}}}$ [m]',   'x_B'),
    ('d_acc',     'thr_a', 'accommodation change\n$\\Delta A$ [m]',               None),
]
q_list = [q for q in DISCHARGES if q in profiles]
if q_list:
    fig, axes = plt.subplots(len(row_spec), len(q_list), sharex=True, sharey='row', squeeze=False,
                             figsize=(FIGURE_WIDTH_MM * MM_TO_IN, 120 * MM_TO_IN), constrained_layout=True)
    letters = iter('abcdefghijkl')
    for c, discharge in enumerate(q_list):
        p = profiles[discharge]
        sub = [r for r in p['runs'] if r['n'] == PROFILE_N]
        for k, (key, thr_key, ylabel, limit) in enumerate(row_spec):
            ax = axes[k, c]
            ax.fill_between(X_KM, -p[thr_key], p[thr_key], color='0.55', alpha=0.25, lw=0, zorder=1)
            ax.axhline(0, color=GREY_CONST, lw=0.8, ls='--', zorder=2)
            for r in sub:
                col = PM_COLOR[r['pm']]
                ax.plot(X_KM, r[key], color=col, lw=1.4, zorder=3)
                if key == 'd_wl_peak':           # same scenario at mean flow: morphology effect only
                    ax.plot(X_KM, r['d_wl_mean'], color=col, lw=0.8, ls=':', zorder=3)
                if limit == 'x_A':
                    mark_limit(ax, r['x_A'], col, filled=False)
                elif limit == 'x_B':
                    mark_limit(ax, r['x_B'], col, filled=True)
                    mark_limit(ax, r['x_A'], col, filled=False)
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.95, f'({next(letters)})', transform=ax.transAxes, va='top', fontweight='bold')
            if c == 0:
                ax.set_ylabel(ylabel)
            if k == 0:
                ax.set_title(f'$Q_{{\\mathrm{{mean}}}}$ = {discharge} m$^3$/s', pad=8)
            if k == len(row_spec) - 1:
                ax.set_xlabel('distance along estuary [km]  (sea \u2190 \u2192 river)')
            strip_top_right_spines(ax)
    axes[0, 0].set_xlim(X_RANGE_M[0] / 1000, X_RANGE_M[1] / 1000)
    handles = [Patch(facecolor='0.55', alpha=0.4, label='threshold (constant run)'),
               Line2D([], [], color='0.3', lw=1.4, label='peak flow'),
               Line2D([], [], color='0.3', lw=0.8, ls=':', label='mean flow'),
               Line2D([], [], marker='v', ls='', color='k', mfc='white', label='$x_A$'),
               Line2D([], [], marker='v', ls='', color='k', label='$x_B$')]
    handles += [Line2D([], [], color=PM_COLOR[pm], lw=1.4, label=f'$R_{{\\mathrm{{peak}}}}$ = {pm}')
                for pm in all_pm]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, frameon=False)
    save_figure(fig, 'bar_accommodation_profiles')


# =============================================================================
# %% --- FIGURE 2: the limits vs R_peak and V ---
# =============================================================================
if not summary.empty:
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM_TO_IN, 130 * MM_TO_IN),
                             constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for discharge in q_list:
        col, mk = DISCHARGE_COLORS[discharge], DISCHARGE_MARKERS[discharge]
        sub = summary[summary.discharge == discharge]
        kw = dict(marker=mk, ls='', ms=5, color=col, mec='k', mew=0.4, zorder=4)

        # (a) both limits vs R_peak (small offset so x_A and x_B do not overlap)
        ax_a.plot(sub.pm - 0.08, sub.x_A_km, mfc='white', **{**kw, 'mec': col})
        ax_a.plot(sub.pm + 0.08, sub.x_B_km, **kw)
        if 'LTI_instant_max_km' in sub:
            ax_a.plot(sub.pm, sub.LTI_instant_max_km, marker='x', ls='', ms=4, color=col, zorder=3)

        # (b) prediction x_B = x_A
        ax_b.plot(sub.x_A_km, sub.x_B_km, **kw)
        # (c) does supply move the limit?  (d) does supply set the amplitude?
        ax_c.plot(sub.V_ratio, sub.x_B_km, **kw)
        ax_d.plot(sub.V_ratio, sub.dp95_upstream_m, **kw)
        for _, r in sub.iterrows():
            label_point(ax_b, r.x_A_km, r.x_B_km, r, 'k')
            label_point(ax_c, r.V_ratio, r.x_B_km, r, 'k')
            label_point(ax_d, r.V_ratio, r.dp95_upstream_m, r, 'k')

    lims = [X_RANGE_M[0] / 1000, X_RANGE_M[1] / 1000]
    ax_b.plot(lims, lims, color='k', lw=0.6, zorder=1)
    ax_b.set_xlim(lims)
    ax_b.set_ylim(lims)
    ax_b.set_aspect('equal')
    ax_d.axhline(0, color=GREY_CONST, lw=0.6)

    x_a_label = '$x_A$ [km]' + (' (ceiling)' if X_A_FROM == 'ceiling' else ' (accommodation)')
    ax_a.set_xlabel('discharge amplitude $R_{\\mathrm{peak}}$')
    ax_a.set_ylabel('limit position [km]')
    ax_b.set_xlabel(x_a_label)
    ax_b.set_ylabel('$x_B$ [km]')
    ax_c.set_xlabel('sediment supply $V / V_{\\mathrm{const}}$ [-]')
    ax_c.set_ylabel('$x_B$ [km]')
    ax_d.set_xlabel('sediment supply $V / V_{\\mathrm{const}}$ [-]')
    ax_d.set_ylabel(f'mean $\\Delta p_{{{BAR_PERCENTILE}}}$, km {UPSTREAM_REACH_KM[0]:g}'
                    f'\u2013{UPSTREAM_REACH_KM[1]:g} [m]')
    for ax, letter in zip(axes.flat, 'abcd'):
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.97, f'({letter})', transform=ax.transAxes, va='top', fontweight='bold')
        strip_top_right_spines(ax)

    handles = [Line2D([], [], marker=DISCHARGE_MARKERS[q], ls='', ms=5, color=DISCHARGE_COLORS[q],
                      mec='k', mew=0.4, label=f'Q = {q} m$^3$/s') for q in q_list]
    handles += [Line2D([], [], marker='o', ls='', ms=5, color='0.4', mfc='white', label='$x_A$'),
                Line2D([], [], marker='o', ls='', ms=5, color='0.4', mec='k', mew=0.4, label='$x_B$')]
    if 'LTI_instant_max_km' in summary:
        handles.append(Line2D([], [], marker='x', ls='', ms=4, color='0.4', label='$L_{TI}$'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False)
    fig.text(0.99, 0.005, 'labels: $R_{\\mathrm{peak}}$/$n_{\\mathrm{peaks}}$', ha='right', va='bottom',
             fontsize=6, color='0.4')
    save_figure(fig, 'bar_accommodation_limits')

print(f'\nAll outputs written to: {OUTPUT_DIR}')

# %%
