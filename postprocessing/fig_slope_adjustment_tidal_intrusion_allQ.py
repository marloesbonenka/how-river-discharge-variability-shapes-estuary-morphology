"""Fluvial slope adjustment along the estuary: where does the bed steepen with R_peak?
One panel per discharge (Q = 250 / 500 / 1000 m3/s: tide- to river-dominated), single snapshot.

Hypothesis (fluvial analogue, Exner): discharge variability raises the sediment supply and
the accommodation space, so the bed aggrades most at the river end and the profile steepens
instead of the channels deepening.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one row:
  Local mean-bed slope along the estuary: least-squares line through the mean bed level per
  1-km bin over a SLOPE_WINDOW_KM moving window (positive = bed rising towards the river).
  The slope between neighbouring 1-km bins alone is dominated by bars and channels (the
  console prints the autogenic band width of both). Dotted line: tidal intrusion limit of
  the constant run. Runs: the n_peaks = 3 series (R_peak 2-5) plus the constant run, the
  only series run at all discharges. Colours: R_peak value, Blues palette (light -> dark).
  The bed-level profiles themselves are in fig_bedlevel_percentiles_pm_n.py.

Bed-level population per x-bin: MASK_MODE (default 'union': faces below
CHANNEL_INIT_THRESHOLD at t=0 OR at the snapshot); see functions/F_bedlevel_percentiles.py.
Grey band: autogenic variability of the slope from the noisy repeats of the constant run
plus the constant run itself: +/-2 sigma with >= ENVELOPE_MIN_RUNS reference runs, else
their min-max range. Only Q = 500 has a band; the Q = 250 and Q = 1000 noisy repeats are
unfinished and left out.

Tidal intrusion limits are read from LTI_CSV, written by
plot_map_hydro_tidalmetrics_comparison_allQ.py (same metric as elsewhere in the paper).
"""
#%% --- Imports ---
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
)
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_bedlevel_percentiles import bedlevel_profiles

#%% --- 1. SETTINGS ---
DISCHARGES = [250, 500, 1000]   # tide- to river-dominated
RUN_IDS = [1, 6, 9, 10, 11]     # constant + n_peaks = 3 series, run at all discharges
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

# nearest available timestep per run is used; optional override from the command line, e.g.
# `python fig_slope_adjustment_tidal_intrusion_allQ.py 2030-01-01`. The detailed-hydro runs
# (tidal intrusion) start from the 2031-01-01 morphology.
TARGET_DATE = np.datetime64(sys.argv[1] if len(sys.argv) > 1 else '2031-01-01')

MASK_MODE = 'union'            # 'frozen' | 'union' (see functions/F_bedlevel_percentiles.py)
CHANNEL_INIT_THRESHOLD = 2.2   # channel = bed below this [m]
X_BINS = np.arange(20000, 45000, 1000)   # 1 km bins, 20-44 km
X_KM = 0.5 * (X_BINS[:-1] + X_BINS[1:]) / 1000
y_range = (5000, 10000)
METRICS = ['mean']
SLOPE_WINDOW_KM = 5            # local slope = least-squares line over this window, centred on each bin
MIN_WINDOW_BINS = 3            # fewer valid bins in the window (e.g. at the ends) -> no slope
XTICK_STEP_KM = 5

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Tidal intrusion limit per run (columns: discharge, pm, n, LTI_instant_max_km)
LTI_CSV = BASE_DIR / 'output_plots_combined' / 'hydro_tidalmetrics_comparison' / 'LTI_summary.csv'

# Autogenic variability envelope — noisy repeats of the constant scenario, per discharge:
# (noise folder holding cached_data, [run folders relative to it])
# Only Q = 500: the Q = 250 and Q = 1000 noisy runs are unfinished, so those columns get no band.
SHOW_NOISY_ENVELOPE = True
ENVELOPE_MIN_RUNS = 3          # reference runs (noisy + constant) needed for a ±2σ band
NOISY_ROOT = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output")
NOISY_RUNS = {
    500: (NOISY_ROOT / 'Q500' / '0_Noise_Q500', [
        '1_Q500_noisy0.9095347',
        '1_Q500_noisy1_rst.9160657',
        '1_Q500_noisy2_rst.9160663',
    ]),
}

SCENARIO_LABELS = {
    '1':  'pm1_n0 (constant)',
    '6':  'pm4_n3',
    '9':  'pm5_n3',
    '10': 'pm3_n3',
    '11': 'pm2_n3',
}

RPEAK  = r'$R_{\mathrm{peak}}$'

GREY_CONST = "#7f7f7f"
ENV_COLOR, ENV_ALPHA = '0.55', 0.25
LINE_WIDTH = 1.0         # scenario lines (plot and legend)
LINE_WIDTH_CONST = 0.9   # constant reference line (plot and legend)

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170   # AGU full-page width
FIGURE_HEIGHT_MM = 62
FONT_SIZE = 8           # one size for all text: ticks, labels, titles, legend

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'slope_adjustment_tidal_intrusion_allQ'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#%% --- 2. HELPERS ---
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


def _load_profiles(cache_dir, folder_name, run_paths):
    """Snapshot profiles of one run via the map cache (None if unavailable)."""
    ds = load_or_update_map_cache_multi(
        cache_dir=cache_dir, folder_name=folder_name, run_paths=run_paths,
        var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS, append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )
    if ds is None:
        print(f"  No cached data for {folder_name}, skipping.")
        return None
    profs, actual_dt = _snapshot_profiles(ds)
    ds.close()
    if profs is None:
        print(f"  No timesteps found for {folder_name}, skipping.")
        return None
    print(f"  {folder_name}: snapshot {actual_dt}")
    return profs


def local_slope(y, window_km=SLOPE_WINDOW_KM):
    """Slope [m/km] per x-bin: least-squares line through the bins within window_km / 2 of the
    bin centre (NaN bins skipped). window_km = 1 gives the slope between neighbouring bins."""
    out = np.full(X_KM.size, np.nan)
    half = max(window_km / 2, 1.0)            # at least the two neighbours
    for i, xc in enumerate(X_KM):
        ok = (np.abs(X_KM - xc) <= half + 1e-9) & np.isfinite(y)
        if np.count_nonzero(ok) < (2 if window_km <= 1 else MIN_WINDOW_BINS):
            continue
        x_dev = X_KM[ok] - X_KM[ok].mean()
        out[i] = np.sum(x_dev * (y[ok] - y[ok].mean())) / np.sum(x_dev ** 2)
    return out


def _band(values):
    """Autogenic band of a stack of values or profiles (axis 0): ±2σ, or min-max range
    when there are fewer than ENVELOPE_MIN_RUNS."""
    stk = np.asarray(values)
    if len(stk) >= ENVELOPE_MIN_RUNS:
        mean, std = np.nanmean(stk, axis=0), np.nanstd(stk, axis=0)
        return mean - 2 * std, mean + 2 * std
    return np.nanmin(stk, axis=0), np.nanmax(stk, axis=0)


def _fmt(v):
    return str(int(v)) if v == int(v) else str(v)


#%% --- 3. LOAD MODEL DATA (single snapshot, constant + n_peaks = 3 runs, all discharges) ---
profiles = {}         # {Q: {pm: {metric: array per x-bin}}}
noisy_profiles = {}   # {Q: [{metric: array per x-bin}, ...]}

for Q in DISCHARGES:
    base_path = BASE_DIR / f"Q{Q}"
    cache_dir = base_path / 'cached_data'
    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles[Q] = {}

    model_folders = find_variability_model_folders(
        base_path=base_path, discharge=Q, scenarios_to_process=RUN_IDS, analyze_noisy=False,
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
        profs = _load_profiles(cache_dir, folder.name, run_paths)
        if profs is not None:
            profiles[Q][pm] = profs

    noisy_profiles[Q] = []
    if SHOW_NOISY_ENVELOPE and Q in NOISY_RUNS:
        noise_dir, subfolders = NOISY_RUNS[Q]
        noisy_cache_dir = noise_dir / 'cached_data'
        noisy_cache_dir.mkdir(parents=True, exist_ok=True)
        for sub in subfolders:
            run_folder = noise_dir / sub
            if not run_folder.exists():
                print(f"[WARNING] Noisy run not found: {run_folder}")
                continue
            print(f"Loading noisy run Q{Q}: {run_folder.name}")
            try:
                profs = _load_profiles(noisy_cache_dir, run_folder.name, [run_folder])
            except OSError as exc:   # e.g. network drive dropping out while building the cache
                print(f"[WARNING] Noisy run Q{Q} {run_folder.name} not loaded ({exc}); "
                      f"rerun to retry, a finished cache is reused.")
                continue
            if profs is not None:
                noisy_profiles[Q].append(profs)

all_pm_vals = sorted({pm for Q in DISCHARGES for pm in profiles[Q] if pm > 1})
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}
PM_COLOR[1] = GREY_CONST


#%% --- 4. TIDAL INTRUSION LIMIT OF THE CONSTANT RUN (from LTI_CSV) ---
lti_const = {}   # {Q: limit [km]}
if LTI_CSV.exists():
    lti_df = pd.read_csv(LTI_CSV)
    lti_const = lti_df[lti_df.n == 0].set_index('discharge')['LTI_instant_max_km'].to_dict()
else:
    print(f"[WARNING] Tidal intrusion summary not found: {LTI_CSV} "
          f"(run plot_map_hydro_tidalmetrics_comparison_allQ.py)")


#%% --- 5. LOCAL SLOPES ALONG THE ESTUARY + AUTOGENIC BANDS ---
slopes = {Q: {pm: local_slope(p['mean']) for pm, p in profiles[Q].items()} for Q in DISCHARGES}
slope_env = {}   # {Q: (lo, hi) per x-bin}
for Q in DISCHARGES:
    const = profiles[Q].get(1)
    if noisy_profiles[Q] and const is not None:
        slope_env[Q] = _band([local_slope(p['mean']) for p in noisy_profiles[Q] + [const]])


#%% --- 6. SUMMARY (console) ---
print(f"\nMean-bed slope [m/km] ({MASK_MODE}), {TARGET_DATE}, {SLOPE_WINDOW_KM:g}-km moving window")
for Q in DISCHARGES:
    const = slopes[Q].get(1)
    if const is None:
        continue
    river = X_KM > lti_const.get(Q, np.nan)
    tidal = X_KM < lti_const.get(Q, np.nan)
    vals = ", ".join(f"R{_fmt(pm)} {np.nanmean((s - const)[river]):+.3f}/{np.nanmean((s - const)[tidal]):+.3f}"
                     for pm, s in sorted(slopes[Q].items()) if pm > 1)
    print(f"  Q = {Q}: change vs constant, mean landward/seaward of the tidal limit "
          f"({lti_const.get(Q, np.nan):.2f} km): {vals}")
# Why a moving window: width of the autogenic band for a 1-km slope vs the moving window
for Q in slope_env:
    refs = noisy_profiles[Q] + [profiles[Q][1]]
    for window in (1, SLOPE_WINDOW_KM):
        lo, hi = _band([local_slope(p['mean'], window) for p in refs])
        print(f"  Q = {Q}: mean autogenic band width, {window:g}-km slope: {np.nanmean(hi - lo):.3f} m/km")


#%% --- 7. FIGURE: mean-bed slope along the estuary, one panel per discharge ---
fig, axes = plt.subplots(1, len(DISCHARGES), sharex=True, sharey=True, squeeze=False,
                         figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
                         constrained_layout=True)
axes = axes[0]
for c, (Q, ax) in enumerate(zip(DISCHARGES, axes)):
    if SHOW_NOISY_ENVELOPE and Q in slope_env:
        ax.fill_between(X_KM, *slope_env[Q], color=ENV_COLOR, alpha=ENV_ALPHA, linewidth=0, zorder=1)
    if Q in lti_const:
        ax.axvline(lti_const[Q], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle=':', zorder=2)
    if 1 in slopes[Q]:
        ax.plot(X_KM, slopes[Q][1], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--', zorder=3)
    for pm_val in all_pm_vals:
        if pm_val in slopes[Q]:
            ax.plot(X_KM, slopes[Q][pm_val], color=PM_COLOR[pm_val], linewidth=LINE_WIDTH, zorder=4)
    ax.set_xlim(X_BINS[0] / 1000, X_BINS[-1] / 1000)
    ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
    ax.set_xlabel('Along-estuary distance [km]')
    ax.set_title(f'$Q$ = {Q} m\u00b3/s')
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(-0.02, 1.02, f"({chr(ord('a') + c)})", transform=ax.transAxes, ha='right', va='bottom')
    if c == 0:
        ax.set_ylabel('Mean-bed slope [m km$^{-1}$]')


#%% --- 8. LEGEND + SAVE ---
legend_handles = []
if SHOW_NOISY_ENVELOPE and slope_env:
    legend_handles.append(mpatches.Patch(facecolor=ENV_COLOR, alpha=ENV_ALPHA, label='autogenic variability'))
legend_handles.append(mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                                    label='constant'))
legend_handles += [mlines.Line2D([], [], color=PM_COLOR[pm], linewidth=LINE_WIDTH, label=f'{RPEAK} = {_fmt(pm)}')
                   for pm in all_pm_vals]
legend_handles.append(mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle=':',
                                    label='tidal intrusion limit (constant)'))
fig.legend(handles=legend_handles, loc='outside lower center', ncol=int(np.ceil(len(legend_handles) / 2)),
           frameon=True)

save_name = f"AGU_slope_along_estuary_allQ_{TARGET_DATE}_{MASK_MODE}_w{SLOPE_WINDOW_KM:g}km"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%
