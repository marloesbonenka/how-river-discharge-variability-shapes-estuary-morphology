"""Compare bed level percentiles along the estuary for the same amount of sediment import,
but either by a large peak or frequent small peaks in river discharge.
The comparison is made for a specific snapshot in time, and the bed level is binned along the estuary."""

#%% IMPORTS
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
    get_variability_map,
    get_snapshot_matches_by_target_dates,
)

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# --- CONFIGURATION ---
DISCHARGE = 500
TARGET_DATE = np.datetime64('2031-01-01')
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True
X_RANGE = (20000, 44000)
Y_RANGE = (5000, 10000)
CHANNEL_INIT_THRESHOLD = 2.2

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170/2          # AGU full-page width, since we need 2 columns + colorbar
CBAR_WIDTH_FRACTION = 0.03     # fraction of total width reserved for the shared colorbar

AGU_RC = {
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 10,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Calibri',
    'mathtext.it': 'Calibri:italic',
    'mathtext.bf': 'Calibri:bold',

    # --- Line weights: avoid hairlines (AGU rejects anything under 0.5pt) ---
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
    'grid.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35,
    'ytick.minor.width': 0.35,

    # --- Keep text as editable text in vector exports (not outlined paths) ---
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    # --- Resolution / export ---
    'figure.dpi': 150,          # screen preview only
    'savefig.dpi': 300,         # within AGU's 300-600 ppi raster range
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(AGU_RC)


#%%
# Output
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\comparison_amplitude_frequency_same_Vsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME   = "comparison_amplitude_frequency_same_Vsed"

all_pm_vals = sorted([0, 2, 3, 4, 5])  # pm values to compare
# Colormaps: Blues for pm (light→dark)
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}

all_n_vals = sorted([1, 3, 5])  # n values to compare
# Colormaps: Greens for n (light→dark)
_n_n = max(len(all_n_vals) - 1, 1)
N_COLOR = {n: plt.cm.Greens(0.35 + 0.55 * i / _n_n) for i, n in enumerate(all_n_vals)}

# Paths defined by you
MODEL_PATHS = {
    'pm3_n5': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\03_Qr500_pm3_n5.9600329"),
    # 'pm4_n3': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\06_Qr500_pm4_n3.9517578"),
    'pm5_n1': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\05_Qr500_pm5_n1.9517572"),
   
}

all_pm_vals = sorted([0, 2, 3, 4, 5])
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}

# Directories for cache/helpers
base_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500")
assessment_dir = base_path / 'cached_data'
timed_out_dir = base_path / 'timed-out'

# --- DATA PROCESSING ---
results = {}

for label, folder_path in MODEL_PATHS.items():
    print(f"\nProcessing: {folder_path.name}")

    # 1. Path resolution logic
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_path.name,
        timed_out_dir=timed_out_dir,
        variability_map=get_variability_map(DISCHARGE), # Assuming this helper exists
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]

    # 2. Cache loading
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_path.name,
        run_paths=run_paths,
        var_names=['mesh2d_mor_bl'],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )

    if ds is None:
        print(f"  No cached data for {folder_path.name}, skipping.")
        continue

    # 3. Extraction
    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        print(f"  No snapshot found for {TARGET_DATE}")
        ds.close()
        continue
    
    _, ts_idx, _ = matches[0]
    bedlev = ds['mesh2d_mor_bl'].isel(time=ts_idx).values
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values
    face_x, face_y = _get_face_coords(ds)
    
    # Masking and Binning
    mask = (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1]) & (init_bl < CHANNEL_INIT_THRESHOLD)
    dx = 1000
    x_bins = np.arange(X_RANGE[0], X_RANGE[1] + dx, dx)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    
    p5_profile   = []
    mean_profile = []
    p95_profile  = []
    for k in range(len(x_bins) - 1):
        b_mask = mask & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
        vals = bedlev[b_mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            p5_profile.append(np.percentile(vals, 5))
            mean_profile.append(np.mean(vals))
            p95_profile.append(np.percentile(vals, 95))
        else:
            p5_profile.append(np.nan)
            mean_profile.append(np.nan)
            p95_profile.append(np.nan)

    results[label] = {
        'x':    x_centers / 1000,
        'p5':   np.array(p5_profile),
        'mean': np.array(mean_profile),
        'p95':  np.array(p95_profile),
    }
    ds.close()

#%% --- PLOTTING ---

fig, ax = plt.subplots()

for label, data in results.items():
    match = re.search(r'pm(\d+)', label)
    pm_value = int(match.group(1)) if match else 0
    n_match = re.search(r'n(\d+)', label)
    n_value = int(n_match.group(1)) if n_match else 0

    color = PM_COLOR[pm_value]

    x = data['x']

    # Shaded band: p5 (channel) to p95 (bar)
    # ax.fill_between(x, data['p5'], data['p95'], color=color, alpha=0.18, zorder=1)

    # p5 and p95 dashed lines
    ax.plot(x, data['p95'], color=color, linewidth=2, zorder=2)
    # ax.plot(x, data['p5'],  color=color, linewidth=1.2, linestyle=':',  zorder=2)

    # Mean as solid line
    # ax.plot(x, data['mean'], color=color, linewidth=2.0, linestyle='-', zorder=3)

    # --- Direct annotations at right end of each line ---
    def _right_label(profile, text, color, dy=0):
        valid = ~np.isnan(np.array(profile))
        if not valid.any():
            return
        x_end = x[valid][-1]
        y_end = np.array(profile)[valid][-1]
        ax.annotate(text,
                    xy=(x_end, y_end), xycoords='data',
                    xytext=(4, dy), textcoords='offset points',
                    va='center', ha='left', color=color,
                    annotation_clip=False)
   
    # Scenario name + line type on every line, for every scenario
    # _right_label(data['mean'], f'{label} (mean)', color)
    label_text = fr'$\text{{p95 (bar)}}$'  
    label2_text = fr'$\text{{R}}_{{peak}} = {pm_value}$'
    label3_text = fr'$\text{{n}}_{{peaks}}= {n_value}$'
    _right_label(data['p95'], f'{label2_text}\n{label3_text}', color)  
    # _right_label(data['p5'],   'p5 (channel)',     color)

ax.set_xlabel('along-estuary distance [km]')
ax.set_ylabel('p95 bed level [m]')
ax.set_title(rf'$\text{{Q}}_{{mean}} = {DISCHARGE} \text{{ m³/s}}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--')
# Save as PNG
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME
plt.savefig(OUTPUT_PATH.with_suffix('.png'), dpi=300, bbox_inches='tight')

# Save as PDF
plt.savefig(OUTPUT_PATH.with_suffix('.pdf'), bbox_inches='tight')

print(f"Plot saved to: {OUTPUT_PATH.with_suffix('.png')} and {OUTPUT_PATH.with_suffix('.pdf')}")
plt.show()
# %%
