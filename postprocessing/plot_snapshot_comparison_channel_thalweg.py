"""Compare channel thalweg along the estuary for the same sediment import,
but either a large peak or frequent small peaks in river discharge.

For each 1 km bin, the thalweg is first calculated per grid column
(cross-section) within the frozen t=0 channel footprint:
    column thalweg = mean of the N_DEEPEST deepest bed cells
The plotted bin value is the median of these column thalweg values.
"""


#%% IMPORTS
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


from functions.F_general import (
    get_variability_map,
    get_snapshot_matches_by_target_dates,
    apply_plot_style,
)


from functions.F_map_cache import (
    cache_tag_from_bbox,
    load_or_update_map_cache_multi,
    _get_face_coords,
)
from functions.F_loaddata import get_stitched_map_run_paths


#%% --- CONFIGURATION ---
DISCHARGE = 500
TARGET_DATE = np.datetime64('2031-01-01')


CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True


X_RANGE = (20000, 44000)
Y_RANGE = (5000, 10000)
DX_BIN = 1000


# Thalweg definition
CHANNEL_INIT_THRESHOLD = 2.2   # frozen t=0 channel footprint
N_DEEPEST = 1                  # mean of N deepest cells per grid column
DX_COL = 85                    # approximate grid-cell size in x [m]


# Scenarios and colours: Blues for R_peak, as in the percentile script
MODEL_PATHS = {
    'pm3_n5': Path(
        r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\03_Qr500_pm3_n5.9600329"
    ),
    'pm5_n1': Path(
        r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\05_Qr500_pm5_n1.9517572"
    ),
}


all_pm_vals = sorted([0, 2, 3, 4, 5])
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}


# --- AGU figure style ---
apply_plot_style('AGU', font_size=10, title_delta=0, tick_delta=0,
                 **{'grid.linewidth': 0.5})


#%% --- OUTPUT ---
OUTPUT_DIR = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Output\Q500\detailed-hydro-run\output_plots"
    r"\comparison_amplitude_frequency_same_Vsed"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME = "comparison_thalweg_amplitude_frequency_same_Vsed"


#%% --- HELPERS ---


def build_column_layout(face_x, bin_mask, dx_col):
    """Return sorted face indices and column start indices for one x-bin.

    Grid columns are identified from jumps in sorted face x coordinates.
    """
    idx = np.flatnonzero(bin_mask)
    if idx.size == 0:
        return None
    idx = idx[np.argsort(face_x[idx], kind='stable')]
    starts = np.r_[0, np.flatnonzero(np.diff(face_x[idx]) > 0.5 * dx_col) + 1]
    return idx, starts


def bin_thalweg(bl, layout, n_deepest):
    """Median over grid columns of the mean of the n deepest cells per column."""
    if layout is None:
        return np.nan
    idx, starts = layout
    z = bl[idx]
    z = z[~np.isnan(z)]
    if z.size == 0:
        return np.nan


    column_thalweg = []
    for c in np.split(z, starts[1:]):
        c = c[~np.isnan(c)]
        if c.size == 0:
            continue
        column_thalweg.append(np.sort(c)[:n_deepest].mean())


    if not column_thalweg:
        return np.nan
    return np.median(np.asarray(column_thalweg))


#%% --- DATA PROCESSING ---
base_path = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Output\Q500"
)
assessment_dir = base_path / 'cached_data'
timed_out_dir = base_path / 'timed-out'


x_bins = np.arange(X_RANGE[0], X_RANGE[1] + DX_BIN, DX_BIN)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2


results = {}


for label, folder_path in MODEL_PATHS.items():
    print(f"\nProcessing: {folder_path.name}")


    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_path.name,
        timed_out_dir=timed_out_dir,
        variability_map=get_variability_map(DISCHARGE),
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]


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


    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        print(f"  No snapshot found for {TARGET_DATE}")
        ds.close()
        continue


    _, ts_idx, _ = matches[0]
    bedlev = ds['mesh2d_mor_bl'].isel(time=ts_idx).values
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values
    face_x, face_y = _get_face_coords(ds)


    # Frozen channel footprint from t=0
    footprint = (
        (face_y >= Y_RANGE[0])
        & (face_y <= Y_RANGE[1])
        & (init_bl < CHANNEL_INIT_THRESHOLD)
    )


    thalweg_profile = []
    for k in range(len(x_bins) - 1):
        bin_mask = footprint & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
        layout = build_column_layout(face_x, bin_mask, DX_COL)
        thalweg_profile.append(bin_thalweg(bedlev, layout, N_DEEPEST))


    results[label] = {
        'x': x_centers / 1000,
        'thalweg': np.asarray(thalweg_profile),
    }
    ds.close()


#%% --- PLOTTING ---
fig, ax = plt.subplots()


for label, data in results.items():
    pm_match = re.search(r'pm(\d+)', label)
    n_match = re.search(r'n(\d+)', label)
    pm_value = int(pm_match.group(1)) if pm_match else 0
    n_value = int(n_match.group(1)) if n_match else 0


    color = PM_COLOR[pm_value]
    x = data['x']
    y = data['thalweg']


    ax.plot(x, y, color=color, linewidth=2, zorder=3)


    # Direct annotation at the right end of each line
    valid = ~np.isnan(y)
    if valid.any():
        x_end = x[valid][-1]
        y_end = y[valid][-1]
        ax.annotate(
            fr'$R_{{\mathrm{{peak}}}} = {pm_value}$' + '\n'
            + fr'$n_{{\mathrm{{peaks}}}} = {n_value}$',
            xy=(x_end, y_end),
            xycoords='data',
            xytext=(4, 0),
            textcoords='offset points',
            va='center',
            ha='left',
            color=color,
            annotation_clip=False,
        )


ax.set_xlabel('along-estuary distance [km]')
ax.set_ylabel(f'median column thalweg bed level [m]\n(mean of {N_DEEPEST} deepest cells per column)')
ax.set_title(rf'$Q_{{\mathrm{{mean}}}} = {DISCHARGE}\ \mathrm{{m^3/s}}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--')


#%% --- SAVE ---
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME
plt.savefig(OUTPUT_PATH.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_PATH.with_suffix('.pdf'), bbox_inches='tight')


print(f"\nPlot saved to:\n  {OUTPUT_PATH.with_suffix('.png')}\n  {OUTPUT_PATH.with_suffix('.pdf')}")
plt.show()