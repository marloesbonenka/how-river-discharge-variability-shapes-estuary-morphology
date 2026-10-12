"""3D view of detrended bed level (x, y, z=bed level) per scenario.

For a specific snapshot in time, the detrended bed level (mesh2d_mor_bl) is
shown as a 3D surface over the (x, y) plane, using the same t=0 centerline
detrending approach as postprocessing/plot_snapshot_comparison_bedlevel_cross_sections.py.
One figure is produced per scenario.
"""

#%% IMPORTS
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import (
    get_variability_map,
    get_snapshot_matches_by_target_dates,
    apply_plot_style,
)

from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_morphological_activity import build_centerline_reference

# --- CONFIGURATION ---
DISCHARGE = 500
TARGET_DATE = np.datetime64('2031-01-01')
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True
Y_RANGE = (5000, 10000)

# Region of the domain shown as a 3D surface [m]
PLOT_X_RANGE = (20000, 45000)

# Detrending settings (mirrors postprocessing/plot_map_variable.py)
REFERENCE_TIME_IDX = 0
DETREND_LAND_THRESHOLD = 6.0
CENTERLINE_XMIN = 20000     # [m] start of x-range for the reference profile
CENTERLINE_XMAX = 45000     # [m] end of x-range for the reference profile
CENTERLINE_Y = 7500         # [m] exact y-coordinate of the channel centerline

# Keep the surface responsive: randomly subsample faces above this count.
MAX_SURFACE_POINTS = 40000
RNG_SEED = 0

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170

apply_plot_style('AGU', font_size=10, title_delta=0, tick_delta=0, **{'grid.linewidth': 0.5})

#%%
# Output
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\comparison_amplitude_frequency_same_Vsed\cross_sections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME_PREFIX = "comparison_amplitude_frequency_same_Vsed_3d"

# Paths defined by you
MODEL_PATHS = {
    'pm3_n5': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\03_Qr500_pm3_n5.9600329"),
    'pm5_n1': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\05_Qr500_pm5_n1.9517572"),
}

# Directories for cache/helpers
base_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500")
assessment_dir = base_path / 'cached_data'
timed_out_dir = base_path / 'timed-out'

# --- DATA PROCESSING ---
# results[label] = {'x': ..., 'y': ..., 'bedlevel': ...}  (surface, possibly subsampled)
results = {}
rng = np.random.default_rng(RNG_SEED)

for label, folder_path in MODEL_PATHS.items():
    print(f"\nProcessing: {folder_path.name}")

    # 1. Path resolution logic
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_path.name,
        timed_out_dir=timed_out_dir,
        variability_map=get_variability_map(DISCHARGE),
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

    # 3. Build the t=0 centerline reference profile (same approach as plot_map_variable.py)
    reference_per_face = None
    if 'time' not in ds['mesh2d_mor_bl'].dims:
        print("  [WARNING] Cannot detrend mesh2d_mor_bl: no time dimension found.")
        ds.close()
        continue
    try:
        reference_per_face = build_centerline_reference(
            ds, 'mesh2d_mor_bl', REFERENCE_TIME_IDX,
            xmin=CENTERLINE_XMIN, xmax=CENTERLINE_XMAX,
            centerline_y=CENTERLINE_Y,
        )
    except ValueError as exc:
        print(f"  [WARNING] Could not build centerline reference: {exc}")
        ds.close()
        continue

    # 4. Extraction of the target snapshot
    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        print(f"  No snapshot found for {TARGET_DATE}")
        ds.close()
        continue

    _, ts_idx, _ = matches[0]
    raw_bed = ds['mesh2d_mor_bl'].isel(time=ts_idx).values
    face_x, face_y = _get_face_coords(ds)

    # 5. Detrend: mask land at the CURRENT timestep, then subtract the t=0
    #    centerline reference profile (interpolated per face by x).
    masked_bed = raw_bed.copy()
    masked_bed[raw_bed > DETREND_LAND_THRESHOLD] = np.nan
    detrended_bed = masked_bed - reference_per_face

    # 6. Restrict to the plotted region and drop NaNs (land / outside domain).
    region_mask = (
        (face_x >= PLOT_X_RANGE[0]) & (face_x <= PLOT_X_RANGE[1]) &
        (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1]) &
        ~np.isnan(detrended_bed)
    )
    surf_x = face_x[region_mask]
    surf_y = face_y[region_mask]
    surf_z = detrended_bed[region_mask]

    # Subsample for a responsive 3D render if there are too many faces.
    if surf_x.size > MAX_SURFACE_POINTS:
        keep_idx = rng.choice(surf_x.size, size=MAX_SURFACE_POINTS, replace=False)
        surf_x, surf_y, surf_z = surf_x[keep_idx], surf_y[keep_idx], surf_z[keep_idx]

    results[label] = {'x': surf_x, 'y': surf_y, 'bedlevel': surf_z}

    ds.close()

#%% --- PLOTTING ---
# One 3D figure per scenario: surface colored by bed level.

for label, data in results.items():
    match = re.search(r'pm(\d+)', label)
    pm_value = int(match.group(1)) if match else 0
    n_match = re.search(r'n(\d+)', label)
    n_value = int(n_match.group(1)) if n_match else 0

    fig = plt.figure(figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_WIDTH_MM * MM_TO_IN * 0.8))
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_trisurf(
        data['x'], data['y'], data['bedlevel'],
        cmap='terrain', linewidth=0, antialiased=True, alpha=0.9,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label='detrended bed level [m]')

    ax.set_xlabel('x-coordinate [m]')
    ax.set_ylabel('y-coordinate [m]')
    ax.set_zlabel('detrended bed level [m]')
    ax.set_title(
        rf'$\text{{Q}}_{{mean}} = {DISCHARGE} \text{{ m³/s}}$, '
        rf'$\text{{R}}_{{peak}} = {pm_value}$, $\text{{n}}_{{peaks}} = {n_value}$'
    )
    ax.view_init(elev=25, azim=-60)

    OUTPUT_PATH = OUTPUT_DIR / f"{OUTPUT_FILENAME_PREFIX}_{label}"
    plt.savefig(OUTPUT_PATH.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PATH.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {OUTPUT_PATH.with_suffix('.png')} and {OUTPUT_PATH.with_suffix('.pdf')}")

plt.show()
# %%
