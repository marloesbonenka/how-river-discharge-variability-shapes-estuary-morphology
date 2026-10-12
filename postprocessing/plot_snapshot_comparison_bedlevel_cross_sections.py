"""Compare detrended bed level cross-sections of two scenarios with the same amount of
sediment import, but either by a large peak or frequent small peaks in river discharge.

For a specific snapshot in time, the bed level (mesh2d_mor_bl) is sampled along a few
across-estuary cross-sections (fixed x), detrended using the same t=0 centerline
reference profile approach as in postprocessing/plot_map_variable.py, and plotted as
bed level vs. y-coordinate. One figure is produced per cross-section, each containing
both scenarios."""

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
Y_RANGE = (0, 15000)

# Cross-section locations along the estuary [m]
CROSS_SECTION_X_KM = np.linspace(25, 45, 21)
X_MATCH_TOL = 1.0  # [m] tolerance for matching a face column to a target x

# Percentile metric settings (mirrors postprocessing/plot_sensitivity_pm_n_bedlevel_percentiles.py):
# the horizontal line shows p{DEPTH_PERCENTILE} of the RAW bed level within a frozen t=0
# channel mask (same bed-level threshold approach + same y-window as that script), detrended
# by subtracting the t=0 p{DEPTH_PERCENTILE} of the same mask -- NOT a percentile of this
# cross-section's own detrended profile.
DEPTH_PERCENTILE = 5          # percentile of raw bed level within the frozen t=0 channel mask
CHANNEL_INIT_THRESHOLD = 2.2  # [m] t=0 bed level threshold defining the channel footprint
BIN_HALF_WIDTH = 500          # [m] half-width of the x-window around each cross-section (~1 km bin)
PERCENTILE_Y_RANGE = (5000, 10000)  # [m] y-window for the percentile subarea (matches y_range in that script)

# Detrending settings (mirrors postprocessing/plot_map_variable.py)
REFERENCE_TIME_IDX = 0
DETREND_LAND_THRESHOLD = 6.0
CENTERLINE_XMIN = 20000     # [m] start of x-range for the reference profile
CENTERLINE_XMAX = 45000     # [m] end of x-range for the reference profile
CENTERLINE_Y = 7500         # [m] exact y-coordinate of the channel centerline

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170 / 2  # single-column-ish width, matches the percentile comparison plot

apply_plot_style('AGU', font_size=10, title_delta=0, tick_delta=0, **{'grid.linewidth': 0.5})

#%%
# Output
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\comparison_amplitude_frequency_same_Vsed\cross_sections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME_PREFIX = "comparison_amplitude_frequency_same_Vsed_cross_section"

all_pm_vals = sorted([0, 2, 3, 4, 5])  # pm values to compare
# Colormaps: Blues for pm (light->dark) -- same colors as plot_snapshot_comparison_bedlevel_percentiles.py
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}

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
# results[label][x_km] = {'y': ..., 'bedlevel': ...}
results = {}

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
    init_bed = ds['mesh2d_mor_bl'].isel(time=0).values  # raw t=0 bed level, for the frozen channel mask
    face_x, face_y = _get_face_coords(ds)

    # 5. Detrend: mask land at the CURRENT timestep, then subtract the t=0
    #    centerline reference profile (interpolated per face by x).
    masked_bed = raw_bed.copy()
    masked_bed[raw_bed > DETREND_LAND_THRESHOLD] = np.nan
    detrended_bed = masked_bed - reference_per_face

    # 6. Sample cross-sections: pick the face column nearest to each target x.
    results[label] = {}
    y_mask = (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1])
    percentile_y_mask = (face_y >= PERCENTILE_Y_RANGE[0]) & (face_y <= PERCENTILE_Y_RANGE[1])
    for x_km in CROSS_SECTION_X_KM:
        target_x = x_km * 1000
        nearest_x_value = face_x[np.argmin(np.abs(face_x - target_x))]
        col_mask = y_mask & np.isclose(face_x, nearest_x_value, atol=X_MATCH_TOL)

        y_col = face_y[col_mask]
        z_col = detrended_bed[col_mask]
        order = np.argsort(y_col)

        # 7. p{DEPTH_PERCENTILE} bed level metric, computed the same way as in
        #    plot_sensitivity_pm_n_bedlevel_percentiles.py: percentile of RAW bed level
        #    within a frozen t=0 channel mask (PERCENTILE_Y_RANGE + ~1 km x-bin around this
        #    cross-section + init bed level below CHANNEL_INIT_THRESHOLD), detrended by
        #    subtracting the same percentile computed on the t=0 field.
        bin_mask = (
            percentile_y_mask
            & (face_x >= target_x - BIN_HALF_WIDTH) & (face_x < target_x + BIN_HALF_WIDTH)
            & (init_bed < CHANNEL_INIT_THRESHOLD)
        )
        p_detrended = np.nan
        if np.any(bin_mask):
            init_vals = init_bed[bin_mask]
            init_vals = init_vals[~np.isnan(init_vals)]
            snap_vals = raw_bed[bin_mask]
            snap_vals = snap_vals[~np.isnan(snap_vals)]
            if init_vals.size and snap_vals.size:
                p_init = np.percentile(init_vals, DEPTH_PERCENTILE)
                p_snapshot = np.percentile(snap_vals, DEPTH_PERCENTILE)
                p_detrended = p_snapshot - p_init

        results[label][x_km] = {
            'y': y_col[order],
            'bedlevel': z_col[order],
            'p_detrended': p_detrended,
        }

    ds.close()

#%% --- PLOTTING ---
# One figure per cross-section, with both scenarios shown together.

def _right_label(ax, x, profile, text, color, dy=0):
    valid = ~np.isnan(np.asarray(profile))
    if not valid.any():
        return
    x_end = np.asarray(x)[valid][-1]
    y_end = np.asarray(profile)[valid][-1]
    ax.annotate(text,
                xy=(x_end, y_end), xycoords='data',
                xytext=(4, dy), textcoords='offset points',
                va='center', ha='left', color=color,
                annotation_clip=False)


for x_km in CROSS_SECTION_X_KM:
    fig, ax = plt.subplots()

    for label, data_by_x in results.items():
        if x_km not in data_by_x:
            continue
        data = data_by_x[x_km]

        match = re.search(r'pm(\d+)', label)
        pm_value = int(match.group(1)) if match else 0
        n_match = re.search(r'n(\d+)', label)
        n_value = int(n_match.group(1)) if n_match else 0

        color = PM_COLOR[pm_value]

        y = data['y']
        bedlevel = data['bedlevel']

        ax.plot(y, bedlevel, color=color, linewidth=2, zorder=2)
        p_detrended = data['p_detrended']
        if not np.isnan(p_detrended):
            ax.axhline(p_detrended, color=color, linestyle='--', linewidth=1.2, zorder=1)
            # Anchor the label to the same x-position as the curve's own end label
            # (last non-NaN bedlevel point), not the full y-range of the cross-section.
            p5_profile = np.where(np.isnan(bedlevel), np.nan, p_detrended)
            _right_label(ax, y, p5_profile, 'p5 bed level', color)

        label2_text = fr'$\text{{R}}_{{peak}} = {pm_value}$'
        label3_text = fr'$\text{{n}}_{{peaks}}= {n_value}$'
        _right_label(ax, y, bedlevel, f'{label2_text}\n{label3_text}', color)

    # ax.set_xlim(5000, 10000)
    ax.set_xlabel('y-coordinate [m]')
    ax.set_ylabel('detrended bed level [m]')
    ax.set_title(rf'$\text{{Q}}_{{mean}} = {DISCHARGE} \text{{ m³/s}}$, x = {x_km} km')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--')

    OUTPUT_PATH = OUTPUT_DIR / f"{OUTPUT_FILENAME_PREFIX}_x{x_km}km"
    plt.savefig(OUTPUT_PATH.with_suffix('.png'), dpi=300, bbox_inches='tight')
    # plt.savefig(OUTPUT_PATH.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {OUTPUT_PATH.with_suffix('.png')} and {OUTPUT_PATH.with_suffix('.pdf')}")

plt.show()
# %%
