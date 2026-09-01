"""Analyze effect of MORFAC on the model results. This script is used to create the figures in the appendix of the paper."""
#%% 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from appendix/

from functions.F_general import apply_plot_style, setup_variability_run_context
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_loaddata import get_stitched_map_run_paths

#%%
# =============================================================================
# 0. CONFIGURATION  
# =============================================================================

# Plot style
STYLE = "AGU"
apply_plot_style(STYLE)

# Variables to analyze
var_names = ["mesh2d_mor_bl"] 

# Zoom settings
ZOOM = False
ZOOM_XLIM = (20000, 45000)   # x-range in model coordinates [m]
ZOOM_YLIM = (5000, 10000)    # y-range in model coordinates [m]

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000] # xmin, ymin, xmax, ymax
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

#%%
# =============================================================================
# 1. LOAD DATA
# =============================================================================

BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")

run_context = setup_variability_run_context(BASE_DIR, 
                                            discharge=500,
                                            MORFAC = True)

base_path = run_context['base_path']
assessment_dir = run_context['cache_dir']
timed_out_dir = run_context['timed_out_dir']
model_folders = run_context['model_folders']

configs = {
    'mesh2d_mor_bl': {
        'cmap': cmocean.cm.delta, 
        'vmin': -5,
        'vmax': 5,
        'label': 'bed level [m]',
        'file_tag': 'bedlevel_map'
    },
    }
#%%
# =============================================================================
# 2. PROCESSING LOOP
# =============================================================================

for folder in model_folders:
    model_location = base_path / folder
    _zoom_subfolder = f"zoom" if ZOOM else ""
    output_plots_dir = base_path / 'output_plots' / 'map_plots' / STYLE / _zoom_subfolder if ZOOM else base_path / 'output_plots' / 'map_plots' / STYLE
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {folder.name}")

    # Check for timed-out runs
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [model_location]

    # Load data
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=var_names,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        print(f"Skipping {folder.name}: no data cached.")
        continue

    try:
        # Plot data
        print("no plots yet")

    finally:
        ds.close()

print("\n" + "="*30)
print("PLOTTING COMPLETE")
print("="*30)