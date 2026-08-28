"""Plot map output at a certain timestep"""
#%% 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

from functions.F_general import create_water_colormap, create_shear_stress_colormap, create_terrain_colormap
from functions.F_general import apply_plot_style, compute_map_figsize, setup_variability_run_context
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_morphological_activity import build_centerline_reference

#%% --- 1. SETTINGS ---
# Which scenarios to process (set to None or empty list for all)
SCENARIOS_TO_PROCESS = None #['1', '2', '3', '4']  # Use all scenarios
DISCHARGE = 1000
apply_detrending = True
ZOOM = True          # True → crop axes to ZOOM_XLIM / ZOOM_YLIM

# --- Figure style ---
STYLE = 'AGU'   # 'default'  →  white background, black text/ticks/labels
                    # 'whitefig' →  transparent background, white text/ticks/labels
                    # 'AGU'

# --- AGU figure sizing (figures must be 50–170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 0.5*170   # full-width figure; use ~84 for a single-column figure
CBAR_WIDTH_FRACTION = 0.85  # fraction of total width reserved for the map itself (rest = colorbar + label)

STYLES = {'default', 'whitefig', 'AGU'}
if STYLE not in STYLES:
    raise ValueError(f"Unknown STYLE '{STYLE}'. Choose one of {sorted(STYLES)}.")

apply_plot_style(STYLE, font_size=8)
if STYLE == 'AGU':
    plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

# helper for per-element color (colorbar ticks etc.)
_tc = plt.rcParams['text.color']

# --- Variable selection ---
var_names = ['mesh2d_mor_bl']#, 'mesh2d_s1', 'mesh2d_taus']  # e.g. ['mesh2d_mor_bl'] or all three
target_hydrodynamic_date = '2031-01-01' #'2055-12-31' # e.g. '2055-12-31'; when set, nearest timestep is used per run

# Detrending settings (applies to bed level variable only)
reference_time_idx = 0
detrend_land_threshold = 6.0

# Centerline reference profile (used for detrending): for each x in
# [CENTERLINE_XMIN, CENTERLINE_XMAX], the t=0 reference value is the bed
# level of the face closest to y = CENTERLINE_Y at that x, interpolated
# per-face by x.
CENTERLINE_XMIN = 20000          # [m] start of x-range for the reference profile
CENTERLINE_XMAX = 45000          # [m] end of x-range for the reference profile
CENTERLINE_Y = 7500               # [m] exact y-coordinate of the channel centerline

# Zoom settings
ZOOM_XLIM = (20000, 45000)   # x-range in model coordinates [m]
ZOOM_YLIM = (5000, 10000)    # y-range in model coordinates [m]

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000] # xmin, ymin, xmax, ymax
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

#%%
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")

run_context = setup_variability_run_context(
    base_directory=base_directory,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)
base_path = run_context['base_path']
assessment_dir = run_context['cache_dir']
timed_out_dir = run_context['timed_out_dir']
VARIABILITY_MAP = run_context['variability_map']
model_folders = run_context['model_folders']

configs = {
    'mesh2d_mor_bl': {
        'cmap': cmocean.cm.delta, #create_terrain_colormap(),
        'vmin': -5,
        'vmax': 5,
        'label': 'bed level [m]',
        'file_tag': 'bedlevel_map'
    },
    'mesh2d_s1': {
        'cmap': create_water_colormap(),
        'vmin': -1,   # Adjust based on your tide/datum
        'vmax': 3,
        'label': 'water level [m]',
        'file_tag': 'water_level_map'
    },
    'mesh2d_taus': {
        'cmap': create_shear_stress_colormap(),
        'vmin': 0,
        'vmax': 5,    # Adjust based on flow intensity
        'label': 'bed shear stress [N/m²]',
        'file_tag': 'shear_stress_map'
    }
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

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [model_location]

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
        if 'time' not in ds.dims or len(ds.time) == 0:
            print(f"Skipping {folder.name}: no time dimension found.")
            continue

        time_values_full = np.asarray(ds.time.values).astype('datetime64[ns]')
        print(f"  Found {len(time_values_full)} timestep(s): {time_values_full[0]} -> {time_values_full[-1]}")

        # --- Build the t=0 centerline reference profile (once per folder) ---
        # This replaces the old single-scalar baseline with a profile that
        # varies along x, sampled at the reference timestep only.
        reference_per_face = None
        if apply_detrending and 'mesh2d_mor_bl' in ds:
            if 'time' not in ds['mesh2d_mor_bl'].dims:
                print("  [WARNING] Cannot detrend mesh2d_mor_bl: no time dimension found.")
            elif reference_time_idx >= len(time_values_full):
                print(
                    f"  [WARNING] reference_time_idx={reference_time_idx} out of range "
                    f"for {len(time_values_full)} timestep(s); skipping detrending."
                )
            else:
                try:
                    reference_per_face = build_centerline_reference(
                        ds, 'mesh2d_mor_bl', reference_time_idx,
                        xmin=CENTERLINE_XMIN, xmax=CENTERLINE_XMAX,
                        centerline_y=CENTERLINE_Y,
                    )
                except ValueError as exc:
                    print(f"  [WARNING] Could not build centerline reference: {exc}")
                    reference_per_face = None

        # --- Loop over all timesteps ---

        if target_hydrodynamic_date is not None:
            target_dt = np.datetime64(target_hydrodynamic_date, 'ns')
            time_diffs = np.abs(time_values_full - target_dt)
            nearest_idx = np.argmin(time_diffs)
            selected_indices = [nearest_idx]
            print(f"  Target date {target_hydrodynamic_date} → using nearest timestep: {time_values_full[nearest_idx]}")
        else:
            selected_indices = list(range(len(time_values_full)))

        for idx in range(len(selected_indices)):
            real_idx = selected_indices[idx]
            actual_dt = np.datetime64(time_values_full[real_idx], 'ns')
            actual_label = str(np.datetime_as_string(actual_dt, unit='s')).replace('T', ' ')
            actual_tag = str(np.datetime_as_string(actual_dt, unit='D'))
            print(f"  Plotting timestep {idx+1}/{len(selected_indices)}: {actual_label}")

            ds_t = ds.isel(time=real_idx)

            # --- Loop over all variables ---
            for var_name in var_names:
                if var_name not in ds_t:
                    print(f"    Skipping variable {var_name}: not found in dataset.")
                    continue

                current_cfg = configs[var_name]
                data_to_plot = ds_t[var_name]
                detrend_suffix = ""
                file_detrend_tag = ""
                cmap_to_use = current_cfg['cmap']
                vmin_to_use = current_cfg['vmin']
                vmax_to_use = current_cfg['vmax']

                if var_name == 'mesh2d_mor_bl' and apply_detrending and reference_per_face is not None:
                    raw_bed = np.asarray(data_to_plot.values)

                    # 1) Mask land at the CURRENT timestep FIRST. Using the
                    #    current-time mask (rather than the t=0 mask) ensures
                    #    cells that have eroded from land into estuary since
                    #    t=0 are still included and correctly detrended.
                    masked_bed = raw_bed.copy()
                    masked_bed[raw_bed > detrend_land_threshold] = np.nan

                    # 2) Subtract the t=0 centerline reference profile,
                    #    interpolated to each face's x-coordinate.
                    detrended_bed = masked_bed - reference_per_face

                    data_to_plot = data_to_plot.copy(data=detrended_bed)
                    detrend_suffix = " (Detrended)"
                    file_detrend_tag = "_detrended"
                    cmap_to_use = create_terrain_colormap()
                    # Keep color mapping centered at zero so white corresponds to zero change.
                    detrended_limit = max(abs(current_cfg['vmin']), abs(current_cfg['vmax']))
                    vmin_to_use = -detrended_limit
                    vmax_to_use = detrended_limit
                
                current_xlim = ZOOM_XLIM if ZOOM else (CACHE_BBOX[0], CACHE_BBOX[2])
                current_ylim = ZOOM_YLIM if ZOOM else (CACHE_BBOX[1], CACHE_BBOX[3])
                figsize = compute_map_figsize(current_xlim, current_ylim, FIGURE_WIDTH_MM, CBAR_WIDTH_FRACTION)
                fig, ax = plt.subplots(figsize=figsize)

                #fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                pc = data_to_plot.ugrid.plot(
                    ax=ax,
                    cmap=cmap_to_use,
                    add_colorbar=False,
                    edgecolors='none',
                    vmin=vmin_to_use,
                    vmax=vmax_to_use
                )
                ax.set_aspect('equal')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                if ZOOM:
                    ax.set_xlim(ZOOM_XLIM)
                    ax.set_ylim(ZOOM_YLIM)
                # ax.set_title(f"{current_cfg['label']}{detrend_suffix} | {folder.name} | {actual_label}", color=_tc)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                cbar = plt.colorbar(pc, cax=cax)
                cbar.set_label(current_cfg['label'])
                cbar.ax.yaxis.set_tick_params(color=_tc)
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_tc)

                plt.tight_layout()
                is_final_timestep = (idx == len(selected_indices) - 1)
                _zoom_tag = f"_zoom" if ZOOM else ""
                save_name = f"{STYLE}_{current_cfg['file_tag']}{file_detrend_tag}{_zoom_tag}_{actual_tag}_{folder.name}.png"
                save_path = output_plots_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
                if is_final_timestep:
                    pdf_save_path = save_path.with_suffix('.pdf')
                    plt.savefig(pdf_save_path, bbox_inches='tight', transparent=True)
                    print(f"    Saved PDF: {pdf_save_path.name}")
                plt.close(fig)  # prevents memory issues over many timesteps
                print(f"    Saved: {save_name}")
    finally:
        ds.close()

print("\n" + "="*30)
print("BATCH PLOTTING COMPLETE")
print("="*30)