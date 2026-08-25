"""Plot map output at a certain timestep"""
#%% 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

from FUNCTIONS.F_general import create_bedlevel_colormap, create_terrain_colormap, create_water_colormap, create_shear_stress_colormap
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

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

STYLES = {
    'default': {},   # use matplotlib defaults
    'whitefig': {
        'figure.facecolor':    'none',
        'axes.facecolor':      'none',
        'axes.edgecolor':      'white',
        'axes.labelcolor':     'white',
        'xtick.color':         'white',
        'ytick.color':         'white',
        'text.color':          'white',
        'grid.color':          'white',
        'legend.facecolor':    'none',
        'legend.edgecolor':    'white',
        'savefig.transparent': True,
    },
    'AGU': {
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],  # fallback if Arial unavailable
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 8,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Calibri',
        'mathtext.it': 'Calibri:italic',
        'mathtext.bf': 'Calibri:bold',

        # --- Line weights: avoid hairlines (AGU rejects anything under 0.5pt) ---
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.75,
        'grid.linewidth': 0.4,
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
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    }
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])

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
config = f"Model_Output/Q{DISCHARGE}"
base_path = base_directory / config #Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q500\0_Noise_Q500") 

assessment_dir = base_path / 'cached_data'

timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')
    #raise FileNotFoundError(f"Timed-out directory not found: {timed_out_dir}")

VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)

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
def build_centerline_reference(ds, var_name, reference_time_idx, xmin, xmax, centerline_y):
    """
    Build a per-face reference array for detrending, derived from the
    centerline bed-level profile at the reference timestep (t=0), sampled
    along a known/fixed centerline y-coordinate.

    For each unique x in [xmin, xmax], the reference value is the t=0 bed
    level of the face whose y-coordinate is closest to `centerline_y` in
    that x-column. This is interpolated to every face's own x-coordinate.
    Faces with x outside [xmin, xmax] get NaN (no extrapolation), so areas
    where the reference is not meaningful (e.g. the sea basin seaward of
    the estuary mouth) are left unplotted rather than clamped to an edge
    value.

    Note: this only samples the *initial* (t=0) geometry to build the
    reference profile. Any land masking for a given timestep should be
    applied to the data being detrended (at that timestep), not here -
    see the main loop, where the current-timestep land mask is applied
    BEFORE subtracting this reference, so that cells which have eroded
    from land into estuary between t=0 and t=time are correctly included.
    """
    if 'mesh2d_face_x' not in ds or 'mesh2d_face_y' not in ds:
        raise ValueError("Dataset is missing mesh2d_face_x/mesh2d_face_y; cannot build centerline reference.")

    face_x = np.asarray(ds['mesh2d_face_x'].values)
    face_y = np.asarray(ds['mesh2d_face_y'].values)
    reference_bed_full = np.asarray(ds[var_name].isel(time=reference_time_idx).values)

    in_range = (face_x >= xmin) & (face_x <= xmax)
    if not np.any(in_range):
        raise ValueError(f"No faces found with x in [{xmin}, {xmax}]; cannot build centerline reference.")

    x_in = face_x[in_range]
    y_in = face_y[in_range]
    bed_in = reference_bed_full[in_range]

    # For each unique x, pick the face whose y is closest to centerline_y.
    # Sort primarily by x (ascending), secondarily by distance to
    # centerline_y (ascending), so the first entry within each x-group is
    # the closest-to-centerline face for that column.
    y_dist = np.abs(y_in - centerline_y)
    order = np.lexsort((y_dist, x_in))
    x_ord = x_in[order]
    bed_ord = bed_in[order]
    unique_x, first_idx = np.unique(x_ord, return_index=True)
    centerline_bed = bed_ord[first_idx]

    # Interpolate the t=0 centerline profile to every face's x-coordinate.
    # Faces with x outside [xmin, xmax] get NaN (left unplotted), not
    # clamped to an edge value.
    reference_per_face = np.full(face_x.shape, np.nan)
    reference_per_face[in_range] = np.interp(x_in, unique_x, centerline_bed)
    return reference_per_face

def compute_map_figsize(xlim, ylim, width_mm=FIGURE_WIDTH_MM, cbar_frac=CBAR_WIDTH_FRACTION):
    """Derive (width_in, height_in) from data aspect ratio so an equal-aspect
    map fills the frame at the target AGU print width, with space reserved
    for the colorbar."""
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    aspect = y_span / x_span

    fig_width_in = width_mm * MM_TO_IN
    map_width_in = fig_width_in * cbar_frac
    fig_height_in = map_width_in * aspect
    return (fig_width_in, fig_height_in)

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
                figsize = compute_map_figsize(current_xlim, current_ylim)
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