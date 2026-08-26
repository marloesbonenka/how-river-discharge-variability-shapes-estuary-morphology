"""Plot a 3x2 panel snapshot of a map variable at a fixed date.

Layout (AGU style, Calibri font):
  - Left column  (top -> bottom): Q = 250, 500, 1000 m3/s, constant discharge
                                   (the '01_..._pm1_n0' run for each discharge).
  - Right column (top -> bottom): Q = 500 m3/s, peak-discharge amplitude
                                   pm = 3, 4, 5 (n = 3 peaks/year runs).

A single colorbar spans the full height of the combined panels on the right.
"""
#%% --- Imports ---
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import cmocean

from FUNCTIONS.F_general import create_terrain_colormap, create_water_colormap, create_shear_stress_colormap
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

#%% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

TARGET_DATE = '2031-01-01'   # snapshot date; nearest available timestep per run is used

VAR_NAME = 'mesh2d_mor_bl'   # variable to plot (single variable -> single shared colorbar)
apply_detrending = True
ZOOM = True                 # True -> crop axes to ZOOM_XLIM / ZOOM_YLIM

# (discharge, pm, n) per row, top -> bottom
LEFT_COLUMN_ROWS  = [(250, 1, 0), (500, 1, 0), (1000, 1, 0)]   # constant discharge ('01_' folders)
RIGHT_COLUMN_ROWS = [(500, 3, 3), (500, 4, 3), (500, 5, 3)]    # pm = 3, 4, 5 @ Q = 500, n = 3

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170          # AGU full-page width, since we need 2 columns + colorbar
CBAR_WIDTH_FRACTION = 0.03     # fraction of total width reserved for the shared colorbar

AGU_RC = {
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 9,
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
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(AGU_RC)

# Detrending settings (bed level only) - same convention as plot_map_variable.py
reference_time_idx = 0
detrend_land_threshold = 6.0
CENTERLINE_XMIN = 20000          # [m] start of x-range for the reference profile
CENTERLINE_XMAX = 45000          # [m] end of x-range for the reference profile
CENTERLINE_Y = 7500               # [m] exact y-coordinate of the channel centerline

# Zoom settings
ZOOM_XLIM = (20000, 45000)   # x-range in model coordinates [m]
ZOOM_YLIM = (5000, 10000)    # y-range in model coordinates [m]

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000]  # xmin, ymin, xmax, ymax
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

VAR_CONFIG = {
    'mesh2d_mor_bl': {
        'cmap': cmocean.cm.delta,
        'vmin': -5,
        'vmax': 5,
        'label': 'bed level [m]',
        'file_tag': 'bedlevel_map',
    },
    'mesh2d_s1': {
        'cmap': create_water_colormap(),
        'vmin': -1,
        'vmax': 3,
        'label': 'water level [m]',
        'file_tag': 'water_level_map',
    },
    'mesh2d_taus': {
        'cmap': create_shear_stress_colormap(),
        'vmin': 0,
        'vmax': 5,
        'label': 'bed shear stress [N/m\u00b2]',
        'file_tag': 'shear_stress_map',
    },
}
current_cfg = VAR_CONFIG[VAR_NAME]

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'map_snapshot_6panel'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#%% --- 2. HELPERS ---
_FOLDER_RE = re.compile(r'^\d+_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\d+$')


def find_run_folder(q_base_path, discharge, pm, n):
    """Find the run folder matching a given (discharge, pm, n), e.g.
    '10_Qr500_pm3_n3.9599951' for (discharge=500, pm=3, n=3)."""
    candidates = []
    for f in q_base_path.iterdir():
        if not f.is_dir() or not f.name[:1].isdigit():
            continue
        m = _FOLDER_RE.match(f.name)
        if not m:
            continue
        q_val, pm_val, n_val = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if q_val == discharge and pm_val == pm and n_val == n:
            candidates.append(f)
    if not candidates:
        raise FileNotFoundError(
            f"No folder found for Q={discharge}, pm={pm}, n={n} in {q_base_path}"
        )
    candidates.sort(key=lambda x: int(x.name.split('_')[0]))
    return candidates[0]


def build_centerline_reference(ds, var_name, reference_time_idx, xmin, xmax, centerline_y):
    """Build a per-face reference array for detrending, derived from the
    centerline bed-level profile at the reference timestep (t=0), sampled
    along a known/fixed centerline y-coordinate. (Copied from
    plot_map_variable.py so this script is self-contained.)"""
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

    y_dist = np.abs(y_in - centerline_y)
    order = np.lexsort((y_dist, x_in))
    x_ord = x_in[order]
    bed_ord = bed_in[order]
    unique_x, first_idx = np.unique(x_ord, return_index=True)
    centerline_bed = bed_ord[first_idx]

    reference_per_face = np.full(face_x.shape, np.nan)
    reference_per_face[in_range] = np.interp(x_in, unique_x, centerline_bed)
    return reference_per_face


#%% --- 3. FIGURE / GRIDSPEC LAYOUT ---
x_span = ZOOM_XLIM[1] - ZOOM_XLIM[0]
y_span = ZOOM_YLIM[1] - ZOOM_YLIM[0]
aspect = y_span / x_span  # height / width per panel, from data (equal-aspect map)

fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
maps_width_in = fig_width_in * (1 - CBAR_WIDTH_FRACTION)
panel_width_in = maps_width_in / 2
panel_height_in = panel_width_in * aspect

# Extra vertical room (in inches) reserved for row titles / suptitle / x-axis
# label, added on top of the panel heights so the maps themselves don't get
# squeezed to make space for text.
ROW_TITLE_GAP_IN = 0.30   # gap between stacked rows, for each row's subplot title
TOP_MARGIN_IN = 0.55      # space above the top row for the figure suptitle + its own title
BOTTOM_MARGIN_IN = 0.35   # space below the bottom row for x tick labels + x-axis label

fig_height_in = panel_height_in * 3 + 2 * ROW_TITLE_GAP_IN + TOP_MARGIN_IN + BOTTOM_MARGIN_IN

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs = gridspec.GridSpec(
    3, 3, figure=fig,
    width_ratios=[1, 1, 2 * CBAR_WIDTH_FRACTION / (1 - CBAR_WIDTH_FRACTION)],
    wspace=0.08, hspace=ROW_TITLE_GAP_IN / panel_height_in,
    top=1 - TOP_MARGIN_IN / fig_height_in,
    bottom=BOTTOM_MARGIN_IN / fig_height_in,
)

axes = np.empty((3, 2), dtype=object)
ax_ref = None
for r in range(3):
    for c in range(2):
        ax = fig.add_subplot(gs[r, c], sharex=ax_ref, sharey=ax_ref)
        if ax_ref is None:
            ax_ref = ax
        axes[r, c] = ax

cax = fig.add_subplot(gs[:, 2])

_km_formatter = FuncFormatter(lambda val, pos: f'{val / 1000:.0f}')

panel_defs = []
for r, (q, pm, n) in enumerate(LEFT_COLUMN_ROWS):
    panel_defs.append((r, 0, q, pm, n, f'Q = {q} m\u00b3/s (constant)'))
for r, (q, pm, n) in enumerate(RIGHT_COLUMN_ROWS):
    panel_defs.append((r, 1, q, pm, n, f'Q = {q} m\u00b3/s, $R_{{peak}}$ = {pm}'))

#%% --- 4. PLOT EACH PANEL ---
pc = None
detrended_active = False
for r, c, q, pm, n, title in panel_defs:
    ax = axes[r, c]
    q_base_path = BASE_DIR / f"Q{q}"
    folder = find_run_folder(q_base_path, q, pm, n)
    print(f"[row {r}, col {c}] Q={q} pm={pm} n={n} -> {folder.name}")

    assessment_dir = q_base_path / 'cached_data'
    run_paths = get_stitched_map_run_paths(
        base_path=q_base_path, folder_name=folder.name,
        timed_out_dir=None, variability_map=None, analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir, folder_name=folder.name, run_paths=run_paths,
        var_names=[VAR_NAME], bbox=CACHE_BBOX, append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES, cache_tag=cache_tag,
    )
    if ds is None:
        raise RuntimeError(f"No data cached for {folder.name}")

    try:
        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        target_dt = np.datetime64(TARGET_DATE, 'ns')
        nearest_idx = int(np.argmin(np.abs(time_values - target_dt)))
        actual_label = str(np.datetime_as_string(time_values[nearest_idx], unit='D'))

        vmin_to_use, vmax_to_use, cmap_to_use = current_cfg['vmin'], current_cfg['vmax'], current_cfg['cmap']

        if apply_detrending and VAR_NAME == 'mesh2d_mor_bl' and 'time' in ds[VAR_NAME].dims:
            reference_per_face = build_centerline_reference(
                ds, VAR_NAME, reference_time_idx,
                xmin=CENTERLINE_XMIN, xmax=CENTERLINE_XMAX, centerline_y=CENTERLINE_Y,
            )
            ds_t = ds.isel(time=nearest_idx)
            raw_bed = np.asarray(ds_t[VAR_NAME].values)
            masked_bed = raw_bed.copy()
            masked_bed[raw_bed > detrend_land_threshold] = np.nan
            detrended_bed = masked_bed - reference_per_face
            data_to_plot = ds_t[VAR_NAME].copy(data=detrended_bed)
            cmap_to_use = create_terrain_colormap()
            detrended_limit = max(abs(current_cfg['vmin']), abs(current_cfg['vmax']))
            vmin_to_use, vmax_to_use = -detrended_limit, detrended_limit
            detrended_active = True
        else:
            ds_t = ds.isel(time=nearest_idx)
            data_to_plot = ds_t[VAR_NAME]

        pc = data_to_plot.ugrid.plot(
            ax=ax, cmap=cmap_to_use, add_colorbar=False, edgecolors='none',
            vmin=vmin_to_use, vmax=vmax_to_use,
        )
    finally:
        ds.close()

    ax.set_aspect('equal')
    if ZOOM:
        ax.set_xlim(ZOOM_XLIM)
        ax.set_ylim(ZOOM_YLIM)
    ax.set_title(title, fontsize=8)
    ax.xaxis.set_major_formatter(_km_formatter)
    ax.yaxis.set_major_formatter(_km_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')

    if r != 2:
        ax.tick_params(labelbottom=False)
    if c != 0:
        ax.tick_params(labelleft=False)

#%% --- 5. SHARED COLORBAR + SAVE ---
cbar_label = current_cfg['label'] + (' (detrended)' if detrended_active else '')
cbar = fig.colorbar(pc, cax=cax)
cbar.set_label(cbar_label)

fig.supxlabel('x [km]')
fig.supylabel('y [km]')
fig.suptitle(f"{cbar_label} snapshot \u2014 {TARGET_DATE}", fontsize=9, y=0.99)

save_name = f"AGU_{current_cfg['file_tag']}_6panel_{TARGET_DATE}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%
