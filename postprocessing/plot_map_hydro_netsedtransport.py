"""Plot a 3x2 panel snapshot of net (flood- vs ebb-dominant) sediment
transport over one tidal cycle, taken from the last 24h of the simulation
(≈ 2 M2 cycles, averaged to 1 cycle), for the detailed-hydro-run scenarios.

Sign convention (landward = +x, towards the river; seaward = -x, towards the
mouth): 
  FLOOD-directed transport = BLUE (+)   — occurring while local flow is landward
  EBB-directed transport   = RED  (-)   — occurring while local flow is seaward

Computed in two steps, at each cell and timestep:
  1. transport_mag_along_flow = (sx*ucx + sy*ucy) / |u|, clipped to >= 0.
     This is the magnitude of sediment transport aligned with the local flow
     direction
  2. flood_ebb_sign = sign(ucx) — whether the flow itself is landward (+) or
     seaward (-) at the snapshot moment
  s_along = transport_mag_along_flow * flood_ebb_sign

Layout (AGU style, Calibri font — mirrors plot_map_variable_6panel_snapshot.py):
  - Left column  (top -> bottom): Q = 250, 500, 1000 m3/s, constant discharge
                                   (the 'dhr_01_..._pm1_n0' run for each discharge).
  - Right column (top -> bottom): Q = 250, 500, 1000 m3/s, pm = 5, n = 3
                                   (peak-discharge amplitude 5, 3 peaks/year).

A single colorbar spans the full height of the combined panels on the right.
"""

# %% --- Imports ---
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import cmocean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from appendix/

from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, resolve_cache_folder_label
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_general import apply_plot_style, find_run_folder_by_qpmn
from functions.F_sedimentbuffers import compute_net_along_flow_transport

# %% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

# (discharge, pm, n) per row, top -> bottom
LEFT_COLUMN_ROWS  = [(250, 1, 0), (500, 1, 0), (1000, 1, 0)]   # constant discharge ('dhr_01_..._pm1_n0' folders)
RIGHT_COLUMN_ROWS = [(250, 5, 3), (500, 5, 3), (1000, 5, 3)]   # pm = 5, n = 3 @ Q = 250, 500, 1000

LOAD_VARS = ['mesh2d_sxtot', 'mesh2d_sytot', 'mesh2d_ucx', 'mesh2d_ucy']  # sediment transport + velocity components
MIN_VELOCITY_FOR_PROJECTION = 0.0001

# time window for calculation
WINDOW_HOURS = 24.0

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170          # AGU full-page width, since we need 2 columns + colorbar
CBAR_WIDTH_FRACTION = 0.03     # fraction of total width reserved for the shared colorbar

apply_plot_style('AGU', font_size=8)

# Zoom settings
ZOOM = True
ZOOM_XLIM = (20000, 45000)   # x-range in model coordinates [m]
ZOOM_YLIM = (5000, 10000)    # y-range in model coordinates [m]

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000]  # xmin, ymin, xmax, ymax
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# colour scale for net transport (signed, diverging, centered at 0).
VMAX = 5e-7
VMIN = -VMAX
CMAP = cmocean.cm.curl #plt.cm.RdBu   # red = negative (ebb/seaward), blue = positive (flood/landward)
VAR_LABEL = r'net sediment transport [$m^3\,s^{-1}\,m^{-1}$]'

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'net_sed_transport_6panel'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% --- 2. HELPERS ---
# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}
_DHR_FOLDER_RE = re.compile(r'^dhr_\d+_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\w+$')


# %% --- 3. FIGURE / GRIDSPEC LAYOUT ---
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

# %% --- 4. PLOT EACH PANEL ---
pc = None
for r, c, q, pm, n, title in panel_defs:
    ax = axes[r, c]
    q_base_path = BASE_DIR / f"Q{q}"
    folder = find_run_folder_by_qpmn(
        q_base_path / 'detailed-hydro-run', q, pm, n,
        folder_regex=_DHR_FOLDER_RE,
        sort_key=lambda x: int(x.name.split('_')[1]),
    )
    print(f"[row {r}, col {c}] Q={q} pm={pm} n={n} -> {folder.name}")

    dhr_base_path = q_base_path / 'detailed-hydro-run'
    assessment_dir = dhr_base_path / 'cached_data'
    run_paths = get_stitched_map_run_paths(
        base_path=dhr_base_path, folder_name=folder.name,
        timed_out_dir=None, variability_map=None, analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    cache_label = resolve_cache_folder_label(assessment_dir, folder.name, LOAD_VARS, cache_tag)
    if cache_label != folder.name:
        print(f"  [cache] reusing existing cache labeled '{cache_label}' for run folder '{folder.name}'")

    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir, folder_name=cache_label, run_paths=run_paths,
        var_names=LOAD_VARS, bbox=CACHE_BBOX, append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES, cache_tag=cache_tag,
    )
    if ds is None:
        raise RuntimeError(f"No data cached for {folder.name}")

    try:
        net_da, _ = compute_net_along_flow_transport(ds, WINDOW_HOURS, MIN_VELOCITY_FOR_PROJECTION)
        pc = net_da.ugrid.plot(
            ax=ax, cmap=CMAP, add_colorbar=False, edgecolors='none',
            vmin=VMIN, vmax=VMAX,
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

# %% --- 5. SHARED COLORBAR + SAVE ---
cbar = fig.colorbar(pc, cax=cax)
cbar.set_label(VAR_LABEL)

fig.supxlabel('x [km]')
fig.supylabel('y [km]')
fig.suptitle(f"Net sediment transport \u2014 last {WINDOW_HOURS:.0f}h tidal cycle", fontsize=9, y=0.99)

save_name = "AGU_net_sed_transport_6panel"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%