"""Along-estuary 'active width': the part of the estuary cross-section where
sediment is actually moving over one tidal cycle, taken from the last 24h of
the simulation (≈ 2 M2 cycles, averaged to 1 cycle), for the detailed-hydro-run
scenarios. Quantifies the observation in plot_map_hydro_netsedtransport.py
that upstream (x = 35-45 km) the active part of the estuary is narrower for
variable discharge.

Per face, the time-averaged GROSS along-flow transport (flood + ebb magnitude,
no cancellation; compute_net_along_flow_transport(..., return_gross=True)) is
used rather than the net transport, so a channel with balanced flood and ebb
transport still counts as active.

A face is 'active' if its gross transport exceeds ACTIVE_FRAC x the p95 gross
transport of the CONSTANT run at the same Q (over X_RANGE x Y_RANGE). The same
threshold is applied to both scenarios of that Q, so differences in active
width are not caused by a different threshold.
  active width(x) = sum(A_f of active faces in bin) / bin length   [km]

The console prints the mean active width over UPSTREAM_KM for a few threshold
fractions, to check that the result does not depend on ACTIVE_FRAC.

Layout (AGU style, Calibri font):
  - Rows (top -> bottom): Q = 250, 500, 1000 m3/s.
  - In each panel: constant discharge (grey dashed) and R_peak = 5, n = 3 (dark blue).
"""

# %% --- Imports ---
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, resolve_cache_folder_label
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_general import apply_plot_style, find_run_folder_by_qpmn
from functions.F_sedimentbuffers import compute_net_along_flow_transport

# %% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

DISCHARGES = [250, 500, 1000]   # one row per discharge, top -> bottom
# (label, pm, n, colour, linestyle) per line in each panel; the first one is the threshold reference
SCENARIOS = [
    ('constant',              1, 0, '#7f7f7f',           '--'),
    (r'$R_{\mathrm{peak}}$ = 5', 5, 3, plt.cm.Blues(0.90), '-'),
]

LOAD_VARS = ['mesh2d_sxtot', 'mesh2d_sytot', 'mesh2d_ucx', 'mesh2d_ucy']  # sediment transport + velocity components
MIN_VELOCITY_FOR_PROJECTION = 0.0001

# time window for calculation
WINDOW_HOURS = 24.0

# Along-estuary binning
X_RANGE = (20000, 45000)   # along-estuary extent [m] (same as the map zoom)
Y_RANGE = (5000, 10000)    # cross-estuary extent [m]
X_BIN_M = 500              # bin length [m]

# Active-face threshold, as a fraction of the constant run's p95 gross transport
ACTIVE_FRAC = 0.10
ACTIVE_FRAC_CHECK = [0.05, 0.10, 0.20]   # printed sensitivity check
UPSTREAM_KM = (35, 45)                   # reach highlighted in the figure and summarised in the console

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
PANEL_HEIGHT_IN = 1.25
LINE_WIDTH = 1.0
XTICK_STEP_KM = 5

apply_plot_style('AGU', font_size=8)

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000]  # xmin, ymin, xmax, ymax
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'net_sed_transport_alongestuary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_EDGES = np.arange(X_RANGE[0], X_RANGE[1] + 1, X_BIN_M)
X_KM = 0.5 * (X_EDGES[:-1] + X_EDGES[1:]) / 1000


# %% --- 2. HELPERS ---
# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}
_DHR_FOLDER_RE = re.compile(r'^dhr_\d+_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\w+$')


def load_gross_transport(q, pm, n):
    """Per-face time-averaged gross along-flow transport of the detailed-hydro run for (Q, pm, n).
    Returns (gross, face_x, face_y, area) as numpy arrays."""
    q_base_path = BASE_DIR / f"Q{q}"
    dhr_base_path = q_base_path / 'detailed-hydro-run'
    folder = find_run_folder_by_qpmn(
        dhr_base_path, q, pm, n,
        folder_regex=_DHR_FOLDER_RE,
        sort_key=lambda x: int(x.name.split('_')[1]),
    )
    print(f"Q={q} pm={pm} n={n} -> {folder.name}")

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
        _, gross_da, _ = compute_net_along_flow_transport(
            ds, WINDOW_HOURS, MIN_VELOCITY_FOR_PROJECTION, return_gross=True,
        )
        grid = gross_da.ugrid.grid
        return (np.asarray(gross_da.values, dtype=float), np.asarray(grid.face_x),
                np.asarray(grid.face_y), np.asarray(grid.area))
    finally:
        ds.close()


def active_width_km(gross, face_x, face_y, area, threshold):
    """Width [km] of faces with gross transport > threshold, per x-bin in X_EDGES."""
    keep = ((face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1])
            & np.isfinite(gross) & (gross > threshold))
    b = np.digitize(face_x[keep], X_EDGES) - 1
    ok = (b >= 0) & (b < len(X_EDGES) - 1)
    a_active = np.bincount(b[ok], weights=area[keep][ok], minlength=len(X_EDGES) - 1)
    return a_active / np.diff(X_EDGES) / 1000


def in_domain(face_x, face_y):
    return ((face_x >= X_RANGE[0]) & (face_x <= X_RANGE[1])
            & (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1]))


# %% --- 3. COMPUTE ACTIVE WIDTH ---
fields = {}       # {(q, pm, n): (gross, face_x, face_y, area)}
p95_ref = {}      # {q: p95 gross transport of the reference (constant) run}
for q in DISCHARGES:
    for i, (_, pm, n, _, _) in enumerate(SCENARIOS):
        fields[(q, pm, n)] = load_gross_transport(q, pm, n)
        if i == 0:
            gross, fx, fy, _ = fields[(q, pm, n)]
            p95_ref[q] = np.nanpercentile(gross[in_domain(fx, fy)], 95)

upstream = (X_KM >= UPSTREAM_KM[0]) & (X_KM <= UPSTREAM_KM[1])
print(f"\nMean active width [km] over x = {UPSTREAM_KM[0]}-{UPSTREAM_KM[1]} km "
      f"(threshold = frac x p95 gross transport of the constant run)")
for q in DISCHARGES:
    for frac in ACTIVE_FRAC_CHECK:
        widths = [np.mean(active_width_km(*fields[(q, pm, n)], frac * p95_ref[q])[upstream])
                  for _, pm, n, _, _ in SCENARIOS]
        txt = ', '.join(f"pm{pm}_n{n}: {w:.2f}" for (_, pm, n, _, _), w in zip(SCENARIOS, widths))
        print(f"  Q={q:<5} frac={frac:.2f} -> {txt}  (ratio {widths[1] / widths[0]:.2f})")

profiles = {key: active_width_km(*f, ACTIVE_FRAC * p95_ref[key[0]]) for key, f in fields.items()}


# %% --- 4. FIGURE ---
fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
fig = plt.figure(figsize=(fig_width_in, PANEL_HEIGHT_IN * len(DISCHARGES) + 0.9))
gs = gridspec.GridSpec(len(DISCHARGES), 1, figure=fig, hspace=0.12)

axes = []
for r, q in enumerate(DISCHARGES):
    ax = fig.add_subplot(gs[r], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
    axes.append(ax)

    ax.axvspan(*UPSTREAM_KM, color='0.93', linewidth=0, zorder=0)
    for label, pm, n, color, ls in SCENARIOS:
        ax.plot(X_KM, profiles[(q, pm, n)], color=color, linestyle=ls, linewidth=LINE_WIDTH, zorder=3)

    ax.set_xlim(X_RANGE[0] / 1000, X_RANGE[1] / 1000)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(0.01, 0.95, f"({chr(ord('a') + r)}) Q = {q} m\u00b3/s", transform=ax.transAxes,
            ha='left', va='top')

    if r != len(DISCHARGES) - 1:
        ax.tick_params(labelbottom=False)

axes[-1].set_xlabel('distance along estuary [km]  (sea \u2190 \u2192 river)')
fig.supylabel('active width [km]')

legend_handles = [mlines.Line2D([], [], color=c, linestyle=ls, linewidth=LINE_WIDTH, label=label)
                  for label, _, _, c, ls in SCENARIOS]
axes[0].legend(handles=legend_handles, loc='upper right', frameon=True)

fig.suptitle(f"Active width (gross transport > {ACTIVE_FRAC:.0%} of constant-run p95) "
             f"\u2014 last {WINDOW_HOURS:.0f}h tidal cycle", fontsize=9)

save_name = f"AGU_active_width_alongestuary_frac{ACTIVE_FRAC:.2f}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%
