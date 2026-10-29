"""Along-estuary collapse of the net (flood- vs ebb-dominant) sediment transport
over one tidal cycle, taken from the last 24h of the simulation (≈ 2 M2 cycles,
averaged to 1 cycle), for the detailed-hydro-run scenarios. 1D version of
plot_map_hydro_netsedtransport.py.

Per face, the net transport q_f is computed exactly as in the map version
(compute_net_along_flow_transport; landward = +x = FLOOD, seaward = -x = EBB).
A plain width-average of q_f is ebb-dominant everywhere, because the ebb
channels cancel the flood-dominant faces. So instead, per along-estuary x-bin,
the share of the cross-section that is ebb- or flood-dominant is computed:
  ebb-dominant   [%] = 100 * sum(A_f where a_f < -ASYM_THRESHOLD) / (bin length * Y_RANGE width)
  flood-dominant [%] = 100 * sum(A_f where a_f > +ASYM_THRESHOLD) / (bin length * Y_RANGE width)
with a_f = net / gross = (F - E) / (F + E) the transport asymmetry of the face
(F, E = flood- and ebb-directed transport over the cycle; -1 = pure ebb, +1 = pure
flood). This is independent of the transport magnitude, so weakly but consistently
flood-dominant flats count as much as strongly ebb-dominant channels. Only faces
with gross transport > GROSS_MIN (sediment moves at all) are classified.
Each panel is annotated with the share over the whole X_RANGE for both scenarios
and the relative change of R_peak = 5 vs constant.

Layout (AGU style, Calibri font):
  - Rows (top -> bottom): Q = 250, 500, 1000 m3/s.
  - Columns: ebb-dominant (left), flood-dominant (right).
  - In each panel: constant discharge (grey dashed, 'dhr_01_..._pm1_n0') and
                   R_peak = 5, n = 3 (dark blue), as in the map version's two columns.
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
X_BIN_M = 1000             # bin length [m]

# Flood/ebb-dominant classification by transport asymmetry a = net / gross
ASYM_THRESHOLD = 0.1                  # |a| above this -> flood- (a > 0) or ebb-dominant (a < 0)
GROSS_MIN = 0.0                       # [m^2/s] faces with gross transport <= this (no sediment moving) are skipped
ASYM_CHECK = [0.0, 0.1, 0.25]         # printed sensitivity check (thresholds)
GROSS_MIN_CHECK = [0.0, 1e-10, 1e-9]  # printed sensitivity check (gross floors)

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


def load_net_transport(q, pm, n):
    """Per-face net and gross along-flow transport of the detailed-hydro run for (Q, pm, n).
    Returns (net, gross, face_x, area) as numpy arrays, restricted to Y_RANGE."""
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
        net_da, gross_da, _ = compute_net_along_flow_transport(
            ds, WINDOW_HOURS, MIN_VELOCITY_FOR_PROJECTION, return_gross=True,
        )
        grid = net_da.ugrid.grid
        net = np.asarray(net_da.values, dtype=float)
        gross = np.asarray(gross_da.values, dtype=float)
        face_x, face_y, area = np.asarray(grid.face_x), np.asarray(grid.face_y), np.asarray(grid.area)
    finally:
        ds.close()

    keep = (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1]) & np.isfinite(net) & np.isfinite(gross)
    return net[keep], gross[keep], face_x[keep], area[keep]


def bin_sum(values, face_x):
    """Sum of values per x-bin in X_EDGES."""
    b = np.digitize(face_x, X_EDGES) - 1
    ok = (b >= 0) & (b < len(X_EDGES) - 1)
    return np.bincount(b[ok], weights=values[ok], minlength=len(X_EDGES) - 1)


def dominant_pct(net, gross, face_x, area, asym_threshold, gross_min):
    """Ebb- and flood-dominant share [%] of the cross-section (Y_RANGE width) per x-bin,
    and the same share over the whole X_RANGE. Returns (ebb, flood, ebb_total, flood_total)."""
    moving = gross > gross_min
    asym = np.where(moving, net / np.where(moving, gross, 1.0), 0.0)
    bin_area = np.diff(X_EDGES) * (Y_RANGE[1] - Y_RANGE[0])
    ebb_a = bin_sum(np.where(moving & (asym < -asym_threshold), area, 0.0), face_x)
    flood_a = bin_sum(np.where(moving & (asym > asym_threshold), area, 0.0), face_x)
    return (100 * ebb_a / bin_area, 100 * flood_a / bin_area,
            100 * ebb_a.sum() / bin_area.sum(), 100 * flood_a.sum() / bin_area.sum())


# %% --- 3. COMPUTE PROFILES ---
fields = {}     # {(q, pm, n): (net, gross, face_x, area)}
for q in DISCHARGES:
    for _, pm, n, _, _ in SCENARIOS:
        fields[(q, pm, n)] = load_net_transport(q, pm, n)

dominant = {key: dominant_pct(*f, ASYM_THRESHOLD, GROSS_MIN) for key, f in fields.items()}

print(f"\nEbb- / flood-dominant share [%] of x = {X_RANGE[0] / 1000:.0f}-{X_RANGE[1] / 1000:.0f} km "
      f"(asymmetry threshold, gross-transport floor)")
for q in DISCHARGES:
    for asym_thr in ASYM_CHECK:
        for g_min in GROSS_MIN_CHECK:
            totals = [dominant_pct(*fields[(q, pm, n)], asym_thr, g_min)[2:] for _, pm, n, _, _ in SCENARIOS]
            txt = ', '.join(f"pm{pm}_n{n}: ebb {e:.1f} / flood {f:.1f}"
                            for (_, pm, n, _, _), (e, f) in zip(SCENARIOS, totals))
            print(f"  Q={q:<5} |a|>{asym_thr:.2f} gross>{g_min:.0e} -> {txt}")


def _change_txt(ref, new):
    return f"{100 * (new - ref) / ref:+.0f}%" if ref > 0 else 'n/a'


# %% --- 4. FIGURE ---
fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
fig = plt.figure(figsize=(fig_width_in, PANEL_HEIGHT_IN * len(DISCHARGES) + 0.9))
gs = gridspec.GridSpec(len(DISCHARGES), 2, figure=fig, hspace=0.15, wspace=0.18)

axes = np.empty((len(DISCHARGES), 2), dtype=object)
for r, q in enumerate(DISCHARGES):
    for c in range(2):   # c = 0: ebb, c = 1: flood
        ax = fig.add_subplot(gs[r, c], sharex=axes[0, 0], sharey=axes[0, c])
        axes[r, c] = ax

        for label, pm, n, color, ls in SCENARIOS:
            ax.plot(X_KM, dominant[(q, pm, n)][c], color=color, linestyle=ls, linewidth=LINE_WIDTH, zorder=3)

        # share over the whole reach, constant vs R_peak = 5, and the relative change
        (_, pm0, n0, _, _), (_, pm1, n1, _, _) = SCENARIOS
        tot0, tot1 = dominant[(q, pm0, n0)][2 + c], dominant[(q, pm1, n1)][2 + c]
        ax.text(0.99, 0.95, f"{SCENARIOS[0][0]}: {tot0:.1f}%\n{SCENARIOS[1][0]}: {tot1:.1f}% "
                            f"({_change_txt(tot0, tot1)})",
                transform=ax.transAxes, ha='right', va='top', linespacing=1.4,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

        ax.set_xlim(X_RANGE[0] / 1000, X_RANGE[1] / 1000)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
        ax.grid(True, color='0.9', linewidth=0.5)
        ax.text(0.01, 0.95, f"({chr(ord('a') + 2 * r + c)}) Q = {q} m\u00b3/s", transform=ax.transAxes,
                ha='left', va='top')
        if r != len(DISCHARGES) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('distance along estuary [km]')

# headroom per column, so the panel labels / reach totals stay clear of the lines
for c in range(2):
    col_max = max(np.nanmax(dominant[(q, pm, n)][c]) for q in DISCHARGES for _, pm, n, _, _ in SCENARIOS)
    axes[0, c].set_ylim(0, 1.45 * col_max)

axes[0, 0].set_title('ebb-dominant (seaward net transport)')
axes[0, 1].set_title('flood-dominant (landward net transport)')
axes[1, 0].set_ylabel('share of cross-section [%]')

legend_handles = [mlines.Line2D([], [], color=c, linestyle=ls, linewidth=LINE_WIDTH, label=label)
                  for label, _, _, c, ls in SCENARIOS]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(f"Net sediment transport along the estuary \u2014 last {WINDOW_HOURS:.0f}h tidal cycle", fontsize=9)

save_name = "AGU_net_sed_transport_alongestuary"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%
