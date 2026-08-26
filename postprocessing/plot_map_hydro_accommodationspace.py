"""
Along-estuary p5 / p95 water level profiles for four detailed-hydro-run scenarios,
plus an accommodation-space figure (Figure 2) overlaying p95 water level and p95 bed
level on the same axes to visualise the vertical gap available for bar aggradation.

For each scenario, the script:
  1. Loads the cached water level (mesh2d_s1) and bed level (mesh2d_mor_bl).
  2. Bins face cells by x-coordinate into equal-width intervals.
  3. Computes p5 / p95 across (time × faces) within each bin.
  4. Produces two figures:
       Fig 1 — p95 (high water) and p5 (low water) water level, all scenarios.
       Fig 2 — p95 water level vs p95 bed level on the same axes; one panel per
               scenario pair (constant vs. peak flow) to make the accommodation
               space argument explicit.

Styling notes:
  - All font sizes/weights for text are driven by the AGU_RC rcParams (font.size,
    axes.titlesize, figure.titlesize, legend.fontsize, ...). No fontsize= kwargs
    are set manually anywhere below.
  - Top and right axes spines are removed on every panel.
  - Curves are labelled directly at their right-hand endpoint instead of using a
    separate legend; shaded accommodation-space regions get an in-place label.
"""

# %% Imports
import sys
from pathlib import Path

import cmocean                          # noqa: F401  (kept for colourmap consistency)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'mean flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
    'peak flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
}

VAR_WL    = 'mesh2d_s1'
VAR_BL    = 'mesh2d_mor_bl'
LOAD_VARS = [VAR_WL, VAR_BL]

# Spatial extent used when building / reading the cache
CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Along-estuary binning (x-coordinate)
X_BIN_WIDTH = 500          # metres — adjust to match your morpho script's resolution
X_MIN       = 19000        # trim sea-side margin (model mouth)
X_MAX       = 45000        # upstream limit

# Wet-cell filter: exclude cells that are essentially always dry
# (water level == fill value or NaN).  Set to None to skip.
WET_FRACTION_THRESHOLD = 0.1   # cell must be wet in ≥10 % of timesteps

# Colours for the four scenarios — order matches SCENARIOS dict
cmap = cmocean.cm.amp

# Flow categories ordered from low (light) to peak (dark)
flow_scenarios = ['low flow', 'mean flow', 'peak flow']

# Sample 3 evenly spaced points along the colormap
# Tip: Using range 0.2 to 0.9 stops 'low flow' from being completely white/invisible on white backgrounds
samples = np.linspace(0.15, 1, len(flow_scenarios))

# Map to hex colors
SCENARIO_COLORS = {
    'constant': '#888888',
    **{name: mcolors.to_hex(cmap(val)) for name, val in zip(flow_scenarios, samples)}
}

SCENARIO_LINESTYLE = {
    'constant':  '--',
    'low flow':  '-',
    'mean flow': '-',
    'peak flow': '-',
}

# Output
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\water level")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME_WL    = "waterlevel_p5_p95_along_estuary.png"
OUTPUT_FILENAME_ACCOM = "accommodation_space_along_estuary.png"

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170/2          # AGU full-page width, since we need 2 columns + colorbar
CBAR_WIDTH_FRACTION = 0.03     # fraction of total width reserved for the shared colorbar

# Fraction of the x-range added as blank space on the right of each panel so
# the direct end-of-line labels have room to sit without being clipped.
LABEL_MARGIN_FRAC = 0.16

FONTSIZE = 11

AGU_RC = {
    'font.size': FONTSIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': FONTSIZE,
    'axes.titlesize': FONTSIZE,
    'xtick.labelsize': FONTSIZE - 1,
    'ytick.labelsize': FONTSIZE - 1,
    'legend.fontsize': FONTSIZE - 1,
    'figure.titlesize': FONTSIZE + 1,
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

#%% ---------------------------------------------------------------------------
# DIRECT-LABEL HELPERS (replace fig.legend() calls)
# ---------------------------------------------------------------------------

def add_direct_labels(ax, curves, min_sep_frac=0.09, x_offset=6):
    """Label each line directly to the right of its endpoint instead of using a

    legend.

    Parameters
    ----------
    ax : matplotlib Axes
    curves : list of (x_data, y_data, text, color) tuples
    min_sep_frac : minimum vertical spacing between labels (fraction of y-range)
    x_offset : horizontal distance in points between line end and label
    """
    entries = []
    for x_data, y_data, text, color in curves:
        finite = np.isfinite(y_data)
        if not finite.any():
            continue
        entries.append([x_data[finite][-1], y_data[finite][-1], text, color])

    if not entries:
        return

    # Sort and adjust vertical positions to prevent overlaps
    y_lo, y_hi = ax.get_ylim()
    min_gap = min_sep_frac * (y_hi - y_lo)

    entries.sort(key=lambda e: e[1])
    for i in range(1, len(entries)):
        if entries[i][1] - entries[i - 1][1] < min_gap:
            entries[i][1] = entries[i - 1][1] + min_gap

    label_bbox = dict(
        boxstyle="round,pad=0.15",
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
    )

    for x_end, y_end, text, color in entries:
        ax.annotate(
            text,
            xy=(x_end, y_end),
            xytext=(x_offset, 0),  # Positive offset shifts text to the right
            textcoords="offset points",
            color=color,
            ha="left",  # Left-aligned anchor places text extending rightward
            va="center",
            bbox=label_bbox,
            clip_on=False,  # Allows label to draw outside the main axes boundary
        )

    # Optional: Automatically pad the right x-margin so labels have space inside the figure
    ax.set_xmargin(0.1)
 
 
def add_fill_label(ax, x_data, y_lower, y_upper, text, color):
    """Place a label inside a fill_between region, centred on its x-extent."""
    finite = np.isfinite(y_lower) & np.isfinite(y_upper)
    if not finite.any():
        return
    xf = x_data[finite]
    yl = y_lower[finite]
    yu = y_upper[finite]
 
    x_mid = 0.355 * (xf.min() + xf.max())
    idx = np.argmin(np.abs(xf - x_mid))
 
    ax.text(
        xf[idx], 0.5 * (yl[idx] + yu[idx]),
        text, color=color, ha='center', va='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7),
        clip_on=True,
    )
 
 
def strip_top_right_spines(*axes):
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
#%% ---------------------------------------------------------------------------
# BIN EDGES
# ---------------------------------------------------------------------------
 
bin_edges   = np.arange(X_MIN, X_MAX + X_BIN_WIDTH, X_BIN_WIDTH)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / 1000   # → km for x-axis
n_bins = len(bin_centres)
 
# ---------------------------------------------------------------------------
# COLLECT PERCENTILE PROFILES PER SCENARIO
# ---------------------------------------------------------------------------
 
results = {}   # label → {'p5': array, 'p95': array, 'p95_bl': array}
 
for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")
 
    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)
 
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=None,
        variability_map=None,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]
 
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=LOAD_VARS,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )
 
    if ds is None:
        print(f"  [SKIP] No cached data found.")
        continue
 
    if VAR_WL not in ds:
        print(f"  [SKIP] '{VAR_WL}' not in dataset.")
        ds.close()
        continue
 
    # --- Face x-coordinates ---
    face_x = ds.grid.face_coordinates[:, 0]   # shape (nFaces,)
 
    # --- Water level array: (time, nFaces) ---
    wl = ds[VAR_WL].values   # shape (time, nFaces)
 
    # --- Bed level array: (time, nFaces) or (nFaces,) ---
    bl_raw = ds[VAR_BL].values if VAR_BL in ds else None
    if bl_raw is not None and bl_raw.ndim == 1:
        # static bed level — broadcast to (1, nFaces) so binning is uniform
        bl_raw = bl_raw[np.newaxis, :]
 
    # --- Optional: mask cells that are almost always dry (fill / NaN) ---
    if WET_FRACTION_THRESHOLD is not None:
        wet_frac = np.isfinite(wl).mean(axis=0)
        active   = wet_frac >= WET_FRACTION_THRESHOLD
        face_x_a = face_x[active]
        wl_a     = wl[:, active]
        bl_a     = bl_raw[:, active] if bl_raw is not None else None
        print(f"  Active cells after wet-fraction filter: {active.sum()} / {len(active)}")
    else:
        face_x_a, wl_a, bl_a = face_x, wl, bl_raw
 
    # --- Bin by x-coordinate and compute percentiles ---
    p5_profile     = np.full(n_bins, np.nan)
    p95_profile    = np.full(n_bins, np.nan)
    p95_bl_profile = np.full(n_bins, np.nan)
 
    bin_indices = np.digitize(face_x_a, bin_edges) - 1   # 0-based bin index
 
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
 
        # Water level
        vals_wl = wl_a[:, mask].ravel()
        vals_wl = vals_wl[np.isfinite(vals_wl)]
        if len(vals_wl) > 0:
            p5_profile[b]  = np.percentile(vals_wl, 5)
            p95_profile[b] = np.percentile(vals_wl, 95)
 
        # Bed level p95 (bar top)
        if bl_a is not None:
            vals_bl = bl_a[:, mask].ravel()
            vals_bl = vals_bl[np.isfinite(vals_bl)]
            if len(vals_bl) > 0:
                p95_bl_profile[b] = np.percentile(vals_bl, 95)
 
    results[label] = {'p5': p5_profile, 'p95': p95_profile, 'p95_bl': p95_bl_profile}
    ds.close()
    print(f"  Profile computed. p5  WL: [{np.nanmin(p5_profile):.2f}, {np.nanmax(p5_profile):.2f}] m")
    print(f"                    p95 WL: [{np.nanmin(p95_profile):.2f}, {np.nanmax(p95_profile):.2f}] m")
    print(f"                    p95 BL: [{np.nanmin(p95_bl_profile):.2f}, {np.nanmax(p95_bl_profile):.2f}] m")
 
# ---------------------------------------------------------------------------
# PLOT - FIGURE 1: WATER LEVEL ENVELOPE
# ---------------------------------------------------------------------------
 
fig1, axes1 = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    constrained_layout=True,
)
 
ax_p95, ax_p5 = axes1   # top = high water, bottom = low water
 
for label, prof in results.items():
    color = SCENARIO_COLORS.get(label, 'black')
    ls    = SCENARIO_LINESTYLE.get(label, '-')
    lw    = 1.5 if label == 'constant' else 2.0
 
    ax_p95.plot(bin_centres, prof['p95'], color=color, ls=ls, lw=lw)
    ax_p5.plot( bin_centres, prof['p5'],  color=color, ls=ls, lw=lw)
 
# --- Formatting Fig 1 ---
for ax, title in zip([ax_p95, ax_p5], ['p95 water level (high water)', 'p5 water level (low water)']):
    ax.set_ylabel('water level [m]')
    ax.set_title(title, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.7, ls=':')   # MSL reference
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
 
strip_top_right_spines(ax_p95, ax_p5)
 
ax_p5.set_xlabel('distance along estuary [km]')
ax_p5.set_xlim(X_MIN / 1000, X_MAX / 1000)
 
# --- Direct labels instead of a shared legend ---
add_direct_labels(
    ax_p95,
    [(bin_centres, prof['p95'], label, SCENARIO_COLORS.get(label, 'black')) for label, prof in results.items()],
)
add_direct_labels(
    ax_p5,
    [(bin_centres, prof['p5'], label, SCENARIO_COLORS.get(label, 'black')) for label, prof in results.items()],
)
 
fig1.suptitle(
    'Along-estuary water level envelope\n'
    r'$Q_r = 500\ \mathrm{m^3/s}$,  all timesteps (2 tidal cycles)'
)
 
# --- Save Fig 1 ---
out_path_wl = OUTPUT_DIR / OUTPUT_FILENAME_WL
fig1.savefig(out_path_wl, bbox_inches='tight')
print(f"\nSaved Figure 1: {out_path_wl}")
 
 
# ---------------------------------------------------------------------------
# PLOT - FIGURE 2: ACCOMMODATION SPACE
# ---------------------------------------------------------------------------
 
fig2, axes2 = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
 
ax_top, ax_bot = axes2
 
# --- Top Panel: Constant Flow ---
top_curves = []
if 'constant' in results:
    prof_c  = results['constant']
    color_c = SCENARIO_COLORS.get('constant', '#888888')
 
    # Plot Water Level & Bed Level
    ax_top.plot(bin_centres, prof_c['p95'], color=color_c, lw=1.5, ls='--')
    ax_top.plot(bin_centres, prof_c['p95_bl'], color='saddlebrown', lw=2.0)
 
    # Shade Accommodation Space
    ax_top.fill_between(bin_centres, prof_c['p95_bl'], prof_c['p95'], color=color_c, alpha=0.15)
 
    top_curves = [
        (bin_centres, prof_c['p95'], 'p95 water level', color_c),
        (bin_centres, prof_c['p95_bl'], 'p95 bed level (bar top)', 'saddlebrown'),
    ]
 
    # Formatting
    ax_top.set_ylabel('elevation [m]')
    ax_top.set_title(
        r"constant discharge $(Q_{\mathrm{mean}} = 500\ \mathrm{m^3/s})$",
        fontweight="bold",
    )
    ax_top.axhline(0, color='grey', lw=0.7, ls=':')
    ax_top.grid(True, lw=0.4, alpha=0.5)
    ax_top.yaxis.set_minor_locator(mticker.AutoMinorLocator())
 
 
# --- Bottom Panel: Hydro-variability (Low, Mean, Peak) ---
bot_curves = []
if 'peak flow' in results:
    prof_p = results['peak flow']
 
    # 1. Plot Bed Level first so it sits cleanly in the background/fill
    ax_bot.plot(bin_centres, prof_p['p95_bl'], color='saddlebrown', lw=2.0)
    bot_curves.append((bin_centres, prof_p['p95_bl'], 'p95 bed level (bar top)', 'saddlebrown'))
 
    # 2. Plot Low Flow Water Level
    if 'low flow' in results:
        ax_bot.plot(bin_centres, results['low flow']['p95'], color=SCENARIO_COLORS['low flow'], lw=1.5)
        bot_curves.append((bin_centres, results['low flow']['p95'], 'p95 water level (low flow)', SCENARIO_COLORS['low flow']))
 
    # 3. Plot Mean Flow Water Level
    if 'mean flow' in results:
        ax_bot.plot(bin_centres, results['mean flow']['p95'], color=SCENARIO_COLORS['mean flow'], lw=1.5)
        bot_curves.append((bin_centres, results['mean flow']['p95'], 'p95 water level (mean flow)', SCENARIO_COLORS['mean flow']))
 
    # 4. Plot Peak Flow Water Level
    ax_bot.plot(bin_centres, prof_p['p95'], color=SCENARIO_COLORS['peak flow'], lw=2.0)
    bot_curves.append((bin_centres, prof_p['p95'], 'p95 water level (peak flow)', SCENARIO_COLORS['peak flow']))
 
    # 5. Shade total maximum accommodation space (bounded by the Peak Flow high water mark)
    ax_bot.fill_between(bin_centres, prof_p['p95_bl'], prof_p['p95'], color=SCENARIO_COLORS['peak flow'], alpha=0.15)
 
    # Formatting
    ax_bot.set_ylabel('elevation [m]')
    ax_bot.set_title(
        r"variable discharge $(Q_{\mathrm{mean}} = 500\ \mathrm{m^3/s})$",
        fontweight="bold",
    )
    ax_bot.axhline(0, color='grey', lw=0.7, ls=':')
    ax_bot.grid(True, lw=0.4, alpha=0.5)
    ax_bot.yaxis.set_minor_locator(mticker.AutoMinorLocator())
 
strip_top_right_spines(ax_top, ax_bot)
 
# Global X-axis formatting
ax_bot.set_xlabel('distance along estuary [km]')
ax_bot.set_xlim(X_MIN / 1000, X_MAX / 1000)
 
# --- Direct labels instead of a shared legend ---
add_direct_labels(ax_top, top_curves)
add_direct_labels(ax_bot, bot_curves)
 
if 'constant' in results:
    add_fill_label(ax_top, bin_centres, prof_c['p95_bl'], prof_c['p95'], 'accommodation space', color_c)
if 'peak flow' in results:
    add_fill_label(ax_bot, bin_centres, prof_p['p95_bl'], prof_p['p95'], 'accommodation space', SCENARIO_COLORS['peak flow'])
 
# --- Save Fig 2 ---
out_path_accom = OUTPUT_DIR / OUTPUT_FILENAME_ACCOM
fig2.savefig(out_path_accom, bbox_inches='tight')
plt.show()
print(f"Saved Figure 2: {out_path_accom}")
 
# %%
