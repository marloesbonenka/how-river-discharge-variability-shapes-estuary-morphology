"""
Visualization: Discharge scenarios – one panel per n_peaks, amplitudes as colors
Author : Marloes Bonenkamp
Date   : April 2026

Creates one subplot per number-of-peaks value.  Within each panel all
peak / mean ratios are overlaid, each with a consistent color.

Input   : u:/…/Q500/  (scans sub-folders automatically)
Output  : u:/…/Q500/plots_river_bct/scenario_lines_Q500.png

Style mirrors plot_sensitivity_pm_n_bedlevel_15panel.py: AGU rcParams
(Calibri, 8pt), gridspec-based spacing with inch-based margins, and a
single shared legend at the bottom of the figure. Only one row of panels
is used here (one panel per n_peaks / per peak_ratio).
"""
#%%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# ============================================================================
# Configuration
# ============================================================================
TOTAL_Q    = 500          # m³/s  – used only for labels and output filename
BASE_DIR   = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Input\Q500"
)
OUTPUT_DIR  = BASE_DIR / "plots_river_bct"

# --- Constant scenario colour ---
GREY_CONST = "#7f7f7f"   # grey for the constant (pm1_n0) reference line

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170          # AGU full-page width

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

# Margins in inches (space outside the panels), same convention as the
# 15-panel sensitivity plot, but with a single row of panels here.
TOP_MARGIN_IN = 0.4        # space above the panels for the figure suptitle + subplot titles
BOTTOM_MARGIN_IN = 1.0      # space below the panels for x tick labels + supxlabel + legend
LEGEND_Y_IN = 0.05           # legend baseline, measured up from the figure's bottom edge
SUPXLABEL_Y_IN = 0.45        # supxlabel baseline, measured up from the figure's bottom edge (above the legend)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Discover scenario folders and parse peak_ratio / n_peaks
# ============================================================================
FOLDER_RE = re.compile(r"_pm(\d+(?:\.\d+)?)_n(\d+)$", re.IGNORECASE)

scenarios: dict[tuple, dict] = {}

for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue
    m = FOLDER_RE.search(folder.name)
    if not m:
        continue
    peak_ratio = float(m.group(1))
    n_peaks    = int(m.group(2))
    csv_path   = folder / "boundaryfiles_csv" / "discharge_cumulative.csv"
    if not csv_path.exists():
        print(f"  WARNING – CSV not found for '{folder.name}', skipping.")
        continue
    scenarios[(peak_ratio, n_peaks)] = {
        "name": folder.name,
        "csv":  csv_path,
    }

if not scenarios:
    raise FileNotFoundError(f"No valid scenario folders found under:\n  {BASE_DIR}")

print(f"Found {len(scenarios)} scenario(s).")

# ============================================================================
# Build sorted axis values
# ============================================================================
all_n_peaks     = sorted({k[1] for k in scenarios})   # one panel each
all_peak_ratios = sorted({k[0] for k in scenarios})   # one color each

print(f"  n_peaks     axis : {all_n_peaks}")
print(f"  peak_ratio  axis : {all_peak_ratios}")

# ============================================================================
# Consistent color map: one fixed color per peak_ratio value
# ============================================================================
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
]

_n_r = max(len(all_peak_ratios) - 1, 1)
RATIO_COLOR: dict[float, str] = {
    ratio: plt.cm.Blues(0.35 + 0.55 * i / _n_r)
    for i, ratio in enumerate(all_peak_ratios)
}

# ============================================================================
# Pre-compute global y-axis limits
# ============================================================================
global_ymin = np.inf
global_ymax = -np.inf

for key, info in scenarios.items():
    df_tmp = pd.read_csv(info["csv"])
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    first_yr = df_tmp["timestamp"].dt.year.min()
    q = df_tmp[df_tmp["timestamp"].dt.year == first_yr]["discharge_m3s"]
    global_ymin = min(global_ymin, q.min())
    global_ymax = max(global_ymax, q.max())

ypad = max((global_ymax - global_ymin) * 0.08, 1.0)
global_ylim = (global_ymin - ypad, global_ymax + ypad)

# ============================================================================
# Plot 1 – one panel per n_peaks, lines colored by peak_ratio
# ============================================================================
n_cols1 = len(all_n_peaks)

panel_width_in = FIGURE_WIDTH_MM * MM_TO_IN / n_cols1
panel_height_in = 0.8 * panel_width_in

fig1_width_in = panel_width_in * n_cols1
fig1_height_in = panel_height_in + TOP_MARGIN_IN + BOTTOM_MARGIN_IN

fig1 = plt.figure(figsize=(fig1_width_in, fig1_height_in))
gs1 = gridspec.GridSpec(
    1, n_cols1, figure=fig1,
    wspace=0.08,
    top=1 - TOP_MARGIN_IN / fig1_height_in,
    bottom=BOTTOM_MARGIN_IN / fig1_height_in,
)

axes1 = np.empty(n_cols1, dtype=object)
ax_ref1 = None
for c in range(n_cols1):
    ax = fig1.add_subplot(gs1[0, c], sharey=ax_ref1)
    if ax_ref1 is None:
        ax_ref1 = ax
    axes1[c] = ax

for ci, n_peaks in enumerate(all_n_peaks):
    ax = axes1[ci]
    for peak_ratio in all_peak_ratios:
        if (peak_ratio, n_peaks) not in scenarios:
            continue
        df = pd.read_csv(scenarios[(peak_ratio, n_peaks)]["csv"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df_yr = df[df["timestamp"].dt.year == df["timestamp"].dt.year.min()].copy()
        pr_label = f"{int(peak_ratio)}" if peak_ratio == int(peak_ratio) else f"{peak_ratio}"
        is_const = (peak_ratio == 1.0 and n_peaks == 0)
        ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
                color=GREY_CONST if is_const else RATIO_COLOR[peak_ratio],
                linewidth=0.9 if is_const else 1.0,
                linestyle='--' if is_const else '-',
                zorder=2 if is_const else 3,
                label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}")
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x")
    ax.set_title(f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}", fontsize=8)
    if ci == 0:
        ax.set_ylabel("Q [m³/s]", fontsize=8)
    else:
        ax.tick_params(labelleft=False)

fig1.legend(
    handles=[
        mlines.Line2D([], [], color=GREY_CONST if (r == 1.0 and min(all_n_peaks) == 0) else RATIO_COLOR[r],
                      linewidth=0.9 if (r == 1.0 and min(all_n_peaks) == 0) else 1.2,
                      linestyle='--' if (r == 1.0 and min(all_n_peaks) == 0) else '-',
                      label=(f"$R_{{\\mathrm{{peak}}}}$ = {int(r)}"
                             if r == int(r) else f"$R_{{\\mathrm{{peak}}}}$ = {r}")
                           + (" (constant)" if (r == 1.0 and min(all_n_peaks) == 0) else ""))
        for r in all_peak_ratios
    ],
    fontsize=8, loc="lower center", ncol=len(all_peak_ratios), bbox_to_anchor=(0.5, LEGEND_Y_IN / fig1_height_in), frameon=True,
)
fig1.supxlabel("month of year", y=SUPXLABEL_Y_IN / fig1_height_in)
fig1.suptitle(f"River discharge scenarios ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", fontsize=8, y=0.99)

fig1.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_frequency.png", dpi=300, bbox_inches="tight", transparent=True)
fig1.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_frequency.pdf", bbox_inches="tight", transparent=True)
plt.show(fig1)
print(f"Saved: scenario_lines_Q{TOTAL_Q}_by_frequency (.png/.pdf)")

# ============================================================================
# Plot 2 – one panel per peak_ratio, lines colored by n_peaks
# ============================================================================
_n_n = max(len(all_n_peaks) - 1, 1)
NPEAK_COLOR: dict[int, str] = {
    n: plt.cm.Greens(0.35 + 0.55 * i / _n_n)
    for i, n in enumerate(all_n_peaks)
}

n_cols2 = len(all_peak_ratios)

panel_width_in2 = FIGURE_WIDTH_MM * MM_TO_IN / n_cols2
panel_height_in2 = 0.8 * panel_width_in2

fig2_width_in = panel_width_in2 * n_cols2
fig2_height_in = panel_height_in2 + TOP_MARGIN_IN + BOTTOM_MARGIN_IN

fig2 = plt.figure(figsize=(fig2_width_in, fig2_height_in))
gs2 = gridspec.GridSpec(
    1, n_cols2, figure=fig2,
    wspace=0.08,
    top=1 - TOP_MARGIN_IN / fig2_height_in,
    bottom=BOTTOM_MARGIN_IN / fig2_height_in,
)

axes2 = np.empty(n_cols2, dtype=object)
ax_ref2 = None
for c in range(n_cols2):
    ax = fig2.add_subplot(gs2[0, c], sharey=ax_ref2)
    if ax_ref2 is None:
        ax_ref2 = ax
    axes2[c] = ax

for ci, peak_ratio in enumerate(all_peak_ratios):
    ax = axes2[ci]
    for n_peaks in all_n_peaks:
        if (peak_ratio, n_peaks) not in scenarios:
            continue
        df = pd.read_csv(scenarios[(peak_ratio, n_peaks)]["csv"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df_yr = df[df["timestamp"].dt.year == df["timestamp"].dt.year.min()].copy()
        is_const = (peak_ratio == 1.0 and n_peaks == 0)
        ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
                color=GREY_CONST if is_const else NPEAK_COLOR[n_peaks],
                linewidth=0.9 if is_const else 1.0,
                linestyle='--' if is_const else '-',
                zorder=2 if is_const else 3,
                label=f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}")
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x")
    pr_label = f"{int(peak_ratio)}" if peak_ratio == int(peak_ratio) else f"{peak_ratio}"
    ax.set_title(f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}", fontsize=8)
    if ci == 0:
        ax.set_ylabel("Q [m³/s]", fontsize=8)
    else:
        ax.tick_params(labelleft=False)

fig2.legend(
    handles=[
        mlines.Line2D([], [], color=GREY_CONST if (r == 1.0 and n == 0) else NPEAK_COLOR[n],
                      linewidth=0.9 if (r == 1.0 and n == 0) else 1.2,
                      linestyle='--' if (r == 1.0 and n == 0) else '-',
                      label=f"$n_{{\\mathrm{{peaks}}}}$ = {n}"
                           + (" (constant)" if (r == 1.0 and n == 0) else ""))
        for r in [min(all_peak_ratios)] for n in all_n_peaks
    ],
    fontsize=8, loc="lower center", ncol=len(all_n_peaks), bbox_to_anchor=(0.5, LEGEND_Y_IN / fig2_height_in), frameon=True,
)
fig2.supxlabel("month of year", y=SUPXLABEL_Y_IN / fig2_height_in)
fig2.suptitle(f"River discharge scenarios ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", fontsize=8, y=0.99)

fig2.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_amplitude.png", dpi=300, bbox_inches="tight", transparent=True)
fig2.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_amplitude.pdf", bbox_inches="tight", transparent=True)
plt.show(fig2)
print(f"Saved: scenario_lines_Q{TOTAL_Q}_by_amplitude (.png/.pdf)")
#%%
