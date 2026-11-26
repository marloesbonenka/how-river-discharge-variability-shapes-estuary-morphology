"""
Visualization: Discharge scenarios – one panel per n_peaks, amplitudes as colors
Author : Marloes Bonenkamp
Date   : April 2026

Creates one subplot per number-of-peaks value.  Within each panel all
peak / mean ratios are overlaid, each with a consistent color.

Input   : u:/…/Q500/  (scans sub-folders automatically)
Output  : u:/…/Q500/plots_river_bct/scenario_lines_Q500.png

Style mirrors postprocessing/fig_bedlevel_percentiles_pm_n.py: AGU rcParams
(Calibri, 8pt), gridspec-based spacing with inch-based margins, and a
single shared legend at the bottom of the figure. Only one row of panels
is used here (one panel per n_peaks / per peak_ratio).
"""
#%%
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from model_setup/

from functions.F_general import apply_plot_style

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

apply_plot_style('AGU', font_size=8)

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
    fontsize=8, loc="lower center", ncol=len(all_peak_ratios), bbox_to_anchor=(0.5, LEGEND_Y_IN / fig1_height_in), frameon=False,
)
fig1.supxlabel("month of year", y=SUPXLABEL_Y_IN / fig1_height_in)
fig1.suptitle(f"River discharge scenarios ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", fontsize=8, y=0.99)

fig1.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_frequency.png", dpi=300, bbox_inches="tight", transparent=True)
fig1.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_frequency.pdf", bbox_inches="tight", transparent=True)
plt.show()
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
    fontsize=8, loc="lower center", ncol=len(all_n_peaks), bbox_to_anchor=(0.5, LEGEND_Y_IN / fig2_height_in), frameon=False,
)
fig2.supxlabel("month of year", y=SUPXLABEL_Y_IN / fig2_height_in)
fig2.suptitle(f"River discharge scenarios ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", fontsize=8, y=0.99)

fig2.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_amplitude.png", dpi=300, bbox_inches="tight", transparent=True)
fig2.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_by_amplitude.pdf", bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: scenario_lines_Q{TOTAL_Q}_by_amplitude (.png/.pdf)")

#%% ============================================================================
# Plot 3 – one standalone single-panel plot per peak_ratio, for a fixed n_peaks
# ============================================================================
TARGET_N_PEAKS = 3
target_ratios = sorted(r for (r, n) in scenarios if n == TARGET_N_PEAKS and r != 1.0)

if not target_ratios:
    print(f"  WARNING – no scenarios found for n_peaks = {TARGET_N_PEAKS}; skipping individual plots.")

SINGLE_WIDTH_MM = 85          # AGU single-column width
single_width_in = SINGLE_WIDTH_MM * MM_TO_IN
single_height_in = 0.8 * single_width_in + TOP_MARGIN_IN + BOTTOM_MARGIN_IN

for peak_ratio in target_ratios:
    df = pd.read_csv(scenarios[(peak_ratio, TARGET_N_PEAKS)]["csv"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df_yr = df[df["timestamp"].dt.year == df["timestamp"].dt.year.min()].copy()

    fig3 = plt.figure(figsize=(single_width_in, single_height_in))
    gs3 = gridspec.GridSpec(
        1, 1, figure=fig3,
        top=1 - TOP_MARGIN_IN / single_height_in,
        bottom=BOTTOM_MARGIN_IN / single_height_in,
    )
    ax = fig3.add_subplot(gs3[0, 0])
    ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
            color=RATIO_COLOR[peak_ratio], linewidth=1.0, zorder=3)
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_ylabel("Q [m³/s]", fontsize=8)

    pr_label = f"{int(peak_ratio)}" if peak_ratio == int(peak_ratio) else f"{peak_ratio}"
    ax.set_title(f"$n_{{\\mathrm{{peaks}}}}$ = {TARGET_N_PEAKS}, $R_{{\\mathrm{{peak}}}}$ = {pr_label}", fontsize=8)
    fig3.supxlabel("month of year", y=SUPXLABEL_Y_IN / single_height_in)

    pr_tag = pr_label.replace(".", "p")
    fig3.savefig(OUTPUT_DIR / f"scenario_line_Q{TOTAL_Q}_n{TARGET_N_PEAKS}_pm{pr_tag}.png", dpi=300, bbox_inches="tight", transparent=True)
    fig3.savefig(OUTPUT_DIR / f"scenario_line_Q{TOTAL_Q}_n{TARGET_N_PEAKS}_pm{pr_tag}.pdf", bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved: scenario_line_Q{TOTAL_Q}_n{TARGET_N_PEAKS}_pm{pr_tag} (.png/.pdf)")

#%% ============================================================================
# Plot 4 – single panel with all peak_ratio scenarios overlaid, for a fixed n_peaks
# ============================================================================
if target_ratios:
    fig4 = plt.figure(figsize=(single_width_in, single_height_in))
    gs4 = gridspec.GridSpec(
        1, 1, figure=fig4,
        top=1 - TOP_MARGIN_IN / single_height_in,
        bottom=BOTTOM_MARGIN_IN / single_height_in,
    )
    ax = fig4.add_subplot(gs4[0, 0])
    for peak_ratio in target_ratios:
        df = pd.read_csv(scenarios[(peak_ratio, TARGET_N_PEAKS)]["csv"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df_yr = df[df["timestamp"].dt.year == df["timestamp"].dt.year.min()].copy()
        pr_label = f"{int(peak_ratio)}" if peak_ratio == int(peak_ratio) else f"{peak_ratio}"
        ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
                color=RATIO_COLOR[peak_ratio], linewidth=1.0, zorder=3,
                label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}")
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_ylabel("Q [m³/s]", fontsize=8)
    ax.set_title(f"$n_{{\\mathrm{{peaks}}}}$ = {TARGET_N_PEAKS}", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig4.supxlabel("month of year", y=SUPXLABEL_Y_IN / single_height_in)

    fig4.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_n{TARGET_N_PEAKS}_allratios.png", dpi=300, bbox_inches="tight", transparent=True)
    fig4.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_n{TARGET_N_PEAKS}_allratios.pdf", bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved: scenario_lines_Q{TOTAL_Q}_n{TARGET_N_PEAKS}_allratios (.png/.pdf)")

#%% ============================================================================
# Plot 5 – flow duration curves (cf. Arkesteijn et al., 2019): Q_w versus P
# ============================================================================
# P = probability of the flow rate over one full repeating cycle (365 daily values).
# EXCEEDANCE = True : P is the exceedance probability, P(Q > Q_w)  (standard flow-duration curve)
# EXCEEDANCE = False: P is the cumulative (non-exceedance) probability, P(Q <= Q_w)
EXCEEDANCE = True
P_LABEL = "P [-] (exceedance probability)" if EXCEEDANCE else "P [-] (cumulative probability)"
CYCLE_DAYS = 365   # length of the repeating discharge pattern in the boundary files


def flow_duration_curve(csv_path, n_days=CYCLE_DAYS):
    """Return (P, Q_w) for the first `n_days` of a discharge_cumulative.csv.

    Using exactly one cycle (and not calendar years) avoids counting a repeated
    day, since the pattern repeats every 365 days. Weibull plotting position.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    t0 = df["timestamp"].iloc[0]
    q = df.loc[df["timestamp"] < t0 + pd.Timedelta(days=n_days), "discharge_m3s"].to_numpy()
    q_desc = np.sort(q)[::-1]
    p_exc = np.arange(1, len(q_desc) + 1) / (len(q_desc) + 1)
    return (p_exc if EXCEEDANCE else 1 - p_exc), q_desc


def _pr_label(r):
    return f"{int(r)}" if r == int(r) else f"{r}"


def _plot_fdc(ax, peak_ratio, n_peaks, info):
    p, q = flow_duration_curve(info["csv"])
    is_const = (peak_ratio == 1.0 and n_peaks == 0)
    ax.plot(p, q,
            color=GREY_CONST if is_const else RATIO_COLOR[peak_ratio],
            linewidth=0.9 if is_const else 1.0,
            linestyle='--' if is_const else '-',
            zorder=2 if is_const else 3)


fdc_handles = [
    mlines.Line2D([], [],
                  color=GREY_CONST if (r == 1.0 and min(all_n_peaks) == 0) else RATIO_COLOR[r],
                  linewidth=0.9 if (r == 1.0 and min(all_n_peaks) == 0) else 1.2,
                  linestyle='--' if (r == 1.0 and min(all_n_peaks) == 0) else '-',
                  label=f"$R_{{\\mathrm{{peak}}}}$ = {_pr_label(r)}"
                        + (" (constant)" if (r == 1.0 and min(all_n_peaks) == 0) else ""))
    for r in all_peak_ratios
]

# --- 5a: one panel, all scenarios overlaid (curves with equal R_peak coincide) ---
fig5 = plt.figure(figsize=(single_width_in, single_height_in))
gs5 = gridspec.GridSpec(
    1, 1, figure=fig5,
    top=1 - TOP_MARGIN_IN / single_height_in,
    bottom=BOTTOM_MARGIN_IN / single_height_in,
)
ax = fig5.add_subplot(gs5[0, 0])
for (peak_ratio, n_peaks), info in sorted(scenarios.items()):
    _plot_fdc(ax, peak_ratio, n_peaks, info)
ax.axhline(TOTAL_Q, color="k", linewidth=0.4, linestyle=":", zorder=1)
ax.set_xlim(0, 1)
ax.set_ylim(global_ylim)
ax.grid(True, alpha=0.2, linewidth=0.3)
ax.set_ylabel("$Q_{\\mathrm{w}}$ [m³/s]", fontsize=8)
ax.set_title(f"Flow duration curves ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", fontsize=8)
ax.legend(handles=fdc_handles, fontsize=6, loc="upper right", frameon=False)
fig5.supxlabel(P_LABEL, y=SUPXLABEL_Y_IN / single_height_in)
fig5.savefig(OUTPUT_DIR / f"flow_duration_curves_Q{TOTAL_Q}.png", dpi=300, bbox_inches="tight", transparent=True)
fig5.savefig(OUTPUT_DIR / f"flow_duration_curves_Q{TOTAL_Q}.pdf", bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: flow_duration_curves_Q{TOTAL_Q} (.png/.pdf)")

# --- 5b: one panel per n_peaks, colored by peak_ratio (same layout as Plot 1) ---
fig6 = plt.figure(figsize=(fig1_width_in, fig1_height_in))
gs6 = gridspec.GridSpec(
    1, n_cols1, figure=fig6,
    wspace=0.08,
    top=1 - TOP_MARGIN_IN / fig1_height_in,
    bottom=BOTTOM_MARGIN_IN / fig1_height_in,
)
ax_ref6 = None
for ci, n_peaks in enumerate(all_n_peaks):
    ax = fig6.add_subplot(gs6[0, ci], sharey=ax_ref6)
    if ax_ref6 is None:
        ax_ref6 = ax
    for peak_ratio in all_peak_ratios:
        if (peak_ratio, n_peaks) in scenarios:
            _plot_fdc(ax, peak_ratio, n_peaks, scenarios[(peak_ratio, n_peaks)])
    ax.axhline(TOTAL_Q, color="k", linewidth=0.4, linestyle=":", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0.25, 0.5, 0.75])   # avoid touching tick labels between panels
    ax.tick_params(axis="x", labelsize=6)
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.set_title(f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}", fontsize=8)
    if ci == 0:
        ax.set_ylabel("$Q_{\\mathrm{w}}$ [m³/s]", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
fig6.legend(handles=fdc_handles, fontsize=8, loc="lower center", ncol=len(all_peak_ratios),
            bbox_to_anchor=(0.5, LEGEND_Y_IN / fig1_height_in), frameon=False)
fig6.supxlabel(P_LABEL, y=SUPXLABEL_Y_IN / fig1_height_in)
fig6.suptitle(f"Flow duration curves ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", fontsize=8, y=0.99)
fig6.savefig(OUTPUT_DIR / f"flow_duration_curves_Q{TOTAL_Q}_by_frequency.png", dpi=300, bbox_inches="tight", transparent=True)
fig6.savefig(OUTPUT_DIR / f"flow_duration_curves_Q{TOTAL_Q}_by_frequency.pdf", bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: flow_duration_curves_Q{TOTAL_Q}_by_frequency (.png/.pdf)")
#%%
