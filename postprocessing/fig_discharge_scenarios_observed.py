"""Idealized river discharge scenarios, with observed rivers placed in the same parameter space.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide):
  (a)  top row, full width: the idealized scenarios (Q_mean = 500 m3/s), one panel per
       n_peaks, lines coloured by R_peak (Blues; constant run dashed grey). Same content
       as Plot 1 of model_setup/plot_scenario_lines.py.
  (b)  Chao Phraya and (c) Tay (Firth of Tay): daily WBMsed discharge 1980-2010 (Nienhuis et al., 2020
       extraction, see global_data_analysis/run_estuary_discharge_analysis.py) with the annual
       maxima, mean Q and mean annual maximum. Same content as plot_annual_max_peaks().
  (d)  parameter space n_peaks x R_peak: the simulated scenarios (squares, coloured as in (a)),
       the rivers of (b) and (c) (orange diamond / triangle), the LABELLED_D rivers (own marker,
       labelled directly) and all other estuaries in CACHE_FILE except EXCLUDE (small light-orange dots).

Observed metrics, defined as for the model forcing (all peaks of a model year are equally high):
  R_peak  = mean annual maximum / mean discharge (calendar years, as in
            run_estuary_discharge_analysis.py).
  n_peaks = mean number of flood peaks per water year. Per water year (starting in the month
            with the lowest mean discharge, so a wet season is not split at New Year), the
            discharge is smoothed with an N_PEAKS_SMOOTH_DAYS moving mean and a peak counts when
              - it rises at least N_PEAKS_FRACTION of the way from Q_mean to that year's maximum,
              - its prominence is at least that same excess (flow drops well between two peaks),
              - it lies at least N_PEAKS_MIN_SEP_DAYS from a higher peak.
            The annual maximum always counts, so n_peaks >= 1. Incomplete water years at the
            ends of the record are skipped. The metrics are written to <SAVE_NAME>_metrics.csv.
"""
#%% --- Imports ---
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style, save_agu_figure

#%% --- 1. SETTINGS ---
TOTAL_Q = 500   # m3/s, discharge of the scenario set shown in (a) and (d)
SCENARIO_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
                    rf"\Model_Input\Q{TOTAL_Q}")
CACHE_FILE = Path(r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_variability_WBMsed\estuary_timeseries_cache.xlsx")
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
                  r"\Model_Output\output_plots_combined\discharge_scenarios_observed")
SAVE_NAME = f"discharge_scenarios_observed_Q{TOTAL_Q}"

HIGHLIGHT = ['Chao Phraya', 'Firth of Tay']
DISPLAY_NAMES = {'Chao Phraya': 'Chao Phraya (TH)',   # name shown in the figure, if different
                 'Firth of Tay': 'Tay (UK)'}          # from the cache sheet name
EXCLUDE = ['Colorado (MX)']   # left out of (d) and the metrics CSV   # panels (b), (c), labelled in (d); sheet 'Q_<name>' in CACHE_FILE

# n_peaks definition (see docstring)
N_PEAKS_SMOOTH_DAYS = 7
N_PEAKS_FRACTION = 0.5
N_PEAKS_MIN_SEP_DAYS = 30

HIGHLIGHT_MARKERS = ['D', '^']   # (d): markers of the HIGHLIGHT rivers
LABELLED_D = {}   # (d) only, e.g. {'Yangon': 'v'}: extra rivers with their own marker + direct label

# --- Colours ---
GREY_CONST = "#7f7f7f"        # constant scenario (R_peak = 1, n_peaks = 0)
COLOR_DISCHARGE = '#363732'   # observed daily discharge + mean-Q line
COLOR_OBSERVED = '#BE5A38'    # annual maxima, mean-annual-max line, observed rivers in (d)
COLOR_OTHER = tuple(0.45 * c + 0.55 for c in mcolors.to_rgb(COLOR_OBSERVED))   # (d): other estuaries, opaque tint

RPEAK = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'

# --- Layout, all in inches (AGU full page width: 170 mm) ---
MM_TO_IN = 1 / 25.4
FIG_W = 170 * MM_TO_IN
LEFT = 0.52          # room for y tick labels + y label
RIGHT = 0.06
TOP = 0.36           # room for the (a) title + the n_peaks panel titles
A_H = 0.95           # height of the scenario panels
A_WSPACE = 0.06      # gap between scenario panels
A_BELOW = 0.42       # under (a): x tick labels + x label
A_LEGEND_W = 0.92    # right of (a): its R_peak legend (one column)
ROW_GAP = 0.22       # between the (a) legend and the (b)/(d) titles
TS_H = 0.92          # height of each observed time series panel
TS_GAP = 0.28        # between (b) and (c): room for the (c) title
TS_LABEL_W = 0.95    # right of (b)/(c): direct labels of the dashed lines
D_W = 1.75           # width of the parameter space panel
D_LEFT_GAP = 0.36    # left of (d): its y tick labels + y label
BOTTOM = 0.42        # under (c): x tick labels + x label

#%% --- 2. LOAD SCENARIOS ---
FOLDER_RE = re.compile(r"_pm(\d+(?:\.\d+)?)_n(\d+)$", re.IGNORECASE)


def read_first_year(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values('timestamp')
    return df[df['timestamp'].dt.year == df['timestamp'].dt.year.min()]


scenarios = {}   # (R_peak, n_peaks) -> first year of discharge_cumulative.csv
for folder in sorted(SCENARIO_DIR.iterdir()):
    m = FOLDER_RE.search(folder.name)
    csv_path = folder / "boundaryfiles_csv" / "discharge_cumulative.csv"
    if folder.is_dir() and m and csv_path.exists():
        scenarios[(float(m.group(1)), int(m.group(2)))] = read_first_year(csv_path)
if not scenarios:
    raise FileNotFoundError(f"No scenario folders found under {SCENARIO_DIR}")

all_n_peaks = sorted({n for _, n in scenarios})
all_ratios = sorted({r for r, _ in scenarios})
print(f"Found {len(scenarios)} scenarios; n_peaks {all_n_peaks}, R_peak {all_ratios}")

_n_r = max(len(all_ratios) - 1, 1)
RATIO_COLOR = {r: plt.cm.Blues(0.35 + 0.55 * i / _n_r) for i, r in enumerate(all_ratios)}


def is_const(r, n):
    return r == 1.0 and n == 0


def ratio_color(r, n=None):
    return GREY_CONST if (r == 1.0 and (n is None or n == 0)) else RATIO_COLOR[r]


def fmt_ratio(r):
    return f"{int(r)}" if r == int(r) else f"{r}"


#%% --- 3. LOAD OBSERVED DISCHARGE + METRICS ---
cache = pd.read_excel(CACHE_FILE, sheet_name=None, index_col=0)


def peaks_per_water_year(q):
    """Mean number of flood peaks per water year (definition in the module docstring)."""
    smooth = q.rolling(N_PEAKS_SMOOTH_DAYS, center=True, min_periods=1).mean()
    q_mean = q.mean()
    start_month = q.groupby(q.index.month).mean().idxmin()   # driest month
    water_year = smooth.index.year - (smooth.index.month < start_month)
    counts = []
    for _, g in smooth.groupby(water_year):
        if len(g) < 360:   # incomplete water year at the start/end of the record
            continue
        excess = N_PEAKS_FRACTION * (g.max() - q_mean)
        if excess <= 0:    # dry year, the annual maximum stays below Q_mean
            counts.append(1)
            continue
        vals = g.to_numpy()
        padded = np.r_[vals.min(), vals, vals.min()]   # a peak on the first/last day still counts
        peaks, _ = find_peaks(padded, height=q_mean + excess, prominence=excess,
                              distance=N_PEAKS_MIN_SEP_DAYS)
        counts.append(max(len(peaks), 1))
    return float(np.mean(counts))


observed = {}
for sheet, df in cache.items():
    name = sheet[2:]
    if not sheet.startswith('Q_') or name in EXCLUDE:
        continue
    q = df.squeeze().dropna()
    q.index = pd.to_datetime(q.index)
    peak_dates = q.groupby(q.index.year).idxmax()
    mean_q = q.mean()
    mean_annual_max = q[peak_dates].mean()
    observed[name] = dict(
        q=q, peak_dates=peak_dates, mean_q=mean_q, mean_annual_max=mean_annual_max,
        r_peak=mean_annual_max / mean_q, n_peaks=peaks_per_water_year(q),
    )
missing = [name for name in HIGHLIGHT if name not in observed]
if missing:
    raise KeyError(f"Not in {CACHE_FILE.name}: {missing}")

metrics = pd.DataFrame(
    [(name, o['mean_q'], o['mean_annual_max'], o['r_peak'], o['n_peaks']) for name, o in observed.items()],
    columns=['estuary', 'Q_mean_m3s', 'mean_annual_max_m3s', 'R_peak', 'n_peaks'],
).sort_values('R_peak')
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

#%% --- 4. FIGURE ---
apply_plot_style('AGU', font_size=8)

fig_h = TOP + A_H + A_BELOW + ROW_GAP + 2 * TS_H + TS_GAP + BOTTOM
fig = plt.figure(figsize=(FIG_W, fig_h))


def add_axes_in(left, bottom, width, height, **kw):
    """Axes positioned in inches from the lower-left corner of the figure."""
    return fig.add_axes([left / FIG_W, bottom / fig_h, width / FIG_W, height / fig_h], **kw)


def panel_label(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc='left', pad=4)


# --- (a) idealized scenarios ---
a_bottom = fig_h - TOP - A_H
n_cols = len(all_n_peaks)
a_w = (FIG_W - LEFT - RIGHT - A_LEGEND_W - (n_cols - 1) * A_WSPACE) / n_cols
a_right = LEFT + n_cols * a_w + (n_cols - 1) * A_WSPACE
ymax = max(df['discharge_m3s'].max() for df in scenarios.values())
axes_a = []
for ci, n in enumerate(all_n_peaks):
    ax = add_axes_in(LEFT + ci * (a_w + A_WSPACE), a_bottom, a_w, A_H,
                     sharey=axes_a[0] if axes_a else None)
    axes_a.append(ax)
    for r in all_ratios:
        if (r, n) not in scenarios:
            continue
        df = scenarios[(r, n)]
        ax.plot(df['timestamp'], df['discharge_m3s'], color=ratio_color(r, n),
                linewidth=0.9 if is_const(r, n) else 1.0, linestyle='--' if is_const(r, n) else '-',
                zorder=2 if is_const(r, n) else 3)
    ax.set_ylim(0, ymax * 1.08)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 5, 8, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.set_title(f"{NPEAKS} = {n}", pad=3)
    if ci == 0:
        ax.set_ylabel('$Q_{\\mathrm{river}}$ [m³/s]')
    else:
        ax.tick_params(labelleft=False)
fig.text(LEFT / FIG_W, (fig_h - 0.04) / fig_h,
         f"(a)  Idealized discharge scenarios ($Q_{{\\mathrm{{mean}}}}$ = {TOTAL_Q} m³/s)", ha='left', va='top')
fig.text((LEFT + a_right) / 2 / FIG_W, (a_bottom - 0.30) / fig_h, 'month of year',
         ha='center', va='center')

legend_handles = [
    mlines.Line2D([], [], color=ratio_color(r), linewidth=0.9 if r == 1.0 else 1.2,
                  linestyle='--' if r == 1.0 else '-',
                  label=f"{RPEAK} = {fmt_ratio(r)}")
    for r in all_ratios
]
fig.legend(handles=legend_handles, loc='center left', ncol=1, frameon=True, labelspacing=0.35,
           handlelength=1.6, borderaxespad=0,
           bbox_to_anchor=((a_right + 0.08) / FIG_W, (a_bottom + A_H / 2) / fig_h))

# --- (b), (c) observed discharge ---
ts_w = FIG_W - LEFT - TS_LABEL_W - D_LEFT_GAP - D_W - RIGHT
c_bottom = BOTTOM
b_bottom = BOTTOM + TS_H + TS_GAP
ax_b = add_axes_in(LEFT, b_bottom, ts_w, TS_H)
ax_c = add_axes_in(LEFT, c_bottom, ts_w, TS_H, sharex=ax_b)

for ax, letter, name in zip((ax_b, ax_c), 'bc', HIGHLIGHT):
    o = observed[name]
    q = o['q']
    ax.plot(q.index, q.values, color=COLOR_DISCHARGE, linewidth=0.5)
    ax.scatter(o['peak_dates'], q[o['peak_dates']], color=COLOR_OBSERVED, s=8, zorder=5, linewidths=0)
    ax.axhline(o['mean_q'], linestyle='--', linewidth=0.75, color=COLOR_DISCHARGE)
    ax.axhline(o['mean_annual_max'], linestyle='--', linewidth=0.75, color=COLOR_OBSERVED)
    ax.set_ylim(0, q.max() * 1.08)
    ax.set_ylabel('$Q_{\\mathrm{river}}$ [m³/s]')
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    panel_label(ax, letter, f"{DISPLAY_NAMES.get(name, name)}   ({NPEAKS} = {o['n_peaks']:.1f} yr$^{{-1}}$)")

    # direct labels of the two dashed lines, right of the axes: the upper line labelled above it,
    # the lower one below it, so they never overlap when the lines are close (Colorado)
    for y, text, color, va, dy in [
        (o['mean_annual_max'], f"mean annual max\n({RPEAK} = {o['r_peak']:.1f})", COLOR_OBSERVED, 'bottom', 1),
        (o['mean_q'], f"mean $Q$ = {o['mean_q']:.0f} m³/s", COLOR_DISCHARGE, 'top', -1),
    ]:
        ax.annotate(text, xy=(1, y), xycoords=('axes fraction', 'data'), xytext=(4, dy),
                    textcoords='offset points', ha='left', va=va, color=color,
                    annotation_clip=False)

ax_c.xaxis.set_major_locator(mdates.YearLocator(5))
ax_c.xaxis.set_minor_locator(mdates.YearLocator(1))
ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_c.set_xlim(pd.Timestamp('1979-07-01'), pd.Timestamp('2011-07-01'))
ax_b.tick_params(labelbottom=False)
ax_c.set_xlabel('year')

# --- (d) parameter space ---
d_left = FIG_W - RIGHT - D_W
ax_d = add_axes_in(d_left, BOTTOM, D_W, 2 * TS_H + TS_GAP)
for (r, n) in scenarios:
    ax_d.scatter(n, r, marker='s', s=34, color=ratio_color(r, n), edgecolors='white', linewidths=0.5, zorder=3)

for name, o in observed.items():
    if name not in HIGHLIGHT and name not in LABELLED_D:
        ax_d.scatter(o['n_peaks'], o['r_peak'], marker='o', s=14, color=COLOR_OTHER,
                     linewidths=0, zorder=4)
for name, marker in zip(HIGHLIGHT, HIGHLIGHT_MARKERS):
    o = observed[name]
    ax_d.scatter(o['n_peaks'], o['r_peak'], marker=marker, s=22, color=COLOR_OBSERVED,
                 edgecolors=COLOR_DISCHARGE, linewidths=0.5, zorder=5)
for name, marker in LABELLED_D.items():
    o = observed[name]
    ax_d.scatter(o['n_peaks'], o['r_peak'], marker=marker, s=22, color=COLOR_OBSERVED,
                 edgecolors=COLOR_DISCHARGE, linewidths=0.5, zorder=5)
    ax_d.annotate(name, xy=(o['n_peaks'], o['r_peak']), xytext=(5, 0), textcoords='offset points',
                  ha='left', va='center', color=COLOR_OBSERVED)

r_axis_max = max(max(all_ratios), max(o['r_peak'] for o in observed.values())) + 0.5
ax_d.set_xlim(-0.5, max(all_n_peaks) + 0.5)
ax_d.set_ylim(0.5, r_axis_max)
ax_d.set_xticks(range(0, max(all_n_peaks) + 1))
ax_d.set_yticks(range(1, int(r_axis_max) + 1))
ax_d.set_xlabel(f"{NPEAKS} [yr$^{{-1}}$]")
ax_d.set_ylabel(f"{RPEAK} [-]")
ax_d.grid(True, alpha=0.2, linewidth=0.3)
ax_d.spines[['top', 'right']].set_visible(False)
panel_label(ax_d, 'd', 'Parameter space')
ax_d.legend(handles=[
    mlines.Line2D([], [], marker='s', linestyle='none', markersize=5, color=RATIO_COLOR[all_ratios[-1]],
                  markeredgecolor='white', label='simulated'),
    *[mlines.Line2D([], [], marker=marker, linestyle='none', markersize=4.5, color=COLOR_OBSERVED,
                    markeredgecolor=COLOR_DISCHARGE, markeredgewidth=0.5, label=DISPLAY_NAMES.get(name, name))
      for name, marker in zip(HIGHLIGHT, HIGHLIGHT_MARKERS)],
    mlines.Line2D([], [], marker='o', linestyle='none', markersize=4, color=COLOR_OTHER,
                  markeredgewidth=0, label='other estuaries'),
], loc='upper right', frameon=True, handletextpad=0.3, borderpad=0.4, labelspacing=0.35)

#%% --- 5. SAVE ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_agu_figure(fig, OUTPUT_DIR / SAVE_NAME)
try:
    metrics.to_csv(OUTPUT_DIR / f"{SAVE_NAME}_metrics.csv", index=False, float_format='%.3f')
except PermissionError:
    print(f"  WARNING: {SAVE_NAME}_metrics.csv is open elsewhere (Excel?); metrics not written.")
plt.show()
