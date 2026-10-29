"""Intertidal area and its elevation in one view: higher discharge peaks give more intertidal
area and raise its upper part, the raised bars.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one panel per discharge
(Q = 250 / 500 / 1000 m3/s), n_peaks = PROFILE_N series:
  Hypsometric curve of the intertidal zone in absolute area: the intertidal faces sorted by
  bed elevation, x = cumulative intertidal area [km2] (area lower than that elevation),
  y = bed elevation. The end of each curve (dot) is the total intertidal area, so a longer
  curve = more intertidal area, a higher curve = higher intertidal flats.
  Colours: constant run grey, R_peak Blues palette (light -> dark).

Data: intertidal_hypsometry_curves_long.csv (cumulative area fraction per run) and
intertidal_hypsometry_area_summary.csv (total intertidal area per run) of
figures/plot_map_hypsometry_intertidal_allQ.py (run that first). The intertidal zone is the
wet/dry classification of the final hours of the mean-flow detailed-hydro runs, the same as
the intertidal area in plot_map_hydro_tidalmetrics_comparison_allQ.py.
"""
#%% --- Imports ---
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style

#%% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
SRC_DIR = BASE_DIR / 'output_plots_combined' / 'intertidal_hypsometry_comparison'
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'intertidal_hypsometry_area'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISCHARGES = [250, 500, 1000]
PROFILE_N = 3                  # n_peaks of the R_peak series run at every Q

RPEAK = r'$R_{\mathrm{peak}}$'
GREY_CONST = "#7f7f7f"
LINE_WIDTH = 1.2
END_MARKER_SIZE = 3.5

# --- AGU figure sizing ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 65
FONT_SIZE = 8

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)


#%% --- 2. LOAD ---
curves = pd.read_csv(SRC_DIR / 'intertidal_hypsometry_curves_long.csv')
areas = pd.read_csv(SRC_DIR / 'intertidal_hypsometry_area_summary.csv')
curves = curves.merge(areas[['label', 'intertidal_area_km2']], on='label')
curves = curves[(curves.n == 0) | (curves.n == PROFILE_N)]
curves['cum_area_km2'] = curves.cum_area_frac * curves.intertidal_area_km2

q_list = [q for q in DISCHARGES if q in set(curves.discharge)]
all_pm = sorted(curves.loc[curves.n > 0, 'pm'].unique())
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / max(len(all_pm) - 1, 1)) for i, pm in enumerate(all_pm)}
PM_COLOR[1] = GREY_CONST

print("Total intertidal area [km2]:")
print(areas[(areas.n == 0) | (areas.n == PROFILE_N)]
      .pivot(index='pm', columns='discharge', values='intertidal_area_km2').round(2).to_string())


#%% --- 3. FIGURE ---
fig, axes = plt.subplots(1, len(q_list), sharey=True, squeeze=False,
                         figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
                         constrained_layout=True)
axes = axes[0]
for c, (q, ax) in enumerate(zip(q_list, axes)):
    for pm in [1] + all_pm:
        run = curves[(curves.discharge == q) & (curves.pm == pm)].sort_values('cum_area_km2')
        if run.empty:
            continue
        ax.plot(run.cum_area_km2, run.bed_level_m, color=PM_COLOR[pm], linewidth=LINE_WIDTH,
                linestyle='--' if pm == 1 else '-', zorder=3 if pm > 1 else 2)
        ax.plot(run.cum_area_km2.iloc[-1], run.bed_level_m.iloc[-1], 'o', color=PM_COLOR[pm],
                markersize=END_MARKER_SIZE, markeredgecolor='k', markeredgewidth=0.3, zorder=4)
    ax.set_xlim(left=0)
    ax.set_title(f'$Q$ = {q} m\u00b3/s')
    ax.set_xlabel('Cumulative intertidal area [km\u00b2]')
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(-0.02, 1.02, f'({chr(ord("a") + c)})', transform=ax.transAxes, ha='right', va='bottom')
    if c == 0:
        ax.set_ylabel('Bed elevation [m]')

legend_handles = [mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH, linestyle='--', label='constant')]
legend_handles += [mlines.Line2D([], [], color=PM_COLOR[pm], linewidth=LINE_WIDTH, label=f'{RPEAK} = {pm}')
                   for pm in all_pm]
legend_handles.append(mlines.Line2D([], [], color='0.5', marker='o', linestyle='', markersize=END_MARKER_SIZE,
                                    markeredgecolor='k', markeredgewidth=0.3, label='total intertidal area'))
fig.legend(handles=legend_handles, loc='outside lower center', ncol=len(legend_handles), frameon=True)

save_name = f"AGU_intertidal_hypsometry_area_allQ_n{PROFILE_N}"
fig.savefig(OUTPUT_DIR / f"{save_name}.png", dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {OUTPUT_DIR / save_name}.png")

# %%
