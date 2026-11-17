"""Intertidal hypsometry: higher discharge peaks raise the intertidal surface.

Layout (AGU style, Calibri 8 pt for all text, 170 mm wide), one row, n_peaks = PROFILE_N series:
  (a-c) one panel per discharge (Q = 250 / 500 / 1000 m3/s): non-dimensional hypsometric
        curve of the subaerially exposed (= intertidal) area: the exposed faces sorted by bed
        elevation, x = Ai / Atot (cumulative area fraction lower than that elevation),
        y = bed elevation. A higher curve = higher flats, independent of the total exposed
        area (printed to the console, not shown).
  Colours: constant run grey, R_peak Blues palette (light -> dark).
  The tidal prism is not in this figure: it is panel (d) of fig_raised_bars_tidal_intrusion_prism_allQ.py.

Data: intertidal_hypsometry_curves_long.csv (cumulative area fraction per run) and
intertidal_hypsometry_area_summary.csv (total intertidal area per run) of
postprocessing/plot_map_hypsometry_intertidal_allQ.py (run that first). Both come from the
mean-flow detailed-hydro runs on the final morphology; the intertidal zone is the wet/dry
classification of their final 12 hours.
"""
#%% --- Imports ---
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style, save_agu_figure

#%% --- 1. SETTINGS ---
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
HYPSO_DIR = BASE_DIR / 'output_plots_combined' / 'intertidal_hypsometry_comparison'
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'intertidal_hypsometry_allQ'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISCHARGES = [250, 500, 1000]
PROFILE_N = 3                  # n_peaks of the R_peak series run at every Q

RPEAK = r'$R_{\mathrm{peak}}$'
GREY_CONST = "#7f7f7f"
LINE_WIDTH = 1.2
LINE_WIDTH_CONST = 0.9

# --- AGU figure sizing ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FIGURE_HEIGHT_MM = 70
FONT_SIZE = 8

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)


#%% --- 2. LOAD ---
curves = pd.read_csv(HYPSO_DIR / 'intertidal_hypsometry_curves_long.csv')
areas = pd.read_csv(HYPSO_DIR / 'intertidal_hypsometry_area_summary.csv')
curves = curves[(curves.n == 0) | (curves.n == PROFILE_N)]

q_list = [q for q in DISCHARGES if q in set(curves.discharge)]
all_pm = sorted(curves.loc[curves.n > 0, 'pm'].unique())
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / max(len(all_pm) - 1, 1)) for i, pm in enumerate(all_pm)}
PM_COLOR[1] = GREY_CONST

sel = areas[(areas.n == 0) | (areas.n == PROFILE_N)]
print("Total intertidal area [km2]:")
print(sel.pivot(index='pm', columns='discharge', values='intertidal_area_km2').round(2).to_string())
for frac in (0.5, 0.9, 1.0):
    print(f"\nBed elevation at Ai/Atot = {frac:g} [m]:")
    print(curves.groupby(['pm', 'discharge'])
          .apply(lambda r: np.interp(frac, r.cum_area_frac, r.bed_level_m))
          .unstack().round(2).to_string())


#%% --- 3. FIGURE ---
fig, axes = plt.subplots(1, len(q_list), sharey=True, squeeze=False,
                         figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
                         constrained_layout=True)
hyp_axes = axes[0]

# (a-c) hypsometry, one panel per discharge
for c, (q, ax) in enumerate(zip(q_list, hyp_axes)):
    for pm in [1] + all_pm:
        run = curves[(curves.discharge == q) & (curves.pm == pm)].sort_values('cum_area_frac')
        if run.empty:
            continue
        ax.plot(run.cum_area_frac, run.bed_level_m, color=PM_COLOR[pm],
                linewidth=LINE_WIDTH_CONST if pm == 1 else LINE_WIDTH,
                linestyle='--' if pm == 1 else '-', zorder=3 if pm > 1 else 2)
    ax.set_xlim(0, 1)
    ax.set_title(f'$Q$ = {q} m\u00b3/s')
    ax.set_xlabel('Exposed area fraction\n$A_i / A_{\\mathrm{tot}}$ [\u2013]')
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(-0.02, 1.02, f'({chr(ord("a") + c)})', transform=ax.transAxes, ha='right', va='bottom')
hyp_axes[0].set_ylabel('Bed elevation [m]')

legend_handles = [mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                                label='constant')]
legend_handles += [mlines.Line2D([], [], color=PM_COLOR[pm], linewidth=LINE_WIDTH, label=f'{RPEAK} = {pm}')
                   for pm in all_pm]
hyp_axes[0].legend(handles=legend_handles, loc='upper left', ncol=1, frameon=True, framealpha=1,
                   borderaxespad=0.3, handlelength=1.6, handletextpad=0.4, labelspacing=0.2, borderpad=0.3)

save_agu_figure(fig, OUTPUT_DIR / f"AGU_intertidal_hypsometry_allQ_n{PROFILE_N}")
plt.show()

# %%
