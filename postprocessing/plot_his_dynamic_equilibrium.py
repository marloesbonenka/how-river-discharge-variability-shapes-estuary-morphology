"""Dynamic equilibrium of the estuary (km20-45) from the his-file sediment budget.

Dynamic equilibrium: net export at the sea side equals net import plus river supply, i.e.
net export through km20 = river supply through km45 (no net storage in between).
Method: functions/F_dynamic_equilibrium.py (cumulative bed load through the his
observation cross-sections; positive = seaward; already x MorFac).

Runs: constant (pm1_n0) and R_peak = 5, n_peaks = 1 (pm5_n1), Q = 500 m3/s.
Two tests, both with tolerance EQ_TOL on net export / river supply:
  - per tidal cycle (ebb onset to ebb onset at km20), running mean over EQ_SMOOTH_YEARS
    morphological years; for the R_peak run this includes the flood-peak imbalance
  - per hydrograph year (calendar year of the run = MORFAC morphological years), the
    natural unit for a repeating hydrograph
Equilibrium time = first time after which the ratio stays within the tolerance until the
end of the run. Context printed as well: other seaward boundaries (km18, sea boundary),
storage per 5-km reach and a closure check against the map bed-volume change.

Bed-level change from the MAP output (after Braat et al., 2017, Fig. 3e, j), over the
active faces in km20-45 (bed changes > ACTIVE_CHANGE_M at some time, so including the
eroded banks), between consecutive map outputs, per morphological year:
  - morphodynamic activity = area-weighted mean |dz|
  - net bed-level change   = area-weighted mean dz (+ = accretion); = his storage / area

Layout (AGU style, Calibri 8 pt, 170 mm wide):
  (a) net export / river supply per tidal cycle (running mean) vs morphological time
  (b) net export / river supply per hydrograph year
  (c) morphodynamic activity
  (d) net bed-level change
"""
#%% --- Imports ---
from pathlib import Path
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from postprocessing/

from functions.F_general import apply_plot_style
from functions.F_loaddata import get_stitched_his_paths
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_dynamic_equilibrium import (
    SED_VAR,
    equilibrium_time,
    read_his_sections,
    running_mean,
    tidal_cycle_budget,
    yearly_budget,
)
from functions.F_morphological_activity import area_mean_bed_change

warnings.filterwarnings("ignore", category=UserWarning, module="xugrid")

#%% --- 1. SETTINGS ---
DISCHARGE = 500
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
base_path = BASE_DIR / f"Q{DISCHARGE}"

RPEAK = r'$R_{\mathrm{peak}}$'
NPEAKS = r'$n_{\mathrm{peaks}}$'
# label: (folder, colour) -- colours as in fig_bedlevel_percentiles_pm_n.py
RUNS = {
    'constant': ('01_Qr500_pm1_n0.9600302', '#7f7f7f'),
    f'{RPEAK} = 5, {NPEAKS} = 1': ('05_Qr500_pm5_n1.9517572', plt.cm.Blues(0.90)),
}

SEA_KM, RIVER_KM = 20, 45            # budget of the estuary between these his sections
CONTEXT_SEA = [18, 'ObservationCrossSection_SeaBnd_km0']   # other seaward boundaries (printed only)
REACH_EDGES_KM = [20, 25, 30, 35, 40, 45]
HIS_FILE = 'output/FlowFM_0000_his.nc'

MORFAC = 100
CDRYB = 1600.0                       # dry bed density [kg/m3] (.sed file): kg -> m3 of bed
EQ_SMOOTH_YEARS = 10.0               # running mean of the per-tidal-cycle fluxes [morph. years]
EQ_TOL = 0.10                        # equilibrium: |net export / river supply - 1| < EQ_TOL

# Map-based bed-level change (Braat et al., 2017, Fig. 3e, j), same estuary km20-45
# Active area: faces whose bed changes > ACTIVE_CHANGE_M at some time. This includes the
# eroded banks (initial land at ~9 m lowered to ~2 m: the estuary widens), which a mask
# on the initial channel would miss; the net change then closes with the his budget.
ACTIVE_CHANGE_M = 0.01
LAND_THRESHOLD = 6.0                 # [m] initial bed above this = land (only for the channel / bank split)
CACHE_BBOX = [1, 1, 45000, 15000]

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170
FONT_SIZE = 8
FIG_HEIGHT_IN = 4.6

apply_plot_style('AGU', font_size=FONT_SIZE, title_delta=0)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'dynamic_equilibrium'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / 'timed-out'


def _his_path(folder):
    """The single his file of a run; a run split over restarts is refused (cumulative
    transports restart from zero per part and would need stitching)."""
    parts = get_stitched_his_paths(base_path=base_path, folder_name=folder,
                                   timed_out_dir=timed_out_dir if timed_out_dir.exists() else None,
                                   variability_map=None, analyze_noisy=False)
    if parts and len(parts) > 1:
        raise NotImplementedError(f"{folder}: his output split over {len(parts)} parts")
    return Path(parts[0]) if parts else base_path / folder / HIS_FILE


def _label_km(km):
    return 'sea boundary' if isinstance(km, str) else f'km{km}'


#%% --- 2. BUDGETS PER RUN ---
results = {}
for label, (folder, color) in RUNS.items():
    his = _his_path(folder)
    print(f"\n===== {label}: {folder}\n  his file: {his}")
    cyc = tidal_cycle_budget(his, SEA_KM, RIVER_KM, MORFAC, CDRYB, extra_kms=CONTEXT_SEA)
    yrs = yearly_budget(his, SEA_KM, RIVER_KM, MORFAC, CDRYB,
                        extra_kms=CONTEXT_SEA + REACH_EDGES_KM[1:-1])

    n_win = max(1, int(round(EQ_SMOOTH_YEARS * 365.25 / MORFAC * 24 / cyc['period_h'].mean())))
    ratio_cyc = running_mean(cyc[SEA_KM], n_win) / running_mean(cyc[RIVER_KM], n_win)
    ratio_yr = yrs[SEA_KM] / yrs[RIVER_KM]
    results[label] = dict(color=color, cyc=cyc, yrs=yrs, ratio_cyc=ratio_cyc, ratio_yr=ratio_yr,
                          t_eq_cyc=equilibrium_time(cyc['morph_yr'], ratio_cyc, EQ_TOL),
                          t_eq_yr=equilibrium_time(yrs['yr0'], ratio_yr, EQ_TOL))
    r = results[label]

    print(f"  {len(cyc['morph_yr'])} tidal cycles of {cyc['period_h'].mean():.2f} h; running mean "
          f"{EQ_SMOOTH_YEARS:g} morph. yr = {n_win} cycles")
    print(f"  per tidal cycle: river supply {cyc[RIVER_KM].mean():.0f} m3, net export {cyc[SEA_KM].mean():.0f} m3")
    print("  per hydrograph year (m3 of bed per year, x MorFac):")
    print("    morph. yr | supply km45 | export km20 | ratio km20 | "
          + " | ".join(f"ratio {_label_km(k)}" for k in CONTEXT_SEA))
    for i in range(len(ratio_yr)):
        flag = '  <- within tolerance' if abs(ratio_yr[i] - 1) < EQ_TOL else ''
        print(f"    {r['yrs']['yr0'][i]:3.0f}-{r['yrs']['yr1'][i]:3.0f}   | {yrs[RIVER_KM][i]:11.3e} | "
              f"{yrs[SEA_KM][i]:11.3e} | {ratio_yr[i]:10.2f} | "
              + " | ".join(f"{yrs[k][i] / yrs[RIVER_KM][i]:{len('ratio ' + _label_km(k))}.2f}"
                           for k in CONTEXT_SEA) + flag)
    tot = {k: yrs[k].sum() for k in [SEA_KM, RIVER_KM] + CONTEXT_SEA + REACH_EDGES_KM[1:-1]}
    print(f"  whole run: storage km{SEA_KM}-{RIVER_KM} {tot[RIVER_KM] - tot[SEA_KM]:+.3e} m3; export / supply "
          + ", ".join(f"{_label_km(k)} {tot[k] / tot[RIVER_KM]:.2f}" for k in [SEA_KM] + CONTEXT_SEA))
    print("  storage per reach over the run [m3] (+ = deposition): "
          + ", ".join(f"km{a}-{b} {tot[b] - tot[a]:+.2e}"
                      for a, b in zip(REACH_EDGES_KM[:-1], REACH_EDGES_KM[1:])))
    for name, t_eq in (('per tidal cycle', r['t_eq_cyc']), ('per hydrograph year', r['t_eq_yr'])):
        print(f"  dynamic equilibrium {name} (within {EQ_TOL:.0%} until the end): "
              + (f"from {t_eq:.0f} morphological years" if t_eq is not None else "NOT reached"))

    # --- map: bed-level change of the estuary (Braat et al., 2017, Fig. 3e, j) ---
    ds = load_or_update_map_cache_multi(
        cache_dir=base_path / 'cached_data', folder_name=folder, run_paths=[base_path / folder],
        var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, None),
    )
    t_map = np.asarray(ds.time.values).astype('datetime64[ns]')
    order = np.unique(t_map, return_index=True)[1]              # the cache time axis is not sorted
    t_map = t_map[order]
    bed = ds['mesh2d_mor_bl'].transpose('time', ...).values[order]
    fc, area = np.asarray(ds.grid.face_coordinates), np.asarray(ds.grid.area)
    ds.close()
    zone = (fc[:, 0] >= SEA_KM * 1000) & (fc[:, 0] <= RIVER_KM * 1000)
    estuary = zone & (np.nanmax(np.abs(bed - bed[0]), axis=0) > ACTIVE_CHANGE_M)
    activity, net = area_mean_bed_change(bed, area, estuary)    # [m per output interval]
    for name, m in (('initial channel', estuary & (bed[0] < LAND_THRESHOLD)),
                    ('eroded banks (initial land)', estuary & (bed[0] >= LAND_THRESHOLD))):
        print(f"  {name}: {area[m].sum() / 1e6:.1f} km2, net volume change over the run "
              f"{np.nansum((bed[-1] - bed[0])[m] * area[m]):+.3e} m3")
    map_yr = (t_map - t_map[0]) / np.timedelta64(1, 'D') * MORFAC / 365.25
    dt_yr = np.diff(map_yr)
    r.update(map_yr=0.5 * (map_yr[1:] + map_yr[:-1]),
             activity=activity / dt_yr * 1000, net=net / dt_yr * 1000)   # [mm / morph. yr]

    # closure: his storage over the same intervals / estuary area vs the map net change
    t_his, cum = read_his_sections(his, [SEA_KM, RIVER_KM], SED_VAR)
    h_his = (t_his - t_his[0]) / np.timedelta64(1, 'h')
    h_map = (t_map - t_his[0]) / np.timedelta64(1, 'h')
    store = np.diff(np.interp(h_map, h_his, cum[RIVER_KM]) - np.interp(h_map, h_his, cum[SEA_KM])) / CDRYB
    a_est = area[estuary].sum()
    print(f"  map bed change km{SEA_KM}-{RIVER_KM} ({a_est / 1e6:.1f} km2, {len(t_map)} outputs, every "
          f"{dt_yr.mean():.1f} morph. yr): net {np.sum(net):+.3f} m over the run "
          f"({np.sum(net) * a_est:+.3e} m3), his storage {store.sum():+.3e} m3; per interval "
          f"max |map - his| {np.max(np.abs(net * a_est - store)):.2e} m3")
    late = r['map_yr'] > r['map_yr'][-1] - 200
    print(f"  morphodynamic activity: first 50 yr {r['activity'][r['map_yr'] < 50].mean():.1f}, last 200 yr "
          f"{r['activity'][late].mean():.1f} mm/yr; net change: first 50 yr "
          f"{r['net'][r['map_yr'] < 50].mean():+.2f}, last 200 yr {r['net'][late].mean():+.2f} mm/yr")


#%% --- 3. FIGURE ---
fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_IN), sharex=True,
                         gridspec_kw=dict(wspace=0.22, hspace=0.25))
(ax_c, ax_y), (ax_act, ax_net) = axes
for ax in (ax_c, ax_y):
    ax.axhspan(1 - EQ_TOL, 1 + EQ_TOL, color='0.85', lw=0)
    ax.axhline(1, color='k', lw=0.75)
ax_net.axhline(0, color='k', lw=0.75)

for label, r in results.items():
    # (a, b) his-file sediment budget
    ax_c.plot(r['cyc']['morph_yr'], r['ratio_cyc'], color=r['color'], lw=1.0)
    edges = np.append(r['yrs']['yr0'], r['yrs']['yr1'][-1])
    ax_y.stairs(r['ratio_yr'], edges, color=r['color'], lw=1.2, baseline=None)
    # (c, d) map bed-level change between consecutive outputs
    ax_act.plot(r['map_yr'], r['activity'], color=r['color'], lw=1.0)
    ax_net.plot(r['map_yr'], r['net'], color=r['color'], lw=1.0)

ax_y.sharey(ax_c)
ax_c.set_ylim(0, None)
ax_act.set_ylim(0, None)
ax_c.set_ylabel('net export / river supply [\u2013]')
ax_c.set_title(f'sediment budget per tidal cycle ({EQ_SMOOTH_YEARS:g}-yr running mean)')
ax_y.set_title(f'sediment budget per hydrograph year (= {MORFAC} yr)')
ax_act.set_ylabel('mean |bed level change| [mm/yr]')
ax_act.set_title('morphodynamic activity')
ax_net.set_ylabel('mean net bed level change [mm/yr]')
ax_net.set_title('net bed level change (+ accretion, \u2212 erosion)')
x_end = round(max(r['yrs']['yr1'][-1] for r in results.values()), -2)
for ax, letter in zip(axes.flat, 'abcd'):
    ax.set_xlim(0, x_end)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.grid(True, color='0.9', linewidth=0.5)
    ax.text(0.015, 0.95, f'({letter})', transform=ax.transAxes, ha='left', va='top')
for ax in axes[1]:
    ax.set_xlabel('morphological time [yr]')
ax_y.tick_params(labelleft=False)
fig.align_ylabels(axes[:, 0])

handles = [mlines.Line2D([], [], color=r['color'], lw=1.0, label=label) for label, r in results.items()]
handles.append(mpatches.Patch(facecolor='0.85', label=f'equilibrium \u00b1{EQ_TOL:.0%}'))
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles),
           frameon=True, handlelength=1.8, columnspacing=1.2)

save_name = f"AGU_dynamic_equilibrium_km{SEA_KM}-{RIVER_KM}_Q{DISCHARGE}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"\nSaved: {save_path}")

# %%
