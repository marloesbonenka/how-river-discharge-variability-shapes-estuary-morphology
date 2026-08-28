"""
Non-dimensional hypsometry of the INTERTIDAL ZONE, compared across scenarios.

Motivation
----------
Intertidal area increases with R_peak (pm). This script tests whether that
increase in *area* is accompanied by a change in *shape* (hypsometry) of the
intertidal zone, or whether the zone simply scales up self-similarly.

Method
------
1. For every scenario folder, classify intertidal faces with the *same*
   wet/dry logic used for the intertidal-area-vs-R_peak plots (last
   IA_WINDOW_HOURS of the run, restricted to the tidal zone in x). This
   guarantees the hypsometry corresponds exactly to the area already reported
   elsewhere.
2. Pull the bed level (mesh2d_mor_bl, final timestep) for exactly those
   intertidal faces.
3. Build an area-weighted, non-dimensional hypsometric curve:
       x = Ai / Atot   (cumulative intertidal area fraction, 0 -> 1)
       y = bed level, sorted ascending
   Atot is the total intertidal area of that scenario (area-weighted sum of
   mesh2d_flowelem_ba over the intertidal mask) -- NOT a shared reference
   area -- so curves from scenarios with very different absolute intertidal
   areas can be compared on the same [0, 1] axis.
4. Plot one figure per manuscript panel: 1 row x N discharges (Q = 250, 500,
   1000 m3/s left -> right), sharing the y-axis, one hypsometric curve per
   scenario (pm, n) within each Q panel. Colour encodes R_peak (pm) via a
   colourbar; linestyle encodes n_peaks; the constant (n=0) run is drawn as the reference curve.

Data source: same MAP cache as compute_hydro_metrics_allQ.py (this script
adds mesh2d_mor_bl to the same cache entry via append_vars).
"""

# %% IMPORTS
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_general import DHR_FOLDER_RE, discover_variability_scenario_folders, get_last_n_hours_window
from functions.F_hypsometry import classify_intertidal_mask, compute_intertidal_hypsometry

# =============================================================================
# %% --- CONFIGURATION ---
# =============================================================================
BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")
DISCHARGES = [250, 500, 1000]
RUN_IDS_TO_INCLUDE = {1, 6, 9, 10, 11}

# Constant scenario colour
GREY_CONST = "#7f7f7f"

# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}   (same as compute_hydro_metrics_allQ.py)
_FOLDER_RE = DHR_FOLDER_RE

# MAP cache settings -- identical bbox to the intertidal-area / LTI scripts so
# this reuses the same cache file (append_vars adds mesh2d_mor_bl if missing)
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
LOAD_VARS = ['mesh2d_waterdepth', 'mesh2d_flowelem_ba', 'mesh2d_mor_bl']

# Intertidal classification -- MUST match compute_hydro_metrics_allQ.py so the
# hypsometry corresponds to the intertidal area already reported there
IA_WET_THRESHOLD = 0.0001   # [m] Epshu in Delft3D-FM
IA_WINDOW_HOURS = 12.0      # [h] 1 tidal cycle, taken from end of run
IA_X_MIN = 20000.0          # [m] 
IA_X_MAX = 45000.0          # [m]
IA_FACE_X_VAR = 'mesh2d_face_x'

# Output
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'intertidal_hypsometry_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure / AGU-style typography
FONTSIZE_TITLE = 10
FONTSIZE_LABELS = 9
FONTSIZE_TICKS = 8
FIGSIZE = (7.5, 3.2)   # AGU full-page width; taller than the metrics-vs-pm figure so curve shape is legible

_AGU_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
    'font.size': FONTSIZE_TICKS,
    'axes.labelsize': FONTSIZE_LABELS,
    'axes.titlesize': FONTSIZE_TITLE,
    'xtick.labelsize': FONTSIZE_TICKS,
    'ytick.labelsize': FONTSIZE_TICKS,
    'legend.fontsize': FONTSIZE_TICKS,
    'pdf.fonttype': 42,   # embed as TrueType, not Type 3
}

LINESTYLES = ['-', '--', ':', '-.']


# =============================================================================
# %% --- INTERTIDAL HYPSOMETRY ---
# =============================================================================
# classify_intertidal_mask() and compute_intertidal_hypsometry() are shared
# with compute_hydro_metrics_allQ.py via functions/F_hypsometry.py, so the
# hypsometry here corresponds exactly to the intertidal area reported there.


# =============================================================================
# %% --- MAIN COMPUTE LOOP ---
# =============================================================================
print('#' * 60)
print('  COMPUTING: Intertidal-zone hypsometry per scenario')
print('#' * 60)

# hypso_store[discharge][label] = {'elev', 'frac', 'area_km2', 'pm', 'n', 'run_id'}
hypso_store = {discharge: {} for discharge in DISCHARGES}
long_rows = []      # for the full curve CSV
summary_rows = []   # for the per-scenario total-area CSV

for discharge in DISCHARGES:
    dhr_base = BASE_DIR / f'Q{discharge}' / 'detailed-hydro-run'
    scenario_folders = discover_variability_scenario_folders(dhr_base, discharge, RUN_IDS_TO_INCLUDE, _FOLDER_RE)

    if not scenario_folders:
        print(f'[SKIP] No matching folders in: {dhr_base}')
        continue

    print(f"\n{'=' * 60}\nQ = {discharge} m3/s: {len(scenario_folders)} scenario(s)\n{'=' * 60}")

    cache_dir = dhr_base / 'cached_data'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)

    for folder_path, run_id, pm_val, n_val in scenario_folders:
        label = f'Qr{discharge}_dhr{run_id:02d}_pm{pm_val}_n{n_val}'
        print(f'  Processing: {folder_path.name}  ->  {label}')

        run_paths = get_stitched_map_run_paths(
            base_path=dhr_base, folder_name=folder_path.name,
            timed_out_dir=None, variability_map=None, analyze_noisy=False,
        ) or [folder_path]

        ds = load_or_update_map_cache_multi(
            cache_dir=cache_dir, folder_name=folder_path.name, run_paths=run_paths,
            var_names=LOAD_VARS, bbox=CACHE_BBOX,
            append_time=True, append_vars=True, cache_tag=cache_tag,
        )

        if ds is None or 'time' not in ds.dims:
            print(f'    [SKIP] missing data for {label}')
            if ds is not None:
                ds.close()
            continue

        missing_vars = [v for v in LOAD_VARS if v not in ds]
        if missing_vars:
            print(f'    [SKIP] missing variable(s) {missing_vars} for {label}')
            ds.close()
            continue

        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        window_mask = get_last_n_hours_window(time_values, IA_WINDOW_HOURS)
        n_window = int(window_mask.sum())

        if n_window < 2:
            print(f'    [SKIP] only {n_window} timestep(s) in last {IA_WINDOW_HOURS}h window')
            ds.close()
            continue

        actual_h = (time_values[window_mask][-1] - time_values[window_mask][0]) / np.timedelta64(1, 'h')
        if actual_h < 11.0:
            print(f'    [WARNING] window is {actual_h:.1f}h -- shorter than one tidal cycle')

        ds_window = ds.isel(time=np.where(window_mask)[0])

        if IA_FACE_X_VAR in ds.coords:
            face_x = np.asarray(ds.coords[IA_FACE_X_VAR].values)
        elif IA_FACE_X_VAR in ds:
            face_x = np.asarray(ds[IA_FACE_X_VAR].values)
        else:
            face_x, _ = _get_face_coords(ds)

        intertidal_mask = classify_intertidal_mask(
            ds_window, IA_WET_THRESHOLD, IA_X_MIN, IA_X_MAX, face_x=face_x,
        )
        if not np.any(intertidal_mask):
            print(f'    [SKIP] no intertidal faces found for {label}')
            ds.close()
            continue

        ba_da = ds['mesh2d_flowelem_ba']
        ba_vals = ba_da.isel(time=0).values if 'time' in ba_da.dims else ba_da.values

        bedlev_final = ds['mesh2d_mor_bl'].isel(time=-1).values

        elev_sorted, cum_area_frac, total_area_m2 = compute_intertidal_hypsometry(
            bedlev_final, ba_vals, intertidal_mask
        )
        ds.close()

        if elev_sorted.size == 0:
            print(f'    [SKIP] empty hypsometric curve for {label}')
            continue

        total_area_km2 = total_area_m2 / 1e6
        print(f'    Intertidal area = {total_area_km2:.4f} km2  '
              f'({elev_sorted.size} faces, elev range {elev_sorted[0]:.2f} to {elev_sorted[-1]:.2f} m)')

        hypso_store[discharge][label] = {
            'elev': elev_sorted,
            'frac': cum_area_frac,
            'area_km2': total_area_km2,
            'pm': pm_val,
            'n': n_val,
            'run_id': run_id,
        }

        summary_rows.append({
            'discharge': discharge, 'label': label, 'run_id': run_id,
            'pm': pm_val, 'n': n_val, 'intertidal_area_km2': total_area_km2,
        })
        for f, e in zip(cum_area_frac, elev_sorted):
            long_rows.append({
                'discharge': discharge, 'label': label, 'pm': pm_val, 'n': n_val,
                'cum_area_frac': f, 'bed_level_m': e,
            })

# %% --- SAVE CSVs ---
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / 'intertidal_hypsometry_area_summary.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'intertidal_hypsometry_area_summary.csv'}")

long_df = pd.DataFrame(long_rows)
long_df.to_csv(OUTPUT_DIR / 'intertidal_hypsometry_curves_long.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'intertidal_hypsometry_curves_long.csv'}")


# =============================================================================
# %% --- PLOT: 1 x N_DISCHARGES HYPSOMETRY COMPARISON ---
# =============================================================================
discharges_with_data = [d for d in DISCHARGES if hypso_store.get(d)]

if not discharges_with_data:
    print('[SKIP PLOT] No hypsometry data available.')
else:
    # Color mapping for R_peak (pm) values
    all_pm_vals = sorted({
        v['pm'] for d in discharges_with_data for v in hypso_store[d].values() if v['n'] != 0
    })
    
    _n_pm = max(len(all_pm_vals) - 1, 1)
    PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}

    # Linestyle keyed to n_peaks (excluding constant n=0 reference)
    all_n = sorted({
        v['n'] for d in discharges_with_data for v in hypso_store[d].values() if v['n'] != 0
    })
    n_to_ls = {n: LINESTYLES[i % len(LINESTYLES)] for i, n in enumerate(all_n)}

    with mpl.rc_context(_AGU_RC):
        fig, axes = plt.subplots(
            1, len(discharges_with_data),
            figsize=FIGSIZE, sharey=True, constrained_layout=True,
        )
        if len(discharges_with_data) == 1:
            axes = [axes]

        for ax, discharge in zip(axes, discharges_with_data):
            scenarios = hypso_store[discharge]

            for label, v in sorted(scenarios.items(), key=lambda kv: (kv[1]['pm'], kv[1]['n'])):
                if v['n'] == 0:
                    color, ls, lw, zorder = GREY_CONST, '-', 2.0, 10
                    plot_label = "Constant (n=0)"
                else:
                    color = PM_COLOR.get(v['pm'], 'blue')
                    ls = n_to_ls.get(v['n'], '-')
                    lw, zorder = 1.3, 4
                    plot_label = f"pm{v['pm']}, n{v['n']}"

                ax.plot(v['frac'], v['elev'], color=color, linestyle=ls,
                        linewidth=lw, alpha=0.9, zorder=zorder, label=plot_label)

            ax.set_xlim(0, 1)
            ax.set_title(f'Q = {discharge} m\u00b3/s', fontsize=FONTSIZE_TITLE)
            ax.set_xlabel('$A_i / A_{tot}$ [\u2013]', fontsize=FONTSIZE_LABELS)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[0].set_ylabel('Bed elevation [m]', fontsize=FONTSIZE_LABELS)

        # --- LEGEND CREATION ---
        proxy_handles = []
        proxy_labels = []

        # 1. Constant reference
        proxy_handles.append(Line2D([0], [0], color=GREY_CONST, lw=2.0, ls='-'))
        proxy_labels.append('Constant (n=0)')

        # 2. R_peak Colors
        for pm in all_pm_vals:
            proxy_handles.append(Line2D([0], [0], color=PM_COLOR[pm], lw=2.0, ls='-'))
            proxy_labels.append(f'$R_{{peak}}$ = {pm}')

        # 3. n_peaks Linestyles
        for n_val, ls in n_to_ls.items():
            proxy_handles.append(Line2D([0], [0], color='black', lw=1.3, ls=ls))
            # proxy_labels.append(f'n = {n_val}')

        # Combined Legend below panels
        fig.legend(
            proxy_handles, proxy_labels, 
            loc='lower center',
            ncol=min(len(proxy_labels), 6), 
            fontsize=FONTSIZE_TICKS,
            bbox_to_anchor=(0.5, -0.12), 
            frameon=False
        )

        fname = 'intertidal_hypsometry_comparison_allQ'
        fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'\nSaved: {OUTPUT_DIR / f"{fname}.png"}')

print(f'\nAll outputs written to: {OUTPUT_DIR.resolve()}')

# %%
# =============================================================================
# %% --- GRANDJEAN ET AL. (2024) STYLE SCATTER PLOT (CUSTOM STYLES) ---
# =============================================================================

# 1. Process data: compute changes relative to Constant (n=0) baseline
scatter_rows = []

for discharge, scenarios in hypso_store.items():
    const_scen = next((v for v in scenarios.values() if v['n'] == 0), None)
    if not const_scen:
        continue

    const_area = const_scen['area_km2']
    idx_const_95 = np.argmin(np.abs(const_scen['frac'] - 0.95))
    const_elev_95 = const_scen['elev'][idx_const_95]

    for label, v in scenarios.items():
        if v['n'] == 0:
            continue  # Skip constant run as reference baseline (0,0)

        delta_area_km2 = v['area_km2'] - const_area
        pct_area_change = (delta_area_km2 / const_area) * 100

        idx_v_95 = np.argmin(np.abs(v['frac'] - 0.95))
        delta_elev_m = v['elev'][idx_v_95] - const_elev_95

        scatter_rows.append({
            'discharge': discharge,
            'label': label,
            'pm': v['pm'],
            'n': v['n'],
            'delta_area_km2': delta_area_km2,
            'pct_area_change': abs(pct_area_change),
            'delta_elev_m': delta_elev_m,
        })

df_scatter = pd.DataFrame(scatter_rows)

if not df_scatter.empty:
    # Marker shapes per discharge
    MARKERS_Q = {250: 'o', 500: 's', 1000: '^'}
    
    # Bubble size scaling factor
    SIZE_SCALE = 12.0
    df_scatter['marker_size'] = np.maximum(df_scatter['pct_area_change'], 5) * SIZE_SCALE

    with mpl.rc_context(_AGU_RC):
        fig, ax = plt.subplots(figsize=(6.5, 7.0), dpi=300)
        
        # --- YOUR CUSTOM STYLING ---
        ax.set_facecolor('#ffffff')  # White background
        ax.grid(True, linestyle='--', color='#cccccc', alpha=0.7, zorder=0)  # Dashed grid lines
        
        # Zero-reference axes
        ax.axhline(0, color='#6c6c6c', linewidth=1.5, zorder=1)
        ax.axvline(0, color='#6c6c6c', linewidth=1.5, zorder=1)

        # Plot data points grouped by Discharge (Q) & colored by R_peak (pm)
        for q_val, group in df_scatter.groupby('discharge'):
            ax.scatter(
                group['delta_area_km2'],
                group['delta_elev_m'],
                s=group['marker_size'],
                c=[PM_COLOR.get(pm, 'blue') for pm in group['pm']],  # Color by R_peak (pm)
                marker=MARKERS_Q.get(q_val, 'o'),
                edgecolor='black',
                linewidth=0.8,
                alpha=0.85,
                zorder=2
            )

        # Text Annotations
        for _, row in df_scatter.iterrows():
            ax.annotate(
                f"pm{row['pm']} (Q{row['discharge']})",
                (row['delta_area_km2'], row['delta_elev_m']),
                xytext=(6, 4), textcoords='offset points',
                fontsize=7, color='#333333'
            )

        # Labels & Spines
        ax.set_xlabel('Lateral change in intertidal area [km$^2$]', fontsize=FONTSIZE_LABELS, labelpad=8)
        ax.set_ylabel('Upper flat elevation change [m relative to constant]', fontsize=FONTSIZE_LABELS, labelpad=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- LEGEND CREATION ---
        legend_handles = []
        legend_labels = []

        # 1. R_peak Colors
        for pm in sorted(PM_COLOR.keys()):
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=PM_COLOR[pm], markeredgecolor='black', markersize=9))
            legend_labels.append(f'$R_{{peak}}$ = {pm}')

        # 2. Discharge Shapes
        for q_val, marker in MARKERS_Q.items():
            legend_handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='grey', markeredgecolor='black', markersize=9))
            legend_labels.append(f'Q = {q_val} m$^3$/s')

        # Combined Legend below plot
        ax.legend(
            legend_handles, legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
            fontsize=FONTSIZE_TICKS,
            columnspacing=1.5
        )

        fname_grandjean = 'intertidal_lateral_vs_vertical_changes_custom'
        fig.savefig(OUTPUT_DIR / f'{fname_grandjean}.png', dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / f'{fname_grandjean}.pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'Saved: {OUTPUT_DIR / f"{fname_grandjean}.png"}')

# %%
# %% --- DIFFERENCE-FROM-CONSTANT HYPSOMETRY: use the whole curve, not one percentile ---
# Interpolate every scenario onto a common Ai/Atot grid and subtract the
# constant (n=0) curve pointwise. Shows exactly where along the curve the
# R_peak effect concentrates, instead of assuming it's at the top (p95).

FRAC_GRID = np.linspace(0.0, 1.0, 201)

diff_rows = []
for discharge, scenarios in hypso_store.items():
    const_scen = next((v for v in scenarios.values() if v['n'] == 0), None)
    if not const_scen:
        continue

    const_elev_grid = np.interp(FRAC_GRID, const_scen['frac'], const_scen['elev'])

    for label, v in scenarios.items():
        if v['n'] == 0:
            continue
        scen_elev_grid = np.interp(FRAC_GRID, v['frac'], v['elev'])
        delta_elev_grid = scen_elev_grid - const_elev_grid

        for frac, delta in zip(FRAC_GRID, delta_elev_grid):
            diff_rows.append({
                'discharge': discharge, 'label': label, 'pm': v['pm'], 'n': v['n'],
                'frac': frac, 'delta_elev_m': delta,
            })

df_diff = pd.DataFrame(diff_rows)
df_diff.to_csv(OUTPUT_DIR / 'intertidal_hypsometry_diff_from_constant.csv', index=False)

# %% --- PLOT: delta elevation vs Ai/Atot, one panel per Q, coloured by R_peak ---
discharges_with_data = [d for d in DISCHARGES if hypso_store.get(d)]
cmap = plt.cm.Blues
norm = mpl.colors.Normalize(vmin=min(all_pm_vals), vmax=max(all_pm_vals))
with mpl.rc_context(_AGU_RC):
    fig, axes = plt.subplots(1, len(discharges_with_data), figsize=(7.5, 3.0),
                              sharey=True, constrained_layout=True)
    if len(discharges_with_data) == 1:
        axes = [axes]

    for ax, discharge in zip(axes, discharges_with_data):
        sub = df_diff[df_diff['discharge'] == discharge]
        for (pm_val, n_val), grp in sub.groupby(['pm', 'n']):
            grp = grp.sort_values('frac')
            ax.plot(grp['frac'], grp['delta_elev_m'],
                    color=cmap(norm(pm_val)), linewidth=1.3, alpha=0.9)

        ax.axhline(0, color='black', linewidth=1.2, zorder=1)  # constant = 0 by construction
        ax.set_xlim(0, 1)
        ax.set_title(f'Q = {discharge} m\u00b3/s', fontsize=FONTSIZE_TITLE)
        ax.set_xlabel('$A_i / A_{tot}$ [\u2013]', fontsize=FONTSIZE_LABELS)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('$\\Delta$ bed elevation vs. constant [m]', fontsize=FONTSIZE_LABELS)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label('$R_{peak}$ [m\u00b3/s]', fontsize=FONTSIZE_LABELS)

    fname = 'intertidal_hypsometry_diff_from_constant_allQ'
    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight')
    plt.show()
# %%
