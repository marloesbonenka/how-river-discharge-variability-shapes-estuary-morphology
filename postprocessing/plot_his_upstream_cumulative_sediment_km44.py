"""
Compute cumulative sediment signal through the upstream cross section (km44)
for each variability scenario.

Outputs:
- One comparison plot with all scenarios overlaid
"""
#%%
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean
from scipy import stats

sys.path.append(r"C:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing\FUNCTIONS")

from FUNCTIONS.F_loaddata import load_and_cache_scenario
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
#%%

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170          # AGU full-page width
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM * MM_TO_IN

# --- AGU figure styling (Calibri font, AGU-compliant line weights/export dpi) ---
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

# =========================
# Configuration
# =========================
SED_VAR = "cross_section_bedload_sediment_transport"
RIVER_KM = 45

SCENARIOS_TO_PROCESS = None  # None = all; e.g. ['1', '2', '3', '4', '5'] for a subset
DISCHARGE = 500
ANALYZE_NOISY = False

# Toggle: set True to produce a matrix plot (columns = n_peaks, rows = peak_ratio)
#         set False to produce the standard overlay comparison plot
PLOT_AS_MATRIX = True

# Toggle heatmap and scatter outputs independently
PLOT_HEATMAPS = True   # set False to skip heatmap panels
PLOT_SCATTER  = True   # set False to skip scatter-correlation plot

SCENARIO_LABELS = None

# colorblind friendly
SCENARIO_COLORS = None
BASE_DIRECTORY = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
CONFIG = f"Model_Output/Q{DISCHARGE}"

if ANALYZE_NOISY:
    BASE_PATH = BASE_DIRECTORY / CONFIG / f"0_Noise_Q{DISCHARGE}"
else:
    BASE_PATH = BASE_DIRECTORY / CONFIG

OUTPUT_DIR = BASE_PATH / "output_plots" / "plots_his_sedimentsupply_km44"
CACHE_DIR = BASE_PATH / "cached_data"
TIMED_OUT_DIR = BASE_PATH / "timed-out"


#%%

if not BASE_PATH.exists():
    raise FileNotFoundError(f"Base path not found: {BASE_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if TIMED_OUT_DIR is None or not TIMED_OUT_DIR.exists():
    TIMED_OUT_DIR = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')

variability_map = get_variability_map(DISCHARGE)

model_folders = find_variability_model_folders(
    base_path=BASE_PATH,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=ANALYZE_NOISY,
)

print(f"Found {len(model_folders)} run folders in {BASE_PATH}")

# Build HIS file path map with the same stitching logic as other scripts.
run_his_paths = {}
for folder in model_folders:
    model_location = BASE_PATH / folder
    his_paths = []
    scenario_num = folder.name.split('_')[0]
    try:
        scenario_key = str(int(scenario_num))
    except Exception:
        scenario_key = scenario_num

    if ANALYZE_NOISY:
        match = re.search(r'noisy(\d+)', folder.name)
        timed_out_folder = None
        if TIMED_OUT_DIR is None:
            print('[WARNING] Timed-out directory not available; skipping timed-out noisy runs.')
        elif match:
            noisy_id = match.group(0)
            for f in TIMED_OUT_DIR.iterdir():
                if f.is_dir() and noisy_id in f.name:
                    timed_out_folder = f.name
                    break
        if timed_out_folder:
            timed_out_path = TIMED_OUT_DIR / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)
    else:
        if TIMED_OUT_DIR is not None:
            timed_out_folder = variability_map.get(scenario_key, folder.name)
            timed_out_path = TIMED_OUT_DIR / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)

    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

if CACHE_DIR.exists():
    if not CACHE_DIR.is_dir():
        raise RuntimeError(f"[ERROR] {CACHE_DIR} exists but is not a directory.")
    try:
        _ = list(CACHE_DIR.iterdir())
    except Exception as e:
        raise RuntimeError(f"[ERROR] {CACHE_DIR} is not accessible: {e}")
else:
    CACHE_DIR.mkdir(exist_ok=True)

comparison_series = []

for folder, his_paths in run_his_paths.items():

    parts = folder.name.split("_")
    scenario_num = str(int(parts[0]))
    run_id = "_".join(parts[1:]) if len(parts) > 1 else folder.name
    cache_file = CACHE_DIR / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

    _, data = load_and_cache_scenario(
        scenario_dir=folder,
        his_file_paths=his_paths,
        cache_file=cache_file,
        boxes=[],
        var_name=SED_VAR,
    )

    km_positions = np.asarray(data["km_positions"])
    idx_upstream = int(np.argmin(np.abs(km_positions - RIVER_KM)))
    km_actual = float(km_positions[idx_upstream])

    time = pd.to_datetime(np.asarray(data["t"]))
    sediment_transport = np.asarray(data[SED_VAR])[:, idx_upstream]

    scenario_label = SCENARIO_LABELS.get(scenario_num, folder.name) if SCENARIO_LABELS else folder.name

    comparison_series.append(
        {
            "scenario_number": scenario_num,
            "scenario_label": scenario_label,
            "run_folder": folder.name,
            "km_actual": km_actual,
            "time": time,
            "sediment_transport": sediment_transport,
            "final_value": float(sediment_transport[-1]),
        }
    )

    print(
        f"{folder.name}: km target={RIVER_KM}, km actual={km_actual:.2f}, "
        f"final value={sediment_transport[-1]:.3e}"
    )

if comparison_series:
    comparison_series.sort(key=lambda d: (int(d["scenario_number"]), d["run_folder"]))

    # -------------------------------------------------------------------------
    # Regex to extract peak_ratio / n_peaks from Gaussian folder names
    # e.g.  "05_Qr500_pm3_n4"  →  peak_ratio=3.0, n_peaks=4
    # -------------------------------------------------------------------------
    _FOLDER_RE = re.compile(r"_pm(\d+(?:\.\d+)?)_n(\d+)", re.IGNORECASE)

    # Attach peak_ratio / n_peaks to all series upfront (needed for CSV too)
    for s in comparison_series:
        m = _FOLDER_RE.search(s["run_folder"])
        s["peak_ratio"] = float(m.group(1)) if m else None
        s["n_peaks"]    = int(m.group(2))   if m else None

    # -------------------------------------------------------------------------
    # Export end-values to CSV
    # -------------------------------------------------------------------------
    csv_rows = [
        {
            "scenario_number": s["scenario_number"],
            "run_folder":      s["run_folder"],
            "peak_ratio":      s["peak_ratio"],
            "n_peaks":         s["n_peaks"],
            "km_actual":       s["km_actual"],
            "final_value_kg":  s["final_value"],
        }
        for s in comparison_series
    ]
    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUTPUT_DIR / f"endvalues_km{RIVER_KM}_bedload_Q{DISCHARGE}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved end-values CSV: {csv_path}")

    if PLOT_AS_MATRIX:
        parseable = [s for s in comparison_series if s["peak_ratio"] is not None]
        if not parseable:
            print("[WARNING] No _pm<r>_n<n> folder names found; "
                  "falling back to overlay plot.")
            PLOT_AS_MATRIX = False

    if PLOT_AS_MATRIX:
        # Build axis values
        all_n_peaks     = sorted({s["n_peaks"]    for s in parseable})
        all_peak_ratios = sorted({s["peak_ratio"] for s in parseable})
        n_cols = len(all_n_peaks)
        n_rows = len(all_peak_ratios)
        row_order = list(reversed(all_peak_ratios))  # highest ratio at top

        # Global y-limits
        all_vals = np.concatenate([s["sediment_transport"] for s in parseable])
        g_ymin, g_ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        ypad = max((g_ymax - g_ymin) * 0.08, 1.0)
        global_ylim = (g_ymin - ypad, g_ymax + ypad)

        # Consistent color map: one fixed color per peak_ratio value
        # (same palette as plot_scenario_lines.py)
        _PALETTE = [
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
        RATIO_COLOR: dict[float, str] = {
            ratio: _PALETTE[i % len(_PALETTE)]
            for i, ratio in enumerate(all_peak_ratios)
        }

        PANEL_W, PANEL_H = FIGURE_WIDTH_IN / n_cols, 2.4
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(PANEL_W * n_cols, PANEL_H * n_rows),
            sharex=False, sharey=True,
        )
        axes = np.atleast_2d(axes)

        for ri, peak_ratio in enumerate(row_order):
            for ci, n_peaks in enumerate(all_n_peaks):
                ax = axes[ri, ci]
                matches = [
                    s for s in parseable
                    if s["peak_ratio"] == peak_ratio and s["n_peaks"] == n_peaks
                ]
                if not matches:
                    ax.set_visible(False)
                    continue

                for s in matches:
                    ax.plot(s["time"], s["sediment_transport"],
                            color=RATIO_COLOR[peak_ratio], linewidth=0.9)

                ax.set_ylim(global_ylim)
                ax.grid(True, alpha=0.22, linewidth=0.5)

                # Scenario number label (top-left)
                ax.text(
                    0.03, 0.93,
                    f"S{matches[0]['scenario_number']}",
                    transform=ax.transAxes, fontsize=7,
                    ha="left", va="top", color="0.45",
                )

                # x-axis: month labels on bottom row only
                if ri == n_rows - 1:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.tick_params(axis="x", labelsize=7, rotation=40)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", length=0)

                # y-axis: tick labels on left column only
                if ci == 0:
                    ax.tick_params(axis="y", labelsize=7)
                    ax.set_ylabel("cum. sed. transport [kg]", fontsize=7, labelpad=3)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", length=0)

        # Column headers (n_peaks)
        for ci, n_peaks in enumerate(all_n_peaks):
            for ri in range(n_rows):
                if axes[ri, ci].get_visible():
                    axes[ri, ci].set_title(
                        f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}",
                        fontsize=9, fontweight="bold", pad=5,
                    )
                    break

        # Row labels (peak_ratio)
        for ri, peak_ratio in enumerate(row_order):
            pr_str = (
                f"{int(peak_ratio)}"
                if peak_ratio == int(peak_ratio)
                else f"{peak_ratio}"
            )
            for ci in range(n_cols - 1, -1, -1):
                if axes[ri, ci].get_visible():
                    axes[ri, ci].yaxis.set_label_position("right")
                    axes[ri, ci].set_ylabel(
                        f"$R_{{\\mathrm{{peak}}}}$ = {pr_str}",
                        rotation=270, labelpad=15,
                        fontsize=9, fontweight="bold",
                    )
                    break

        fig.text(0.5, 0.005, "peak frequency",
                 ha="center", fontsize=10, style="italic", color="0.35")
        fig.text(0.005, 0.5, "peak amplitude",
                 va="center", fontsize=10, style="italic", color="0.35",
                 rotation="vertical")
        fig.suptitle(
            f"Cumulative sediment transport at km {RIVER_KM}  "
            f"($Q_{{\\mathrm{{mean}}}}$ = {DISCHARGE} m³/s)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout(rect=[0.03, 0.03, 1.0, 1.0])

        fig_path = OUTPUT_DIR / (
            f"matrix_upstream_km{RIVER_KM}_bedload_transport_Q{DISCHARGE}.png"
        )
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved matrix plot: {fig_path}")

        # ── Lines plot: one panel per n_peaks, amplitudes as colors ──────────
        import matplotlib.lines as mlines

        n_panels  = len(all_n_peaks)
        LPANEL_W, LPANEL_H = FIGURE_WIDTH_IN / n_panels, 3.0
        fig_l, axes_l = plt.subplots(
            1, n_panels,
            figsize=(LPANEL_W * n_panels, LPANEL_H),
            sharey=True, sharex=False,
            layout="constrained",
        )
        if n_panels == 1:
            axes_l = [axes_l]

        for ci, n_peaks in enumerate(all_n_peaks):
            ax_l = axes_l[ci]
            for peak_ratio in all_peak_ratios:
                group = [
                    s for s in parseable
                    if s["peak_ratio"] == peak_ratio and s["n_peaks"] == n_peaks
                ]
                pr_str = (
                    f"{int(peak_ratio)}"
                    if peak_ratio == int(peak_ratio)
                    else f"{peak_ratio}"
                )
                for s in group:
                    ax_l.plot(
                        s["time"], s["sediment_transport"],
                        color=RATIO_COLOR[peak_ratio],
                        linewidth=1.2,
                        label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_str}",
                    )

            ax_l.set_ylim(global_ylim)
            ax_l.grid(True, alpha=0.22, linewidth=0.5)
            ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax_l.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_l.tick_params(axis="x", labelsize=8, rotation=40)
            ax_l.set_title(
                f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}",
                fontsize=10, fontweight="bold", pad=5,
            )
            if ci == 0:
                ax_l.set_ylabel("cum. sed. transport [kg]", fontsize=9, labelpad=4)
                ax_l.tick_params(axis="y", labelsize=8)

        # Shared legend
        legend_handles = []
        for peak_ratio in all_peak_ratios:
            pr_str = (
                f"{int(peak_ratio)}"
                if peak_ratio == int(peak_ratio)
                else f"{peak_ratio}"
            )
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color=RATIO_COLOR[peak_ratio],
                    linewidth=1.8,
                    label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_str}",
                )
            )
        fig_l.legend(
            handles=legend_handles,
            title="peak amplitude",
            title_fontsize=9,
            fontsize=8,
            loc="lower center",
            ncol=len(all_peak_ratios),
            bbox_to_anchor=(0.5, -0.18),
            frameon=True,
        )
        fig_l.suptitle(
            f"Cumulative sediment transport at km {RIVER_KM}  "
            f"($Q_{{\\mathrm{{mean}}}}$ = {DISCHARGE} m³/s)",
            fontsize=12, fontweight="bold", y=1.02,
        )
        fig_l_path = OUTPUT_DIR / (
            f"lines_upstream_km{RIVER_KM}_bedload_transport_Q{DISCHARGE}.png"
        )
        fig_l.savefig(fig_l_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved lines plot: {fig_l_path}")

    else:
        # ── Standard overlay comparison plot ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 5))
        for series in comparison_series:
            color = SCENARIO_COLORS.get(series["scenario_number"], None) \
                if SCENARIO_COLORS else None
            label = f"{series['scenario_label']}"
            ax.plot(series["time"], series["sediment_transport"],
                    lw=1.8, color=color, label=label)

        ax.set_title(f"km {RIVER_KM} | $Q_{{mean}}$ = {DISCHARGE} m³/s")
        ax.set_xlabel("Time")
        ax.set_ylabel("cumulative sediment transport [kg]")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        fig_path = OUTPUT_DIR / (
            f"comparison_upstream_km{RIVER_KM}_bedload_transport_Q{DISCHARGE}.png"
        )
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved comparison plot: {fig_path}")
else:
    print("No scenarios processed; comparison plot not written.")

# %%
# =============================================================================
# Heatmap: final cumulative sediment transport in peak_ratio × n_peaks space
# =============================================================================
_hm_data = [s for s in comparison_series if s.get("peak_ratio") is not None]

if _hm_data and PLOT_HEATMAPS:
    _hm_n_peaks     = sorted({s["n_peaks"]    for s in _hm_data})
    _hm_peak_ratios = sorted({s["peak_ratio"] for s in _hm_data}, reverse=True)

    # Build 2-D array: rows = peak_ratio (descending), cols = n_peaks
    _grid = np.full((len(_hm_peak_ratios), len(_hm_n_peaks)), np.nan)
    for s in _hm_data:
        ri = _hm_peak_ratios.index(s["peak_ratio"])
        ci = _hm_n_peaks.index(s["n_peaks"])
        _grid[ri, ci] = s["final_value"]

    fig_hm, ax_hm = plt.subplots(figsize=(FIGURE_WIDTH_IN,
                                           max(3, len(_hm_peak_ratios) * 0.9)))

    im = ax_hm.imshow(_grid, aspect="auto", cmap=cmocean.cm.turbid,
                      vmin=np.nanmin(_grid), vmax=np.nanmax(_grid))

    # Annotate each cell with the value
    for ri in range(len(_hm_peak_ratios)):
        for ci in range(len(_hm_n_peaks)):
            val = _grid[ri, ci]
            if not np.isnan(val):
                ax_hm.text(ci, ri, f"{val:.2e}", ha="center", va="center",
                           fontsize=7.5,
                           color="black" if val < 0.6 * np.nanmax(_grid) else "white")

    ax_hm.set_xticks(range(len(_hm_n_peaks)))
    ax_hm.set_xticklabels([str(n) for n in _hm_n_peaks])
    ax_hm.set_yticks(range(len(_hm_peak_ratios)))
    ax_hm.set_yticklabels([
        f"{int(r)}" if r == int(r) else f"{r}" for r in _hm_peak_ratios
    ])
    #fr'$\text{{p95 (bar) R}}_{{peak}}
    ax_hm.set_xlabel(r'$n_{\text{peaks}}$')
    ax_hm.set_ylabel(r'$R_{\text{peak}}$')
    ax_hm.set_title(
        f"Final cumulative sediment transport at km {RIVER_KM}  "
        f"($Q_\\mathrm{{mean}}$ = {DISCHARGE} m³/s)",
        fontweight="bold",
    )

    cbar = fig_hm.colorbar(im, ax=ax_hm, pad=0.02)
    cbar.set_label("cumulative sediment transport [kg]", fontsize=9)

    fig_hm.tight_layout()

    fig_hm_path = OUTPUT_DIR / f"heatmap_endvalue_km{RIVER_KM}_bedload_Q{DISCHARGE}"
    fig_hm.savefig(fig_hm_path.with_suffix('.png'), dpi=300, bbox_inches="tight")
    fig_hm.savefig(fig_hm_path.with_suffix('.pdf'), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved heatmap: {fig_hm_path}")

    # -------------------------------------------------------------------------
    # Normalised heatmap: % change relative to the constant scenario
    # (peak_ratio=1, n_peaks=0)
    # -------------------------------------------------------------------------
    _ref_matches = [
        s for s in _hm_data if s["peak_ratio"] == 1.0 and s["n_peaks"] == 0
    ]
    if not _ref_matches:
        print("[WARNING] Constant scenario (pm1_n0) not found; skipping normalised heatmap.")
    else:
        _ref_val = _ref_matches[0]["final_value"]

        # % change: (scenario - constant) / |constant| * 100
        _grid_pct = np.full_like(_grid, np.nan)
        if _ref_val != 0:
            _grid_pct = (_grid - _ref_val) / abs(_ref_val) * 100.0

        # All values are positive (sediment supply only increases relative to
        # the constant scenario), so use a sequential colormap from 0 rather
        # than a diverging red-blue scale.
        _pct_max = np.nanmax(_grid_pct)

        fig_nm, ax_nm = plt.subplots(figsize=(FIGURE_WIDTH_IN,
                                               max(3, len(_hm_peak_ratios) * 0.9)))

        im_nm = ax_nm.imshow(_grid_pct, aspect="auto", cmap=cmocean.cm.turbid,
                             vmin=0, vmax=_pct_max)

        for ri in range(len(_hm_peak_ratios)):
            for ci in range(len(_hm_n_peaks)):
                val = _grid_pct[ri, ci]
                if not np.isnan(val):
                    ax_nm.text(ci, ri, f"{val:+.0f}%", ha="center", va="center",
                               fontsize=7.5,
                               color="black" if val < 0.6 * _pct_max else "white")

        ax_nm.set_xticks(range(len(_hm_n_peaks)))
        ax_nm.set_xticklabels([str(n) for n in _hm_n_peaks])
        ax_nm.set_yticks(range(len(_hm_peak_ratios)))
        ax_nm.set_yticklabels([
            f"{int(r)}" if r == int(r) else f"{r}" for r in _hm_peak_ratios
        ])
        ax_nm.set_xlabel(r'$n_{\text{peaks}}$')
        ax_nm.set_ylabel(r'$R_{\text{peak}}$')
        ax_nm.set_title(
            f"Change in final cumulative sediment transport vs. constant  (km {RIVER_KM}, "
            f"$Q_\\mathrm{{mean}}$ = {DISCHARGE} m³/s)",
            fontweight="bold",
        )

        cbar_nm = fig_nm.colorbar(im_nm, ax=ax_nm, pad=0.02)
        cbar_nm.set_label("change relative to constant scenario [%]", fontsize=9)

        fig_nm.tight_layout()

        fig_nm_path = OUTPUT_DIR / f"heatmap_normalised_km{RIVER_KM}_bedload_Q{DISCHARGE}"
        fig_nm.savefig(fig_nm_path.with_suffix('.png'), dpi=300, bbox_inches="tight")
        fig_nm.savefig(fig_nm_path.with_suffix('.pdf'), dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved normalised heatmap: {fig_nm_path}")

else:
    if not _hm_data:
        print("No parseable scenarios for heatmap.")
    else:
        print("Skipping heatmaps (PLOT_HEATMAPS = False).")

# %%
# =============================================================================
# Two-panel scatter plot:
#   Left  – n_peaks (x) vs. sediment transport (y), coloured by R_peak
#   Right – R_peak  (x) vs. sediment transport (y), coloured by n_peaks
# =============================================================================
if PLOT_SCATTER and _hm_data:
    _sc_series = [s for s in _hm_data if s["n_peaks"] is not None]

    _x_freq = np.array([s["n_peaks"]    for s in _sc_series], dtype=float)
    _x_amp  = np.array([s["peak_ratio"] for s in _sc_series], dtype=float)
    _y_sed  = np.array([s["final_value"] for s in _sc_series], dtype=float)

    # ── helper: linear regression stats ──────────────────────────────────────
    def _regress(x, y):
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 200)
        return slope, intercept, r, r**2, p, x_fit

    sl_f, ic_f, r_f, r2_f, p_f, xf_freq = _regress(_x_freq, _y_sed)
    sl_a, ic_a, r_a, r2_a, p_a, xf_amp  = _regress(_x_amp,  _y_sed)

    # ── shared y-axis limits ──────────────────────────────────────────────────
    _y_pad  = (_y_sed.max() - _y_sed.min()) * 0.06
    _y_lim  = (_y_sed.min() - _y_pad, _y_sed.max() + _y_pad)

    # ── colourmap normalisation ───────────────────────────────────────────────
    import matplotlib.colors as mcolors
    _norm_amp  = mcolors.Normalize(vmin=_x_amp.min(),  vmax=_x_amp.max())
    _norm_freq = mcolors.Normalize(vmin=_x_freq.min(), vmax=_x_freq.max())

    fig_sc, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(FIGURE_WIDTH_IN, 5),
        sharey=True,
        layout="constrained",
    )

    # ── Left panel: frequency (n_peaks) ──────────────────────────────────────
    sc_l = ax_l.scatter(
        _x_freq, _y_sed,
        c=_x_amp,
        cmap="plasma",
        norm=_norm_amp,
        s=70,
        edgecolors="0.25",
        linewidths=0.5,
        zorder=3,
    )
    ax_l.plot(
        xf_freq, sl_f * xf_freq + ic_f,
        color="0.2", linewidth=1.5, linestyle="--", zorder=2,
        label=(
            f"$r$ = {r_f:.3f}\n"
            f"$R^2$ = {r2_f:.3f}\n"
            f"$p$ = {p_f:.3g}"
        ),
    )
    cbar_l = fig_sc.colorbar(sc_l, ax=ax_l, pad=0.03, aspect=30)
    cbar_l.set_label("$R_\\mathrm{peak}$")
    ax_l.set_xlabel("$n_\\mathrm{peaks}$")
    ax_l.set_ylabel(
        f"Final cumulative sediment transport at km {RIVER_KM}  [kg]", fontsize=11
    )
    ax_l.set_title("Sediment supply vs. Peak Frequency", fontsize=12, fontweight="bold")
    ax_l.set_ylim(_y_lim)
    ax_l.legend(fontsize=9, loc="upper left", framealpha=0.85, title="Linear fit")
    ax_l.grid(True, alpha=0.3, linewidth=0.5)

    # ── Right panel: amplitude (R_peak) ──────────────────────────────────────
    sc_r = ax_r.scatter(
        _x_amp, _y_sed,
        c=_x_freq,
        cmap="viridis",
        norm=_norm_freq,
        s=70,
        edgecolors="0.25",
        linewidths=0.5,
        zorder=3,
    )
    ax_r.plot(
        xf_amp, sl_a * xf_amp + ic_a,
        color="0.2", linewidth=1.5, linestyle="--", zorder=2,
        label=(
            f"$r$ = {r_a:.3f}\n"
            f"$R^2$ = {r2_a:.3f}\n"
            f"$p$ = {p_a:.3g}"
        ),
    )
    cbar_r = fig_sc.colorbar(sc_r, ax=ax_r, pad=0.03, aspect=30)
    cbar_r.set_label("$n_\\mathrm{peaks}$")
    ax_r.set_xlabel("$R_\\mathrm{peak}$")
    ax_r.set_title("Sediment supply vs. Peak Amplitude", fontweight="bold")
    ax_r.set_ylim(_y_lim)
    ax_r.legend(loc="upper left", framealpha=0.85, title="Linear fit")
    ax_r.grid(True, alpha=0.3, linewidth=0.5)

    fig_sc.suptitle(
        f"Sediment supply vs. discharge variability  "
        f"($Q_\\mathrm{{mean}}$ = {DISCHARGE} m³/s)",
        fontweight="bold", y=1.02,
    )

    fig_sc_path = OUTPUT_DIR / f"scatter_2panel_km{RIVER_KM}_bedload_Q{DISCHARGE}"
    fig_sc.savefig(fig_sc_path.with_suffix('.png'), bbox_inches="tight")
    plt.show()
    print(f"Saved two-panel scatter plot: {fig_sc_path}")
elif PLOT_SCATTER:
    print("[INFO] No parseable scenarios for scatter plot.")
# %%