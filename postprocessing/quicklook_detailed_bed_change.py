# %% Quick-look figures: measured bed change in the detailed-hydro runs (no polish)
"""
Basic, direct views of the measured bed-level change (mesh2d_mor_bl, last minus
first output) of the 2-day detailed-hydro runs. No transport reconstruction and no
channel footprint: every figure is the bed level the model computed.

Reads the per-run caches written by fig_peak_mean_sediment_routing.py
(cached_data/routing_bed_wl_<run>.npz); run that script once first.

Figures (in OUTPUT_DIR):
1_bed_change_along_estuary   bed volume change per km of estuary [m3/day per km],
                             summed over the full width; one line per run
2_bed_change_vs_bed_level    mean bed change [mm/day] per initial bed-level bin,
                             lower (20-25 km) and upper (25-45 km) estuary; low bed
                             levels are channels, high ones margins/bars
3_cross_sections             initial bed + low-water level, and bed change across
                             the estuary at a few x (averaged over +-X_HALF_WINDOW_M)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# %% --- SETTINGS --------------------------------------------------------------
BASE_PATH = Path(r"U:\PhDNaturalRhythmEstuaries\Models"
                 r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500")
CACHE_DIR = BASE_PATH / "cached_data"
OUTPUT_DIR = BASE_PATH / "detailed-hydro-run" / "output_plots" / "sediment_budgets" / "quicklook"

RUNS = {
    "constant": "dhr_01_Qr500_pm1_n0.9724783",
    "pm3_n5_mean_flow": "dhr_03_Qr500_pm3_n5_mean.964903",
    "pm3_n5_peak_flow": "dhr_03_Qr500_pm3_n5_peak.964907",
    "pm5_n1_mean_flow": "dhr_05_Qr500_pm5_n1_mean.964908",
    "pm5_n1_peak_flow": "dhr_05_Qr500_pm5_n1_peak.965221",
}
COLORS = {
    "constant": "#7f7f7f",
    "pm3_n5_mean_flow": plt.cm.Blues(0.55), "pm5_n1_mean_flow": plt.cm.Blues(0.92),
    "pm3_n5_peak_flow": plt.cm.Oranges(0.55), "pm5_n1_peak_flow": plt.cm.Oranges(0.92),
}

Y_LIMITS_M = (5_000.0, 10_000.0)
X_LIMITS_KM = (20.0, 45.0)
ZONE_EDGES_KM = np.arange(20.0, 46.0, 5.0)
REACHES_KM = {"lower estuary 20-25 km": (20.0, 25.0), "upper estuary 25-45 km": (25.0, 45.0)}
BED_BIN_M = 0.5
MIN_CELLS_PER_BIN = 30
SECTIONS_KM = (22.5, 32.5, 37.5)
X_HALF_WINDOW_M = 250.0     # cross-section = mean over x +- this (damps cell-scale noise)


# %% --- DATA ------------------------------------------------------------------
def load(folder):
    """Grid the cached faces: initial bed, bed change [m/day], low-water level."""
    cache = CACHE_DIR / f"routing_bed_wl_{folder.replace('.', '_')}.npz"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing: run fig_peak_mean_sediment_routing.py first.")
    run = np.load(cache)
    ux, ix = np.unique(np.round(run["x"], 1), return_inverse=True)
    uy, iy = np.unique(np.round(run["y"], 1), return_inverse=True)

    def grid(values):
        g = np.full((uy.size, ux.size), np.nan)
        g[iy, ix] = values
        return g

    days = run["hours"][-1] / 24.0
    return {"x_km": ux / 1000.0, "y_km": uy / 1000.0,
            "area": np.gradient(uy)[:, None] * np.gradient(ux)[None, :],
            "bed0": grid(run["bl"][0]), "dz_day": grid((run["bl"][-1] - run["bl"][0]) / days),
            "wl_min": grid(run["wl_min"])}


data = {name: load(folder) for name, folder in RUNS.items()}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def strip_mask(d):
    return ((d["y_km"] >= Y_LIMITS_M[0] / 1000) & (d["y_km"] <= Y_LIMITS_M[1] / 1000))[:, None]


# %% --- 1: bed volume change along the estuary --------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
for name, d in data.items():
    vol = np.nan_to_num(d["dz_day"]) * d["area"] * strip_mask(d)      # m3/day per cell
    km = np.floor(d["x_km"])
    bins = np.arange(X_LIMITS_KM[0], X_LIMITS_KM[1])
    per_km = [vol[:, km == b].sum() for b in bins]
    ax.plot(bins + 0.5, per_km, marker="o", ms=3, color=COLORS[name], label=name)
ax.axhline(0, color="k", lw=0.8)
for e in ZONE_EDGES_KM:
    ax.axvline(e, color="0.8", lw=0.6, zorder=0)
ax.set_xlabel("x [km] (sea left, river right)")
ax.set_ylabel("Bed volume change [m$^3$ day$^{-1}$ per km]\n(+ = deposition, full width)")
ax.set_title("1  Where does the bed gain or lose sediment along the estuary?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "1_bed_change_along_estuary.png", dpi=200)


# %% --- 2: bed change versus initial bed level --------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
for col, (reach, (lo, hi)) in enumerate(REACHES_KM.items()):
    for row, flow in enumerate(("peak_flow", "mean_flow")):
        ax = axes[row, col]
        names = [n for n in data if n.endswith(flow)] + (["constant"] if flow == "mean_flow" else [])
        for name in names:
            d = data[name]
            m = strip_mask(d) & ((d["x_km"] >= lo) & (d["x_km"] < hi))[None, :] \
                & np.isfinite(d["bed0"]) & np.isfinite(d["dz_day"])
            edges = np.arange(np.floor(np.nanmin(d["bed0"][m])), np.ceil(np.nanmax(d["bed0"][m])) + BED_BIN_M, BED_BIN_M)
            idx = np.digitize(d["bed0"][m], edges) - 1
            n = np.bincount(idx, minlength=edges.size)
            mean = np.bincount(idx, weights=d["dz_day"][m], minlength=edges.size) / np.maximum(n, 1)
            ok = n >= MIN_CELLS_PER_BIN
            centres = edges + BED_BIN_M / 2
            ax.plot(centres[ok], 1000 * mean[ok], marker="o", ms=3, color=COLORS[name], label=name)
            # typical low-water level of the reach, for orientation
        wl = np.nanmedian(data[names[0]]["wl_min"][strip_mask(data[names[0]])
                                                   & ((data[names[0]]["x_km"] >= lo) & (data[names[0]]["x_km"] < hi))[None, :]])
        ax.axvline(wl, color="tab:blue", ls="--", lw=0.8, label="median low-water level")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{reach}, {flow.replace('_', ' ')}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("Mean bed change [mm day$^{-1}$]")
        if row == 1:
            ax.set_xlabel("Initial bed level [m]  (low = channel, high = margins / bars)")
fig.suptitle("2  Which elevations gain or lose sediment? (peak and mean on their own y-scale)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "2_bed_change_vs_bed_level.png", dpi=200)


# %% --- 3: cross-sections ------------------------------------------------------
fig, axes = plt.subplots(3, len(SECTIONS_KM), figsize=(13, 8), sharex=True)
for col, x0 in enumerate(SECTIONS_KM):
    for row in range(3):
        ax = axes[row, col]
        for name, d in data.items():
            cols = np.abs(d["x_km"] - x0) <= X_HALF_WINDOW_M / 1000
            if row == 0:
                if name not in ("constant", "pm5_n1_mean_flow", "pm3_n5_mean_flow"):
                    continue          # mean and peak start from the same bed
                ax.plot(d["y_km"], np.nanmean(d["bed0"][:, cols], axis=1), color=COLORS[name], label=name)
                ax.plot(d["y_km"], np.nanmean(d["wl_min"][:, cols], axis=1), color=COLORS[name], ls="--", lw=0.8)
            else:
                flow = "peak_flow" if row == 1 else "mean_flow"
                if not (name.endswith(flow) or (row == 2 and name == "constant")):
                    continue
                ax.plot(d["y_km"], 1000 * np.nanmean(d["dz_day"][:, cols], axis=1),
                        color=COLORS[name], label=name)
        if row > 0:
            ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0, col].set_title(f"x = {x0:g} km (mean over ±{X_HALF_WINDOW_M:g} m)", fontsize=9)
axes[0, 0].set_ylabel("Initial bed [m]\n(dashed: low-water level)")
axes[1, 0].set_ylabel("Bed change, PEAK\n[mm day$^{-1}$]")
axes[2, 0].set_ylabel("Bed change, MEAN\n[mm day$^{-1}$]")
for ax in axes[-1]:
    ax.set_xlabel("y [km]")
    ax.set_xlim(Y_LIMITS_M[0] / 1000, Y_LIMITS_M[1] / 1000)
fig.suptitle("3  Across the estuary: does the channel or its flanks gain/lose?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "3_cross_sections.png", dpi=200)

print(f"Saved figures 1-3 in {OUTPUT_DIR}")


# %% --- 4: boundary supply versus pick-up from the bed -------------------------
# Section fluxes and recomputed MPM capacity written by plot_map_hydro_sedimentbudgets.py
# (time means over the run, + = seaward). F(x) / F(45 km) > 1: the flow has picked
# sediment up from the bed between 45 km and x; < 1: it has dropped part of the inflow.
BUDGET_DIR = OUTPUT_DIR.parent
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in RUNS:
    prof = np.genfromtxt(BUDGET_DIR / name / "csv" / "capacity_vs_flux_profile.csv",
                         delimiter=",", names=True)
    i_in = np.argmin(np.abs(prof["x_km"] - 45.0))
    axes[0].plot(prof["x_km"], prof["flux"] / prof["flux"][i_in], marker="o", ms=3,
                 color=COLORS[name], label=f"{name} (inflow {prof['flux'][i_in]:.4f} m$^3$/s)")
    axes[1].plot(prof["x_km"], prof["flux"] / prof["capacity"], marker="o", ms=3,
                 color=COLORS[name], label=name)
axes[0].axhline(1, color="k", lw=0.8)
axes[0].set_ylabel("Net flux F(x) / inflow F(45 km)  [-]\n>1 picked up from bed, <1 deposited")
axes[0].set_title("4a  How much of the flux is boundary supply?", fontsize=9)
axes[1].axhline(1, color="k", lw=0.8)
axes[1].set_ylim(0, 1.5)
axes[1].set_ylabel("Net flux / net MPM capacity  [-]")
axes[1].set_title("4b  Does the flux follow the local capacity?", fontsize=9)
for ax in axes:
    ax.set_xlabel("x [km] (sea left, river right)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "4_supply_vs_pickup.png", dpi=200)
print(f"Saved figure 4 in {OUTPUT_DIR}")
