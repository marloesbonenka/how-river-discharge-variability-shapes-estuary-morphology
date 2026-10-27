# %% Peak vs mean flow: where does the river sediment go? (measured bed change only)
"""
Reasoning figure for the detailed-hydro runs (2 days, MorFac = 1, restarted from the
final bed of each morphodynamic scenario; the mean- and peak-flow run of a scenario
start from the SAME bed).

Question: variable discharge gives higher bars but no deeper channel. Is channel scour
during peaks refilled during mean flow? Answer from these runs: no, the peak does not
scour the channel bed in the first place (except in the lower estuary); it lowers the
channel MARGINS (mm/day, too little to change the wet width measurably), the channel
bed fills, and in the lower estuary the channel feeds the bars. Mean flow is 10-20x
slower, moves sediment from the channel to its margins and does not refill the channel.

Volume changes come from the bed level the model itself computed (mesh2d_mor_bl,
change since the first output x cell area); no transport reconstruction. The
along-estuary flux profiles are read from the CSVs of plot_map_hydro_sedimentbudgets.py
(capacity_vs_flux_profile.csv per run).

Channel footprint: ONE footprint per scenario, used for its mean- and its peak-flow
run (the constant run uses its own): faces wet at LOW water during the last
WET_WINDOW_H of the MEAN-flow run (min water level - initial bed > WET_DEPTH_M),
largest connected component. A footprint from each run's own water level grows by
~50% at peak flow and moves low bars into the "channel" (the earlier, misleading
"channels gain, bars lose at peak" came from that).
Location classes inside the strip Y_LIMITS_M: channel bed (footprint), channel
margin (<= MARGIN_M from the footprint), higher ground (further away: bars, banks,
floodplain). Isolated pits are left out.

Panels
(a) schematic of the routing at peak flow (pm5_n1), annotated with the measured rates
(b) net seaward sediment flux along the estuary: supply at capacity, pick-up, drop-out
(c) whole-width bed volume change per 5-km zone
(d) peak flow: change per zone split into channel bed / margin / higher ground
(e) mean flow: the same split (own y-scale)

Caveats: the pm3_n5 peak run is forced with the same 5x peak (2500 m3/s) as pm5_n1;
the 'mean-flow' runs use 500 m3/s while the variable hydrographs sit at 400 m3/s ~90%
of the time; none of the runs simulates the recession after a peak.
"""

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from functions.F_general import apply_plot_style

# %% --- CONFIGURATION ---------------------------------------------------------
MODEL_ROOT = Path(r"U:\PhDNaturalRhythmEstuaries\Models"
                  r"\2_RiverDischargeVariability_domain45x15_Gaussian")
BASE_PATH = MODEL_ROOT / "Model_Output" / "Q500"
RUN_DIR = BASE_PATH / "detailed-hydro-run"
CACHE_DIR = BASE_PATH / "cached_data"
BUDGET_DIR = RUN_DIR / "output_plots" / "sediment_budgets"   # CSVs of the budget script
OUTPUT_DIR = BUDGET_DIR

RUNS = {
    "constant": "dhr_01_Qr500_pm1_n0.9724783",
    "pm3_n5_mean_flow": "dhr_03_Qr500_pm3_n5_mean.964903",
    "pm3_n5_peak_flow": "dhr_03_Qr500_pm3_n5_peak.964907",
    "pm5_n1_mean_flow": "dhr_05_Qr500_pm5_n1_mean.964908",
    "pm5_n1_peak_flow": "dhr_05_Qr500_pm5_n1_peak.965221",
}
# footprint source per run: the mean-flow run of the same scenario
FOOTPRINT_FROM = {
    "constant": "constant",
    "pm3_n5_mean_flow": "pm3_n5_mean_flow", "pm3_n5_peak_flow": "pm3_n5_mean_flow",
    "pm5_n1_mean_flow": "pm5_n1_mean_flow", "pm5_n1_peak_flow": "pm5_n1_mean_flow",
}
MAIN = "pm5_n1"     # scenario of the schematic and the bars in (d)/(e)
OTHER = "pm3_n5"    # drawn as hollow markers in (d)/(e)

BED_VAR = "mesh2d_mor_bl"
WL_VAR = "mesh2d_s1"
WET_WINDOW_H = 12.0
WET_DEPTH_M = 0.1
MARGIN_M = 150.0    # channel margin = within this distance of the channel footprint

X_LIMITS_M = (20_000.0, 45_000.0)   # zones 20-45 km
Y_LIMITS_M = (5_000.0, 10_000.0)    # estuary strip (bed change outside it is ~0)
ZONE_EDGES_KM = np.arange(20.0, 46.0, 5.0)
LOWER_EDGE_KM = 25.0                # lower estuary (bars) / upper estuary split
FLUX_SMOOTH_KM = 3                  # running mean of the 1-km flux profiles

RUN_COLORS = {
    "constant": "#7f7f7f",
    "pm3_n5_mean_flow": plt.cm.Blues(0.55), "pm5_n1_mean_flow": plt.cm.Blues(0.92),
    "pm3_n5_peak_flow": plt.cm.Oranges(0.55), "pm5_n1_peak_flow": plt.cm.Oranges(0.92),
}
RUN_LABELS = {
    "constant": "constant", "pm3_n5_mean_flow": "pm3_n5 mean", "pm3_n5_peak_flow": "pm3_n5 peak",
    "pm5_n1_mean_flow": "pm5_n1 mean", "pm5_n1_peak_flow": "pm5_n1 peak",
}
# location classes (validated categorical palette: CVD dE >= 8.4, labels give contrast relief)
CLASSES = ["channel", "margin", "high"]
CLASS_COLORS = {"channel": "#199e70", "margin": "#c98500", "high": "#6250d6"}
CLASS_LABELS = {"channel": "channel bed", "margin": f"channel margin <{MARGIN_M:.0f} m",
                "high": "bars / banks / high ground"}
INK = "#333333"
MUTED = "#8a8a8a"


# %% --- DATA: bed level at every output + low-water level, cached per run ------
def read_run(folder):
    """Face coordinates, bed level (n_times, n_faces) and min water level of the
    last WET_WINDOW_H, from the partitioned MAP files (own faces only)."""
    cache = CACHE_DIR / f"routing_bed_wl_{folder.replace('.', '_')}.npz"
    if cache.exists():
        return dict(np.load(cache))
    fx, fy, bl, wl_min = [], [], [], []
    hours = None
    for f in sorted(glob.glob(str(RUN_DIR / folder / "output" / "*_map.nc"))):
        part = int(Path(f).stem.split("_")[-2])
        with xr.open_dataset(f) as ds:
            own = ds["mesh2d_flowelem_domain"].values == part
            t = ds["time"].values
            hours = (t - t[0]) / np.timedelta64(1, "h")
            last = hours >= hours[-1] - WET_WINDOW_H
            fx.append(ds["mesh2d_face_x"].values[own])
            fy.append(ds["mesh2d_face_y"].values[own])
            bl.append(ds[BED_VAR].values[:, own])
            wl_min.append(np.nanmin(ds[WL_VAR].values[last][:, own], axis=0))
    out = {"x": np.concatenate(fx), "y": np.concatenate(fy),
           "bl": np.concatenate(bl, axis=1), "wl_min": np.concatenate(wl_min),
           "hours": hours}
    np.savez_compressed(cache, **out)
    print(f"  cached {cache.name}")
    return out


def to_grid(run):
    """Put the face fields on the rectilinear grid (rows = y, columns = x)."""
    ux, ix = np.unique(np.round(run["x"], 1), return_inverse=True)
    uy, iy = np.unique(np.round(run["y"], 1), return_inverse=True)

    def grid(values, fill=np.nan):
        g = np.full(values.shape[:-1] + (uy.size, ux.size), fill)
        g[..., iy, ix] = values
        return g

    return {"xc": ux, "yc": uy, "dx": np.gradient(ux), "dy": np.gradient(uy),
            "area": np.gradient(uy)[:, None] * np.gradient(ux)[None, :],
            "bl": grid(run["bl"]), "wl_min": grid(run["wl_min"]), "hours": run["hours"]}


def location_classes(g):
    """Channel (wet at low water on the initial bed, largest component), margin
    (<= MARGIN_M from it), higher ground; isolated pits are excluded."""
    with np.errstate(invalid="ignore"):
        wet = (g["wl_min"] - g["bl"][0]) > WET_DEPTH_M
    labels, _ = ndimage.label(wet, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    channel = labels == int(np.argmax(sizes))
    pits = wet & ~channel
    dist = ndimage.distance_transform_edt(~channel, sampling=(g["dy"].mean(), g["dx"].mean()))
    margin = ~channel & ~pits & (dist <= MARGIN_M)
    return {"channel": channel, "margin": margin, "high": ~channel & ~pits & ~margin}


print("Reading MAP output ...")
grids = {name: to_grid(read_run(folder)) for name, folder in RUNS.items()}
classes = {name: location_classes(grids[name]) for name in set(FOOTPRINT_FROM.values())}


# %% --- BUDGETS ----------------------------------------------------------------
def budgets(name):
    """Volume change [m3/day] per zone for each location class and in total."""
    g = grids[name]
    xc, yc = g["xc"], g["yc"]
    strip = ((yc >= Y_LIMITS_M[0]) & (yc <= Y_LIMITS_M[1]))[:, None] \
        & ((xc >= X_LIMITS_M[0]) & (xc < X_LIMITS_M[1]))[None, :]
    days = g["hours"][-1] / 24.0
    dv = np.nan_to_num(g["bl"][-1] - g["bl"][0]) * g["area"] / days     # [m3/day]
    zone = np.digitize(xc / 1000.0, ZONE_EDGES_KM) - 1
    masks = dict(classes[FOOTPRINT_FROM[name]], total=np.ones_like(strip))
    return {key: np.array([(dv * (m & strip))[:, zone == z].sum()
                           for z in range(ZONE_EDGES_KM.size - 1)])
            for key, m in masks.items()}


budget = {name: budgets(name) for name in RUNS}
zone_labels = [f"{a:.0f}-{b:.0f}" for a, b in zip(ZONE_EDGES_KM[:-1], ZONE_EDGES_KM[1:])]
print("\nMeasured volume change [m3/day] per zone " + ", ".join(zone_labels) + " km")
for name in RUNS:
    for key in CLASSES + ["total"]:
        print(f"  {name:18s} {key:8s} " + " ".join(f"{v:7.0f}" for v in budget[name][key])
              + f"   sum {budget[name][key].sum():7.0f}")


def read_flux(name):
    """Seaward net sediment flux profile [m3/s] from the budget script's CSV."""
    data = np.genfromtxt(BUDGET_DIR / name / "csv" / "capacity_vs_flux_profile.csv",
                         delimiter=",", names=True)
    kernel = np.ones(FLUX_SMOOTH_KM) / FLUX_SMOOTH_KM
    smooth = np.convolve(np.pad(data["flux"], FLUX_SMOOTH_KM // 2, mode="edge"),
                         kernel, mode="valid")
    return data["x_km"], data["flux"], smooth


flux = {name: read_flux(name) for name in RUNS}


# %% --- FIGURE -----------------------------------------------------------------
def panel_label(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8, fontweight="bold", color=INK)


def sea_river(ax):
    ax.text(0.01, 0.03, "sea", transform=ax.transAxes, fontsize=7, color=MUTED)
    ax.text(0.99, 0.03, "river", transform=ax.transAxes, fontsize=7, color=MUTED, ha="right")


def draw_schematic(ax):
    """Plan view of the estuary with the measured peak-flow routing (MAIN scenario)."""
    peak, mean = budget[f"{MAIN}_peak_flow"], budget[f"{MAIN}_mean_flow"]
    lo = ZONE_EDGES_KM[:-1] < LOWER_EDGE_KM
    up = ~lo
    m3 = lambda v: f"{v:+,.0f}".replace(",", " ")
    x_flux, fpk, _ = flux[f"{MAIN}_peak_flow"]      # raw (unsmoothed) profiles
    _, fmn, _ = flux[f"{MAIN}_mean_flow"]
    to_day = 86400.0
    i_in, i_out = np.argmax(x_flux), np.argmin(x_flux)

    # symbolic funnel: half-widths in km (not to scale)
    x = np.linspace(20, 45, 200)
    s = (45 - x) / 25
    bank = 0.45 + 2.3 * s ** 1.4
    chan = 0.18 + 0.30 * s
    marg = chan + 0.22
    for sign in (1, -1):
        ax.fill_between(x, sign * marg, sign * bank, color=CLASS_COLORS["high"], alpha=0.18, lw=0)
        ax.fill_between(x, sign * chan, sign * marg, color=CLASS_COLORS["margin"], alpha=0.35, lw=0)
        ax.plot(x, sign * bank, color=MUTED, lw=0.8)
    ax.fill_between(x, -chan, chan, color=CLASS_COLORS["channel"], alpha=0.35, lw=0)
    ax.axvline(LOWER_EDGE_KM, color=MUTED, lw=0.6, ls=":")

    arrow = dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=9)
    # river supply and export
    ax.annotate("", xy=(43.0, 0), xytext=(46.3, 0), arrowprops=dict(arrow, lw=2.4))
    ax.text(45.9, 0.55, f"river supply at capacity\n{fpk[i_in] * to_day:,.0f} m$^3$/d "
            f"(×{fpk[i_in] / fmn[i_in]:.0f} mean flow)".replace(",", " "),
            ha="right", va="bottom", fontsize=7, color=INK)
    ax.annotate("", xy=(17.9, 0), xytext=(21.0, 0), arrowprops=dict(arrow, lw=1.4))
    ax.text(18.1, -0.55, f"export\n{fpk[i_out] * to_day:,.0f} m$^3$/d".replace(",", " "),
            ha="left", va="top", fontsize=7, color=INK)

    # upper estuary: margins erode into the channel (widening), bed fills
    for xa in (31.0, 36.0, 41.0):
        sa = (45 - xa) / 25
        c, mg = 0.18 + 0.30 * sa, 0.18 + 0.30 * sa + 0.22
        for sign in (1, -1):
            ax.annotate("", xy=(xa, sign * (c - 0.05)), xytext=(xa, sign * (mg + 0.25)),
                        arrowprops=dict(arrow, color=CLASS_COLORS["margin"], lw=1.3))
    ax.text(35.0, 1.35, f"margins lower {m3(peak['margin'][up].sum())}\n(< 3 mm/day of lowering, no measurable widening)",
            ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(35.0, -1.35, f"channel bed fills {m3(peak['channel'][up].sum())}\n"
            f"high ground {m3(peak['high'][up].sum())}",
            ha="center", va="top", fontsize=7, color=INK)

    # lower estuary: channel feeds the bars
    for xa in (21.3, 23.6):
        sa = (45 - xa) / 25
        c = 0.18 + 0.30 * sa
        for sign in (1, -1):
            ax.annotate("", xy=(xa, sign * (c + 0.9)), xytext=(xa, sign * (c - 0.1)),
                        arrowprops=dict(arrow, color=CLASS_COLORS["channel"], lw=1.3))
    bars_lo = peak["margin"][lo].sum() + peak["high"][lo].sum()
    ax.text(22.45, 1.5, f"channel {m3(peak['channel'][lo].sum())}\n→ bars {m3(bars_lo)}",
            ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(22.45, -1.5, "bars grow,\nchannel scours",
            ha="center", va="top", fontsize=7, color=INK)

    ax.text(LOWER_EDGE_KM - 0.2, 2.95, "lower estuary", ha="right", fontsize=7, color=MUTED)
    ax.text(LOWER_EDGE_KM + 0.2, 2.95, "upper estuary", ha="left", fontsize=7, color=MUTED)
    ax.text(45.9, -2.3, f"mean flow ({MAIN}): channel bed {m3(mean['channel'].sum())}, "
            f"margins {m3(mean['margin'].sum())}\n→ the channel slowly feeds its margins; "
            "nothing refills it",
            ha="right", va="top", fontsize=7, color=INK)
    ax.set_xlim(17.6, 46.5)
    ax.set_ylim(-3.0, 3.2)
    ax.set_xticks(ZONE_EDGES_KM)
    ax.set_xlabel("Distance x [km]")
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def draw_flux(ax):
    for name in ["constant", f"{MAIN}_mean_flow", f"{OTHER}_peak_flow", f"{MAIN}_peak_flow"]:
        x, raw, smooth = flux[name]
        ax.plot(x, raw, color=RUN_COLORS[name], lw=0.6, alpha=0.35)
        ax.plot(x, smooth, color=RUN_COLORS[name], lw=1.8, label=RUN_LABELS[name])
    x, _, smooth = flux[f"{MAIN}_peak_flow"]
    x_max = x[np.argmax(smooth)]
    ax.axvspan(20, x_max, color=CLASS_COLORS["channel"], alpha=0.07, lw=0)
    ax.axvspan(x_max, 45, color=CLASS_COLORS["margin"], alpha=0.07, lw=0)
    ax.text((20 + x_max) / 2, 0.06, "flux drops seaward:\ndeposition", ha="center",
            fontsize=7, color=INK)
    ax.text((x_max + 45) / 2, 0.06, "flux grows seaward:\npick-up", ha="center",
            fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 0.12)
    ax.set_xlim(20, 45)
    ax.set_ylabel("Net seaward sediment flux\n[m$^3$ s$^{-1}$]")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="center", bbox_to_anchor=(0.5, 0.43))
    ax.grid(alpha=0.25, which="major")
    sea_river(ax)


def draw_zone_total(ax):
    names = ["constant", f"{OTHER}_mean_flow", f"{MAIN}_mean_flow",
             f"{OTHER}_peak_flow", f"{MAIN}_peak_flow"]
    xz = np.arange(len(zone_labels))
    w = 0.16
    for k, name in enumerate(names):
        ax.bar(xz + (k - 2) * w, budget[name]["total"], w * 0.88, color=RUN_COLORS[name],
               label=RUN_LABELS[name])
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(xz)
    ax.set_xticklabels(zone_labels)
    ax.set_xlabel("Zone [km]")
    ax.set_ylabel("Bed volume change, full width\n[m$^3$ day$^{-1}$, + = deposition]")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.25, axis="y")


def draw_split(ax, flow):
    main, other = budget[f"{MAIN}_{flow}"], budget[f"{OTHER}_{flow}"]
    xz = np.arange(len(zone_labels))
    w = 0.26
    for k, key in enumerate(CLASSES):
        xs = xz + (k - 1) * w
        ax.bar(xs, main[key], w * 0.88, color=CLASS_COLORS[key], label=CLASS_LABELS[key])
        ax.plot(xs, other[key], "D", ms=4, mfc="white", mec=INK, mew=0.8)
    if flow == "mean_flow":
        const = budget["constant"]
        for k, key in enumerate(CLASSES):
            ax.plot(xz + (k - 1) * w, const[key], "_", ms=9, mew=1.6, color=RUN_COLORS["constant"])
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(xz)
    ax.set_xticklabels(zone_labels)
    ax.set_xlabel("Zone [km]")
    ax.grid(alpha=0.25, axis="y")


apply_plot_style("AGU", font_size=8)
fig = plt.figure(figsize=(180 / 25.4, 215 / 25.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.55, wspace=0.28)
ax_a = fig.add_subplot(gs[0, :])
ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_d, ax_e = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

draw_schematic(ax_a)
panel_label(ax_a, "a", f"Peak flow ({MAIN}): margins feed the upper channel, the lower channel "
            "feeds the bars [m$^3$ day$^{-1}$]")
draw_flux(ax_b)
panel_label(ax_b, "b", "1  Peak imports sediment at capacity")
draw_zone_total(ax_c)
panel_label(ax_c, "c", "2  Estuary fills where the flux drops")
draw_split(ax_d, "peak_flow")
ax_d.set_ylabel("Bed volume change\n[m$^3$ day$^{-1}$, + = deposition]")
panel_label(ax_d, "d", "3  Peak: margins → channel (25-45 km), channel → bars (20-25 km)")
draw_split(ax_e, "mean_flow")
panel_label(ax_e, "e", "4  Mean flow: 10-20x slower, channel → margins")

handles = [Patch(color=CLASS_COLORS[k], label=CLASS_LABELS[k] + f" ({MAIN})") for k in CLASSES] + [
    Line2D([], [], ls="", marker="D", ms=4, mfc="white", mec=INK, label=OTHER),
    Line2D([], [], ls="", marker="_", ms=9, mew=1.6, color=RUN_COLORS["constant"],
           label="constant run (e)")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, 0.005))
fig.text(0.5, -0.025,
         "Measured bed-level change (mesh2d_mor_bl) of 2-day runs restarted from each scenario's "
         "final bed. Channel = wet at low water under mean flow, the same footprint for the mean "
         "and the peak run.\nBoth peak runs ramp 500 → 2500 → 2180 m³/s (a 5x peak, also for "
         "pm3_n5). No run simulates the recession after a peak. Flux in (b): time mean over the "
         "run, 3-km running mean (thin: raw).",
         ha="center", va="top", fontsize=6.5, color=MUTED)
fig.subplots_adjust(top=0.96, bottom=0.09, left=0.1, right=0.98)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUTPUT_DIR / f"peak_mean_sediment_routing.{ext}", dpi=300, bbox_inches="tight")
print(f"\nSaved {OUTPUT_DIR / 'peak_mean_sediment_routing.png'}")
