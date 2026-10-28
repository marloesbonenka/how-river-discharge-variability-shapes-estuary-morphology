# %% Fig. 3: base run (constant discharge): morphological development, dynamic equilibrium
#    AND the hydrodynamics of the final morphology
"""
Opening figure of the results section. Base run = 01_Qr500_pm1_n0 (constant discharge).

Morphodynamic run (map output, date axis of the run, e.g. 2031 ... 2055):
(a-c) detrended bed level at the start, end of spin-up and end of the run
(d)   braiding index per 1-km segment (initial / end of spin-up / equilibrium mean +/- 2 sigma)
(e)   braiding index vs model time for three reaches (one sample per year)

Detailed-hydro run dhr_01 (2 days of hydrodynamics on the FINAL (t = 2055) bed; its own
dates 2031-01-01 ... 01-03 are NOT on the morphodynamic time axis, so hours are shown):
(f)   water-level anomaly at the thalweg, x vs time (tidal wave propagation)
(g)   discharge at km20 (downstream, sea) and km45 (upstream, river)
(h)   water-level envelope (min-max, final 12 h) along the thalweg + thalweg bed
(i)   water depth at the thalweg (min-max band)

Braiding index: number of separate wet channels per cross-section, on the RAW bed level
at a FIXED reference stage (minimum water level of the detailed-hydro run), with a
wet-depth threshold, so the time series reflects bed change only.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from functions.F_general import (
    apply_plot_style,
    create_terrain_colormap,
    setup_variability_run_context,
)
from functions.F_loaddata import get_stitched_map_run_paths
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_morphological_activity import build_centerline_reference

# harmless: xugrid CRS notices (no CRS in the D-Flow FM files) and all-NaN slices
# (dry faces in the final-12-h min/max; the thalweg faces are always wet)
warnings.filterwarnings("ignore", category=UserWarning, module="xugrid")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

# %% --- CONFIGURATION ---------------------------------------------------------

DISCHARGE = 500
BASE_DIRECTORY = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian"
)
run_context = setup_variability_run_context(
    base_directory=BASE_DIRECTORY, discharge=DISCHARGE, scenarios_to_process=None,
    analyze_noisy=False, MORFAC=False,
)
BASE_PATH = run_context["base_path"]
CACHE_DIR = run_context["cache_dir"]
TIMED_OUT_DIR = run_context["timed_out_dir"]

# Base run: constant discharge
MORPH_RUN = BASE_DIRECTORY / "Model_Output" / "Q500" / "01_Qr500_pm1_n0.9600302"
DETAILED_ROOT = BASE_DIRECTORY / "Model_Output" / "Q500" / "detailed-hydro-run"
# constant scenario: Q = Qmean at all times
DETAILED_RUN = DETAILED_ROOT / "dhr_01_Qr500_pm1_n0.9724783"
# peaky scenario (pm5_n1): same final-morphology, run at the constant flow and at a peak
MEAN_DETAILED_RUN = DETAILED_ROOT / "dhr_05_Qr500_pm5_n1_mean.964908"
PEAK_DETAILED_RUN = DETAILED_ROOT / "dhr_05_Qr500_pm5_n1_peak.965221"
# morphodynamic run of the peaky scenario, only used for the one-year hydrograph in (f)
HYDRO_MORPH_RUN = BASE_DIRECTORY / "Model_Output" / "Q500" / "05_Qr500_pm5_n1.9517572"
HYDROGRAPH_KM = 45
YEAR_DAYS_HYDROGRAPH = 365.25
YEAR_DAYS = 365.25
COLOR_PEAK = "#b2182b"
COLOR_MEAN = "#2166ac"
COLOR_CONST = "#4d4d4d"
LAST_HOURS = 24

BED_VAR = "mesh2d_mor_bl"
WL_VAR = "mesh2d_s1"
HIS_FILENAME = "output/FlowFM_0000_his.nc"
HIS_Q_VAR = "cross_section_discharge"

# The time axis of the morphodynamic map output is used exactly as it is (datetime64),
# as in the map-plot script. One "year" = one year on that axis.
# Base run = constant discharge, so there is no hydrograph phase to match: every map
# output time after spin-up is a valid sample of the dynamic equilibrium.
MORFAC_FACTOR = 100         # 6 hydrodynamic years x 100 = 600 morphological years
SPINUP_YEARS = 50.0         # end of spin-up in MORPHOLOGICAL years (placeholder; read from panel e)
MAX_SAMPLES = 150           # at most this many map times are evaluated (evenly spread)

CACHE_BBOX = [1, 1, 45000, 15000]
X_KM = np.arange(20, 46)
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y = 500
WET_DEPTH_THRESHOLD_M = 0.1

REACHES = {
    "tidal (km 20-25)":   (20, 25),
    "middle (km 30-35)":  (30, 35),
    "fluvial (km 40-45)": (40, 45),
}
REACH_COLORS = ["#08519c", "#3182bd", "#9ecae1"]
COLOR = "#08519c"

DETREND_LAND_THRESHOLD = 6.0
CENTERLINE_XMIN, CENTERLINE_XMAX, CENTERLINE_Y = 20000, 45000, 7500
BED_CMAP = create_terrain_colormap()
BED_LIM = 5
BED_LIM_INITIAL = 2         # colour limits of the actual initial bed: -2 ... 2 m

MAP_XLIM = (20_000, 45_000)
MAP_YLIM = (5_000, 10_000)

OUT_DIR = BASE_PATH / "output_plots" / "fig3_morphology_and_forcing"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

apply_plot_style("AGU", font_size=10, title_delta=0, tick_delta=0, **{"grid.linewidth": 0.5})

# %% --- FUNCTIONS -------------------------------------------------------------


def load_ds(run_dir, var_names, cache_dir, timed_out_dir=None):
    name = run_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(base_path=run_dir.parent, folder_name=name,
                  timed_out_dir=timed_out_dir, analyze_noisy=False)
    if timed_out_dir is None:                           # detailed-hydro run, as in the cross-section script
        kwargs["variability_map"] = None
    run_paths = get_stitched_map_run_paths(**kwargs) or [run_dir]   # morph run: as in the map-plot script
    ds = load_or_update_map_cache_multi(
        cache_dir=cache_dir, folder_name=name, run_paths=run_paths,
        var_names=var_names, bbox=CACHE_BBOX, append_time=True, append_vars=True,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, None),
    )
    if ds is None:
        raise FileNotFoundError(f"No cached map data for {name}")
    return ds


def face_dim(ds, var):
    dims = [d for d in ds[var].dims if d != "time"]
    assert len(dims) == 1, f"Expected one face dim in {var}, got {ds[var].dims}"
    return dims[0]


def build_section_index(ds):
    """Nearest-face index per (cross-section, y-sample); built once per grid."""
    fc = np.asarray(ds.grid.face_coordinates, dtype=float)
    tree = cKDTree(fc[:, :2])
    y = np.linspace(*Y_RANGE_M, N_Y)
    idx = np.empty((len(X_KM), N_Y), dtype=int)
    for i, xk in enumerate(X_KM):
        idx[i] = tree.query(np.column_stack((np.full(N_Y, xk * 1000.0), y)))[1]
    return y, idx


def count_channels(bed_sec, wl_sec, thr=WET_DEPTH_THRESHOLD_M):
    """Number of separate wet channels per cross-section. Inputs: (n_sections, n_y)."""
    depth = wl_sec - bed_sec
    wet = np.isfinite(depth) & (depth > thr)
    return wet[:, 0].astype(int) + (wet[:, 1:] & ~wet[:, :-1]).sum(axis=1)


def segment_mean(counts):
    """1-km segments: mean of the two bounding cross-sections."""
    return 0.5 * (counts[..., :-1] + counts[..., 1:])


def sample_indices(times):
    """Evenly spread subset of the map output times; returns (indices, MORPHOLOGICAL years since start)."""
    idx = np.unique(np.linspace(0, len(times) - 1, min(len(times), MAX_SAMPLES)).astype(int))
    hydro_days = (times[idx] - times[0]) / np.timedelta64(1, "D")
    return idx, hydro_days * MORFAC_FACTOR / YEAR_DAYS


def detailed_profiles(run_dir):
    """
    Hydrodynamics of the detailed-hydro run on its (final) bed.
    Returns per-section thalweg quantities from the final 12 h, the full-length thalweg
    water-level series (for the x-t panel) and the fixed reference stage wl_min_full.
    """
    ds = load_ds(run_dir, [WL_VAR, BED_VAR], cache_dir=run_dir.parent / "cached_data")
    try:
        fd_bed, fd_wl = face_dim(ds, BED_VAR), face_dim(ds, WL_VAR)
        _, idx = build_section_index(ds)
        t = np.asarray(ds.time.values).astype("datetime64[ns]")
        sel = np.where(t >= t[-1] - np.timedelta64(12, "h"))[0]
        wl_fin = ds[WL_VAR].isel(time=sel).transpose("time", fd_wl).values
        wl_max = np.nanmax(wl_fin, axis=0)[idx]
        wl_min = np.nanmin(wl_fin, axis=0)[idx]
        bed = ds[BED_VAR].isel(time=-1).transpose(fd_bed).values[idx]
        rows = np.arange(len(X_KM))
        ith = np.nanargmin(bed, axis=1)                    # thalweg sample per section
        faces = idx[rows, ith]
        bed_t, wmax, wmin = bed[rows, ith], wl_max[rows, ith], wl_min[rows, ith]
        return dict(
            bed=bed_t, wl_max=wmax, wl_min=wmin,
            depth_max=np.clip(wmax - bed_t, 0, None),
            depth_min=np.clip(wmin - bed_t, 0, None),
            wl_min_full=wl_min, bed_full=bed,
        )
    finally:
        ds.close()


def _decode(name_array):
    out = []
    for item in np.asarray(name_array):
        if isinstance(item, bytes):
            s = item.decode("utf-8", errors="ignore")
        elif np.ndim(item) == 0:
            s = str(item)
        else:
            s = "".join(c.decode("utf-8", errors="ignore") if isinstance(c, bytes) else str(c)
                        for c in item)
        out.append(s.replace("\x00", "").strip())
    return out


def his_discharge(his_path, km):
    """Time (datetime64) and discharge at ObservationCrossSection_Estuary_km{km}."""
    ds = xr.open_dataset(his_path)
    try:
        names = _decode(ds["cross_section_name"].values)
        section_name = f"ObservationCrossSection_Estuary_km{km}"
        if section_name in names:                       # same matching as the his-file script
            i = names.index(section_name)
        else:
            matches = [j for j, n in enumerate(names) if section_name.lower() in n.lower()]
            if len(matches) != 1:
                raise ValueError(f"Cross-section '{section_name}': {len(matches)} matches in {names}")
            i = matches[0]
        da = ds[HIS_Q_VAR].isel(cross_section=i)
        for d in list(da.dims):
            if d != "time":
                da = da.isel({d: 0})
        return np.asarray(ds.time.values).astype("datetime64[ns]"), np.asarray(da.values)
    finally:
        ds.close()


def panel_letter(ax, letter):
    ax.text(0.005, 0.97, letter, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))


def clean(ax):
    ax.grid(True, ls="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# %% --- LOAD AND COMPUTE ------------------------------------------------------

# --- detailed-hydro run (2 days on the t = end-of-run bed) ---
prof_const = detailed_profiles(DETAILED_RUN)       # constant scenario, on its final bed
prof_mean = detailed_profiles(MEAN_DETAILED_RUN)   # peaky scenario, flow = constant-scenario flow
prof_peak = detailed_profiles(PEAK_DETAILED_RUN)   # peaky scenario, peak flow
wl_ref = prof_const["wl_min_full"]                 # fixed reference stage for the wet mask

SCEN = {"constant scenario": (DETAILED_RUN, prof_const, COLOR_CONST),
        "peaky scenario, mean flow": (MEAN_DETAILED_RUN, prof_mean, COLOR_MEAN),
        "peaky scenario, peak flow": (PEAK_DETAILED_RUN, prof_peak, COLOR_PEAK)}


def last_window(t, q, hours=LAST_HOURS):
    m = t >= t[-1] - np.timedelta64(hours, "h")
    return (t[m] - t[m][0]) / np.timedelta64(1, "h"), q[m]


q_series = {}   # (scenario, km) -> (hours, Q)
for sname, (rdir, _, _) in SCEN.items():
    for km in (20, 45):
        t, q = his_discharge(rdir / HIS_FILENAME, km)
        q_series[(sname, km)] = last_window(t, q)

# hydrograph of the variable scenario (one year) + constant level, at km45
t_h, q_h = his_discharge(HYDRO_MORPH_RUN / HIS_FILENAME, HYDROGRAPH_KM)
m_h = t_h <= t_h[0] + np.timedelta64(int(YEAR_DAYS_HYDROGRAPH * 86400e9), "ns")
days_h, q_h = (t_h[m_h] - t_h[0]) / np.timedelta64(1, "D"), q_h[m_h]
yr_h = days_h / YEAR_DAYS_HYDROGRAPH               # normalised time: fraction of the hydrological year
i_pk = int(np.nanargmax(q_h))
t_c, q_c = his_discharge(MORPH_RUN / HIS_FILENAME, HYDROGRAPH_KM)
q_const = float(np.nanmean(q_c))
i_mn = int(np.nanargmin(np.abs(q_h - q_const)))   # moment with the constant-scenario flow
print(f"Hydrograph km{HYDROGRAPH_KM}: peak {q_h[i_pk]:.0f} m3/s on day {days_h[i_pk]:.0f}; "
      f"constant run mean {q_const:.0f} m3/s")

# --- morphodynamic run ---
ds = load_ds(MORPH_RUN, [BED_VAR], cache_dir=CACHE_DIR, timed_out_dir=TIMED_OUT_DIR)
fdim = face_dim(ds, BED_VAR)
_, sec_idx = build_section_index(ds)
times = np.asarray(ds.time.values).astype("datetime64[ns]")
samp_i, samp_t = sample_indices(times)
print(f"Map times: {times[0]} -> {times[-1]} ({len(times)} outputs, "
      f"{(times[-1] - times[0]) / np.timedelta64(1, 'D') / YEAR_DAYS:.2f} hydrodynamic years = "
      f"{samp_t[-1]:.0f} morphological years at MorFac {MORFAC_FACTOR}); {len(samp_i)} sampled")
print("  first sampled morphological years: " + ", ".join(f"{d:.1f}" for d in samp_t[:12]))

# consistency check: the detailed run should sit on the final bed of the morphodynamic run
bed_end = ds[BED_VAR].isel(time=-1).transpose(fdim).values[sec_idx]
print(f"Bed check (final morph bed vs constant-scenario detailed-run bed): max |diff| = "
      f"{np.nanmax(np.abs(bed_end - prof_const['bed_full'])):.3f} m")

# peaky scenario (pm5_n1) morphodynamic run, same grid and same processing
ds_p = load_ds(HYDRO_MORPH_RUN, [BED_VAR], cache_dir=CACHE_DIR, timed_out_dir=TIMED_OUT_DIR)
times_p = np.asarray(ds_p.time.values).astype("datetime64[ns]")
samp_i_p, samp_t_p = sample_indices(times_p)
print(f"Peaky run map times: {times_p[0]} -> {times_p[-1]} ({len(times_p)} outputs, "
      f"{samp_t_p[-1]:.0f} morphological years)")

RUNS = {}   # scenario -> dict(ds, ref, idx, t)
for name, (d, i_, t_) in {"constant": (ds, samp_i, samp_t),
                          "peaky": (ds_p, samp_i_p, samp_t_p)}.items():
    RUNS[name] = dict(
        ds=d, idx=i_, t=t_,
        ref=build_centerline_reference(d, BED_VAR, 0, xmin=CENTERLINE_XMIN,
                                       xmax=CENTERLINE_XMAX, centerline_y=CENTERLINE_Y),
        k_mid=int(np.abs(t_ - 0.5 * t_[-1]).argmin()), k_end=len(i_) - 1,
    )


def raw_bed(run, k):
    """Actual bed level (not detrended); land above the threshold masked."""
    da = run["ds"][BED_VAR].isel(time=int(run["idx"][k]))
    raw = np.asarray(da.values)
    return da.copy(data=np.where(raw > DETREND_LAND_THRESHOLD, np.nan, raw))


def detrended_bed(run, k):
    da = run["ds"][BED_VAR].isel(time=int(run["idx"][k]))
    raw = np.asarray(da.values)
    return da.copy(data=np.where(raw > DETREND_LAND_THRESHOLD, np.nan, raw) - run["ref"])


# --- numbers for the text ---
print("\nHydrodynamics of the final morphology (detailed run):")
for sname, (_, pr, _) in SCEN.items():
    for km in (20, 45):
        q = q_series[(sname, km)][1]
        i = int(np.where(X_KM == km)[0][0])
        print(f"  [{sname}] km{km}: Q mean {q.mean():.0f} m3/s, tidal amplitude "
              f"+/-{(q.max() - q.min()) / 2:.0f} m3/s, tidal range "
              f"{pr['wl_max'][i] - pr['wl_min'][i]:.2f} m, thalweg depth "
              f"{pr['depth_min'][i]:.1f}-{pr['depth_max'][i]:.1f} m")

# %% --- FIGURE ----------------------------------------------------------------

fig = plt.figure(figsize=(7.5, 10.5), constrained_layout=True)
sf_top, sf_maps, sf_bot = fig.subfigures(3, 1, height_ratios=[2.3, 3.4, 2.3])

# (c-e) maps: initial bed, then constant and peaky scenario at the end of the run
axm = sf_maps.subplots(3, 1)
tick_x = np.arange(20_000, 45_001, 5_000)
layout = [("c", "constant", 0), ("d", "constant", "k_end"), ("e", "peaky", "k_end")]
for ax, (letter, scen, kk) in zip(axm, layout):
    run = RUNS[scen]
    k = kk if isinstance(kk, int) else run[kk]
    if letter == "c":          # initial bed: actual bed level, not detrended (identical for both runs)
        pc_a = raw_bed(run, 0).ugrid.plot(ax=ax, cmap=BED_CMAP, vmin=-BED_LIM_INITIAL,
                                          vmax=BED_LIM_INITIAL, add_colorbar=False, edgecolors="none")
        pc_a.set_clim(-BED_LIM_INITIAL, BED_LIM_INITIAL)
        label = "initial"
    else:                      # end beds: detrended w.r.t. the t=0 centreline profile
        pc = detrended_bed(run, k).ugrid.plot(ax=ax, cmap=BED_CMAP, vmin=-BED_LIM, vmax=BED_LIM,
                                              add_colorbar=False, edgecolors="none")
        label = f"{scen} scenario \u2013 end ({run['t'][k]:.0f} yr)"
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("y [km]")
    ax.set_xticks(tick_x)
    ax.set_xticklabels([f"{v / 1000:.0f}" for v in tick_x] if letter == "e" else [])
    ax.set_yticks([5_000, 7_500, 10_000])
    ax.set_yticklabels(["5", "7.5", "10"])
    ax.text(0.99, 0.95, label, transform=ax.transAxes, va="top", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    panel_letter(ax, letter)
axm[-1].set_xlabel("x [km]")
cb_a = sf_maps.colorbar(pc_a, ax=axm[0], pad=0.01, aspect=12)
cb_a.set_label("bed level [m]")
cb = sf_maps.colorbar(pc, ax=axm[1:], pad=0.01, aspect=22)
cb.set_label("detrended bed level [m]")

# (a, b, f, g) forcing and hydrodynamics of the final morphologies: constant vs peaky (mean and peak flow)
axg, axf = sf_top.subplots(1, 2)   # left: tide (discharge), right: hydrograph
axh, axi = sf_bot.subplots(1, 2)   # water-level envelope, thalweg depth

# (b, right) one-year hydrograph of the variable scenario, constant level, dots at mean and peak
axf.plot(yr_h, q_h, color="0.6", lw=1.1, label="peaky scenario")
axf.axhline(q_const, color=COLOR_CONST, ls="--", lw=1.1, label="constant scenario")
axf.plot(yr_h[i_mn], q_h[i_mn], "o", color=COLOR_MEAN, ms=6, zorder=5, label="mean flow")
axf.plot(yr_h[i_pk], q_h[i_pk], "o", color=COLOR_PEAK, ms=6, zorder=5, label="peak flow")
axf.set_xlabel("normalised time [t / year]")
axf.set_ylabel("upstream (river) discharge [m$^3$/s]")
axf.set_xlim(0, 1)
axf.legend(frameon=False, fontsize=7, loc="upper right")
panel_letter(axf, "b")

# (a, left) discharge at km20 (solid) and km45 (dashed), last 24 h of each detailed run
for sname, (_, _, c) in SCEN.items():
    for km, ls in ((20, "-"), (45, "--")):
        h, q = q_series[(sname, km)]
        axg.plot(h, q, color=c, ls=ls, lw=1.1)
axg.axhline(0, color="k", lw=0.5, alpha=0.5)
axg.set_xlabel("time [h]")
axg.set_ylabel("discharge [m$^3$/s]")
from matplotlib.lines import Line2D
axg.legend(handles=[Line2D([], [], color=COLOR_CONST, lw=1.1, label="constant"),
                    Line2D([], [], color=COLOR_MEAN, lw=1.1, label="peaky, mean flow"),
                    Line2D([], [], color=COLOR_PEAK, lw=1.1, label="peaky, peak flow"),
                    Line2D([], [], color="k", ls="-", lw=1.1, label="downstream (sea, km 20)"),
                    Line2D([], [], color="k", ls="--", lw=1.1, label="upstream (river, km 45)")],
           frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.02),
           ncol=2, borderaxespad=0.0, columnspacing=1.2, handlelength=1.8)
panel_letter(axg, "a")

# (f) water-level envelope (final 12 h) along the thalweg
axh.plot(X_KM, prof_const["bed"], color="0.5", lw=1.0, label="thalweg bed (constant)")
for sname, (_, pr, c) in SCEN.items():
    axh.fill_between(X_KM, pr["wl_min"], pr["wl_max"], color=c, alpha=0.3, lw=0)
    axh.plot(X_KM, 0.5 * (pr["wl_min"] + pr["wl_max"]), color=c, lw=1.2, label=sname)
axh.set_xlabel("x [km]")
axh.set_ylabel("elevation [m]")
axh.set_xlim(20, 45)
axh.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.02),
           ncol=2, borderaxespad=0.0, columnspacing=1.2, handlelength=1.8)
panel_letter(axh, "f")

# (g) thalweg water depth band
for sname, (_, pr, c) in SCEN.items():
    axi.fill_between(X_KM, pr["depth_min"], pr["depth_max"], color=c, alpha=0.3, lw=0)
    axi.plot(X_KM, 0.5 * (pr["depth_min"] + pr["depth_max"]), color=c, lw=1.2)
axi.set_xlabel("x [km]")
axi.set_ylabel("thalweg water depth [m]")
axi.set_xlim(20, 45)
axi.set_ylim(bottom=0)
panel_letter(axi, "g")

for ax in (axf, axg, axh, axi):
    clean(ax)

out = OUT_DIR / "fig3_morphology_and_forcing"
fig.savefig(out.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"\nSaved: {out}.png / .pdf")
ds.close()
ds_p.close()