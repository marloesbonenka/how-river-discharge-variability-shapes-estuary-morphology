"""Dynamic equilibrium from the his-file sediment budget between two cross-sections.

Dynamic equilibrium: the net sediment export at the seaward section equals the net import
plus the river supply, i.e. net export (seaward section) = river supply (landward section),
so there is no net storage in between.

Source: the his file, NOT the map file. D-Flow FM stores per observation cross-section the
CUMULATIVE bed-load transport (cross_section_bedload_sediment_transport, kg, hourly). It is
accumulated on the flow links that update the bed and already includes MorFac, so a flux
over any interval is the difference of two stored numbers (no map-based reconstruction).
Sign: positive = in the positive section direction, which here is seaward (the river
discharge through km45 is +Q).
"""

import numpy as np
import xarray as xr

SED_VAR = "cross_section_bedload_sediment_transport"   # cumulative [kg], x MorFac
Q_VAR = "cross_section_discharge"                      # [m3/s]


def section_name(km):
    """His cross-section name for an estuary km, or a full name passed through."""
    return km if isinstance(km, str) else f"ObservationCrossSection_Estuary_km{km}"


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


def read_his_sections(his_path, kms, var):
    """Time (datetime64) and {km: series of `var`} for the given cross-sections."""
    ds = xr.open_dataset(his_path)
    try:
        names = _decode(ds["cross_section_name"].values)
        data = ds[var].transpose("time", "cross_section").values
        out = {}
        for km in kms:
            name = section_name(km)
            if name not in names:
                raise ValueError(f"Cross-section '{name}' not in {his_path}")
            out[km] = data[:, names.index(name)]
        return np.asarray(ds.time.values).astype("datetime64[ns]"), out
    finally:
        ds.close()


def tidal_cycle_budget(his_path, sea_km, river_km, morfac, cdryb, extra_kms=()):
    """
    Sediment budget per tidal cycle. A cycle runs from one ebb onset (upward zero crossing
    of the discharge at sea_km, linearly interpolated between the hourly outputs) to the
    next; the cumulative transports are interpolated to those moments and differenced.
    Returns a dict with the morphological years of the cycle centres ('morph_yr'), the
    cycle length [h] ('period_h') and per section km the transport in m3 of bed per
    PHYSICAL tidal cycle (MorFac removed): river supply = [river_km], net export = [sea_km].
    """
    kms = (sea_km, river_km, *extra_kms)
    t, cum = read_his_sections(his_path, kms, SED_VAR)
    _, q = read_his_sections(his_path, [sea_km], Q_VAR)
    th = (t - t[0]) / np.timedelta64(1, "h")
    q_sea = q[sea_km]
    up = np.where((q_sea[:-1] < 0) & (q_sea[1:] >= 0))[0]
    tc = th[up] - q_sea[up] * (th[up + 1] - th[up]) / (q_sea[up + 1] - q_sea[up])
    out = {km: np.diff(np.interp(tc, th, cum[km])) / (morfac * cdryb) for km in kms}
    out["morph_yr"] = 0.5 * (tc[1:] + tc[:-1]) / 24 * morfac / 365.25
    out["period_h"] = np.diff(tc)
    return out


def yearly_budget(his_path, sea_km, river_km, morfac, cdryb, extra_kms=()):
    """
    Sediment budget per calendar year of the run (= one hydrograph year: the hydrographs
    repeat every year, peak mid-year). Returns a dict with the start / end morphological
    year of each year ('yr0', 'yr1') and per section km the transport in m3 of bed per
    hydrograph year INCLUDING MorFac (= per `morfac` morphological years).
    """
    kms = (sea_km, river_km, *extra_kms)
    t, cum = read_his_sections(his_path, kms, SED_VAR)
    years = np.arange(t[0].astype("datetime64[Y]").astype(int),
                      t[-1].astype("datetime64[Y]").astype(int) + 1) + 1970
    edges = np.array([np.datetime64(f"{y}-01-01", "ns") for y in years])
    edges = edges[(edges >= t[0]) & (edges <= t[-1])]
    th = (t - t[0]) / np.timedelta64(1, "h")
    eh = (edges - t[0]) / np.timedelta64(1, "h")
    out = {km: np.diff(np.interp(eh, th, cum[km])) / cdryb for km in kms}
    morph = eh / 24 * morfac / 365.25
    out["yr0"], out["yr1"] = morph[:-1], morph[1:]
    return out


def running_mean(a, n):
    """Centred running mean over n samples (NaN where the window is incomplete)."""
    out = np.full(len(a), np.nan)
    out[n // 2: n // 2 + len(a) - n + 1] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def equilibrium_time(morph_yr, ratio, tol):
    """
    First time after which |ratio - 1| < tol holds until the end of the record
    (ratio = net export / river supply); None if the last valid value is outside.
    """
    valid = np.where(np.isfinite(ratio))[0]
    outside = valid[np.abs(ratio[valid] - 1) >= tol]
    if outside.size == 0:
        return morph_yr[valid[0]]
    if outside[-1] < valid[-1]:
        return morph_yr[outside[-1] + 1]
    return None
