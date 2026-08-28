"""Helper functions for tidal range and current speed analyses from HIS files."""

import re

import numpy as np
import xarray as xr

from FUNCTIONS.F_loaddata import (
    load_cross_section_data,
    load_cross_section_data_from_cache,
    load_and_cache_scenario,
)


def decode_name_array(name_array):
    """Return a list of clean strings from a Delft3D-FM name array."""
    arr = np.asarray(name_array)

    def _decode_char_item(item):
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode('utf-8', errors='ignore')
        if isinstance(item, str):
            return item
        if isinstance(item, (np.integer, int)):
            value = int(item)
            if value == 0:
                return ''
            # Some files store names as numeric character codes.
            if 0 <= value <= 255:
                return chr(value)
            return ''
        return str(item)

    if arr.ndim == 2:
        names = []
        for row in arr:
            chars = [_decode_char_item(item) for item in row]
            names.append(''.join(chars).strip())
        return names

    names = []
    for item in arr:
        names.append(_decode_char_item(item).strip())
    return names


def _normalize_station_name(name):
    """Normalize decoded station names for robust regex matching."""
    cleaned = str(name).replace('\x00', '').strip()
    return ''.join(ch for ch in cleaned if ch.isprintable())


def _extract_estuary_km(name, station_pattern):
    """Extract estuary km from a station label using strict, then tolerant matching."""
    m = re.match(station_pattern, name)
    if m:
        return int(m.group(1))

    # Fallback for labels with hidden/trailing characters or slight format drift.
    m = re.search(r'ObservationCrossSection_Estuary_km\s*(\d+)', name)
    if m:
        return int(m.group(1))
    return None


def _select_estuary_stations(station_names, station_pattern):
    """Return sorted (km, idx, label) tuples for estuary stations."""
    selected = []
    for i, raw_name in enumerate(station_names):
        name_norm = _normalize_station_name(raw_name)
        km = _extract_estuary_km(name_norm, station_pattern)
        if km is not None:
            selected.append((km, i, name_norm))
    selected.sort(key=lambda t: t[0])
    return selected


def _build_station_waterlevel_output(station_names, wl, times, station_pattern, exclude_last_timestep):
    """Select estuary stations and return a standardized payload."""
    selected = _select_estuary_stations(station_names, station_pattern)
    if not selected:
        sample_names = [_normalize_station_name(n) for n in station_names[:10]]
        raise ValueError(
            "No estuary stations matched station_pattern. "
            f"pattern={station_pattern!r}; sample_station_names={sample_names}"
        )

    station_km = np.array([t[0] for t in selected], dtype=float)
    station_idx = np.array([t[1] for t in selected], dtype=int)
    station_labels = [t[2] for t in selected]

    wl_sel = wl[:, station_idx]
    times_sel = times

    if exclude_last_timestep and len(times_sel) > 1:
        wl_sel = wl_sel[:-1, :]
        times_sel = times_sel[:-1]

    return {
        'times': times_sel,
        'waterlevel': wl_sel,
        'station_km': station_km,
        'station_labels': station_labels,
    }


def compute_cycle_windows(times, cycle_hours=12.42):
    """Return fixed-length tidal-cycle windows as (start, end) index pairs."""
    if len(times) < 2:
        return []

    dt_hours = (times[1] - times[0]) / np.timedelta64(1, 'h')
    if dt_hours <= 0:
        return []

    steps = int(np.round(float(cycle_hours) / float(dt_hours)))
    if steps < 1:
        steps = 1

    windows = []
    for start in range(0, len(times), steps):
        end = start + steps
        if end <= len(times):
            windows.append((start, end))
    return windows


def load_velocity_from_his_or_cache(cache_file, his_file_paths, velocity_var='cross_section_velocity', exclude_last_timestep=True):
    """Load cross-section velocity with cache fallback to raw HIS files."""
    kwargs = dict(
        q_var=velocity_var,
        select_cycles_hydrodynamic=False,
        exclude_last_timestep=exclude_last_timestep,
    )
    if cache_file.exists():
        try:
            return load_cross_section_data_from_cache(cache_file, **kwargs)
        except Exception as exc:
            print(f"  [cache miss for {velocity_var}] {exc}")
    return load_cross_section_data(his_file_paths, **kwargs)


def load_station_waterlevels(
    his_file_paths,
    waterlevel_var='waterlevel',
    station_pattern=r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$',
    exclude_last_timestep=True,
):
    """Load and stitch station water levels from one or more HIS files."""
    ds0 = xr.open_dataset(his_file_paths[0])
    if 'station_name' not in ds0:
        ds0.close()
        raise KeyError('station_name not found in HIS file')
    if waterlevel_var not in ds0:
        ds0.close()
        raise KeyError(f"{waterlevel_var} not found in HIS file")

    station_names = decode_name_array(ds0['station_name'].values)

    selected = _select_estuary_stations(station_names, station_pattern)

    if not selected:
        ds0.close()
        sample_names = [_normalize_station_name(n) for n in station_names[:10]]
        raise ValueError(
            "No estuary stations matched station_pattern. "
            f"pattern={station_pattern!r}; sample_station_names={sample_names}"
        )

    selected.sort(key=lambda t: t[0])
    station_km = np.array([t[0] for t in selected], dtype=float)
    station_idx = np.array([t[1] for t in selected], dtype=int)
    station_labels = [t[2] for t in selected]

    ds0.close()

    wl_parts = []
    time_parts = []
    last_time = None

    for p in his_file_paths:
        with xr.open_dataset(p) as ds:
            wl = ds[waterlevel_var].isel(station=station_idx)
            tt = ds['time'].values

            if last_time is not None and len(tt) > 1:
                dt = tt[1] - tt[0]
                offset = (last_time - tt[0]) + dt
                tt = tt + offset

            wl_parts.append(wl.values)
            time_parts.append(tt)
            if len(tt) > 0:
                last_time = tt[-1]

    wl_all = np.concatenate(wl_parts, axis=0)
    t_all = np.concatenate(time_parts)

    if exclude_last_timestep and len(t_all) > 1:
        wl_all = wl_all[:-1, :]
        t_all = t_all[:-1]

    return {
        'times': t_all,
        'waterlevel': wl_all,
        'station_km': station_km,
        'station_labels': station_labels,
    }


def load_station_waterlevels_from_cache_or_his(
    cache_file,
    his_file_paths,
    waterlevel_var='waterlevel',
    station_pattern=r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$',
    exclude_last_timestep=True,
):
    """Load station waterlevels from station cache when available, else from HIS+cache update."""
    if cache_file is not None and cache_file.exists():
        with xr.open_dataset(cache_file) as ds:
            if waterlevel_var in ds and 'station_name' in ds:
                station_names = decode_name_array(ds['station_name'].values)
                wl = ds[waterlevel_var].values
                if 't' in ds:
                    times = ds['t'].values
                elif 'time' in ds:
                    times = ds['time'].values
                else:
                    raise KeyError('Neither t nor time found in station cache file')

                return _build_station_waterlevel_output(
                    station_names=station_names,
                    wl=wl,
                    times=times,
                    station_pattern=station_pattern,
                    exclude_last_timestep=exclude_last_timestep,
                )

    # Cache is missing or incomplete: use the same HIS->cache update path as extract_cache_his.
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _, payload = load_and_cache_scenario(
            scenario_dir=str(cache_file.parent),
            his_file_paths=his_file_paths,
            cache_file=cache_file,
            boxes=[],
            var_name=waterlevel_var,
        )
        station_names = decode_name_array(payload.get('station_name', []))
        wl = payload[waterlevel_var]
        times = payload['t']

        return _build_station_waterlevel_output(
            station_names=station_names,
            wl=wl,
            times=times,
            station_pattern=station_pattern,
            exclude_last_timestep=exclude_last_timestep,
        )

    return load_station_waterlevels(
        his_file_paths=his_file_paths,
        waterlevel_var=waterlevel_var,
        station_pattern=station_pattern,
        exclude_last_timestep=exclude_last_timestep,
    )


def cycle_metric(times, matrix, cycle_hours, reducer):
    """Reduce a (time, x) matrix per cycle using reducer(seg)->(x,)."""
    windows = compute_cycle_windows(times, cycle_hours=cycle_hours)
    if not windows:
        return np.array([]), np.empty((0, matrix.shape[1]))

    out_t = []
    out_v = []
    for start, end in windows:
        seg = matrix[start:end, :]
        out_t.append(times[start])
        out_v.append(reducer(seg))

    return np.array(out_t), np.vstack(out_v)


def compute_slope_cm_per_km(wl_seg, km):
    """Linear-fit slope d(eta)/dx in cm/km from a cycle segment."""
    wl_mean = np.nanmean(wl_seg, axis=0)
    valid = np.isfinite(wl_mean) & np.isfinite(km)
    if np.count_nonzero(valid) < 2:
        return np.nan

    x_m = km[valid] * 1000.0
    y_m = wl_mean[valid]
    slope_m_per_m = np.polyfit(x_m, y_m, 1)[0]
    return slope_m_per_m * 100000.0
