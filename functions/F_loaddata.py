"""Utility functions for loading HIS data and resolving run folders."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

from functions.F_tidalriverdominance import select_max_flood_timestep, select_max_flood_indices_per_cycle

def get_stitched_run_parts(base_path, folder_name, timed_out_dir=None, variability_map=None, analyze_noisy=False):
    """
    Return run-part folders in stitch order: timed-out part first (if found),
    then the main run folder.

    This is shared logic for both HIS and map workflows.
    """
    base_path = Path(base_path)
    folder_path = Path(folder_name)
    folder_label = folder_path.name
    run_folder = folder_path if folder_path.is_absolute() else (base_path / folder_label)
    if timed_out_dir is None:
        timed_out_dir = base_path / "timed-out"
    else:
        timed_out_dir = Path(timed_out_dir)

    parts = []

    if timed_out_dir.exists():
        timed_out_folder = None

        if analyze_noisy:
            match = re.search(r'noisy(\d+)', folder_label)
            if match:
                noisy_id = match.group(0)
                for candidate in timed_out_dir.iterdir():
                    if candidate.is_dir() and noisy_id in candidate.name:
                        timed_out_folder = candidate
                        break
        else:
            scenario_num = folder_label.split('_')[0]
            try:
                scenario_key = str(int(scenario_num))
            except Exception:
                scenario_key = scenario_num

            mf_match = re.search(r"MF(\d+(?:\.\d+)?)", folder_label)
            if mf_match:
                mf_prefix = f"MF{int(float(mf_match.group(1)))}"
                matching = [
                    p for p in timed_out_dir.iterdir()
                    if p.is_dir() and p.name.startswith(mf_prefix + '_')
                ]
                if matching:
                    timed_out_folder = sorted(matching, key=lambda p: p.name)[0]

            if variability_map is not None:
                timed_out_name = variability_map.get(scenario_key, folder_label)
                timed_out_candidate = timed_out_dir / timed_out_name
                if timed_out_folder is None and timed_out_candidate.exists() and timed_out_candidate.is_dir():
                    timed_out_folder = timed_out_candidate

        if timed_out_folder is not None:
            parts.append(timed_out_folder)

    if run_folder.exists() and run_folder.is_dir():
        parts.append(run_folder)

    unique_parts = []
    seen = set()
    for part in parts:
        part_resolved = part.resolve()
        if part_resolved not in seen:
            unique_parts.append(part)
            seen.add(part_resolved)

    return unique_parts


def get_stitched_his_paths(base_path, folder_name, timed_out_dir=None, variability_map=None, analyze_noisy=False):
    """Return stitched HIS file paths (timed-out first, main second if present)."""
    parts = get_stitched_run_parts(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=timed_out_dir,
        variability_map=variability_map,
        analyze_noisy=analyze_noisy,
    )
    paths = []
    for part in parts:
        his_path = part / "output" / "FlowFM_0000_his.nc"
        if his_path.exists():
            paths.append(his_path)
    return paths


def get_stitched_map_run_paths(base_path, folder_name, timed_out_dir=None, variability_map=None, analyze_noisy=False):
    """Return stitched run folders that contain partitioned map output (timed-out first)."""
    parts = get_stitched_run_parts(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=timed_out_dir,
        analyze_noisy=analyze_noisy,
    )
    run_paths = []
    for part in parts:
        output_dir = part / "output"
        if output_dir.exists() and any(output_dir.glob("*_map.nc")):
            run_paths.append(part)
    return run_paths


def select_representative_days(times, n_periods=3):
	"""Select one hydrodynamic day from each period of the simulation."""
	n_total = len(times)
	period_size = n_total / n_periods

	day_duration_seconds = 24 * 3600
	dt = times[1] - times[0]
	dt_seconds = dt / np.timedelta64(1, 's')
	timesteps_per_day = int(np.round(day_duration_seconds / dt_seconds))

	selected_indices = []

	for period in range(n_periods):
		period_start = int(period * period_size)
		day_start = period_start + int(period_size / 2) - timesteps_per_day // 2
		day_start = max(0, min(day_start, n_total - timesteps_per_day))

		day_indices = np.arange(day_start, min(day_start + timesteps_per_day, n_total))
		selected_indices.extend(day_indices)

	return np.array(sorted(set(selected_indices)))


def open_his_dataset(his_paths):
	"""Open a single HIS file as dataset."""
	if isinstance(his_paths, (list, tuple)):
		if len(his_paths) == 1:
			return xr.open_dataset(his_paths[0])
		raise ValueError("Use manual append in load_cross_section_data for multiple HIS files")
	return xr.open_dataset(his_paths)

def load_cross_section_data(his_file_path, q_var='cross_section_discharge',
                            estuary_only=True, km_range=(20, 45),
                            select_cycles_hydrodynamic=True, n_periods=3,
                            select_max_flood=False, flood_sign=-1,
                            select_max_flood_per_cycle=False,
                            exclude_last_timestep=False,
                            exclude_last_n_days=0,
                            selected_time_indices=None,
                            dataset_cache=None):
    """
    Load data from HIS file(s) and extract cross-section information.

    Parameters
    ----------
    q_var : str
        Variable name to extract (e.g. 'cross_section_discharge',
        'cross_section_bedload_sediment_transport').
    dataset_cache : DatasetCache, optional
        A DatasetCache instance for caching datasets. If None, datasets are opened
        without caching and caller is responsible for closing them.
    """
    use_cache = dataset_cache is not None

    if isinstance(his_file_path, (list, tuple)) and len(his_file_path) > 1:
        if use_cache:
            datasets = [dataset_cache.get_xr(p) for p in his_file_path]
            ds_first = datasets[0]
            ds_for_coords = ds_first
        else:
            ds_first = xr.open_dataset(his_file_path[0])
            ds_for_coords = ds_first
            datasets = None
    else:
        ds_first = None
        if use_cache:
            ds_for_coords = dataset_cache.get_xr(
                his_file_path if not isinstance(his_file_path, (list, tuple)) else his_file_path[0]
            )
        else:
            ds_for_coords = open_his_dataset(his_file_path)

    cs_coords = ds_for_coords['cross_section_geom_node_coordx'].values
    cs_count = ds_for_coords['cross_section_geom_node_count'].values

    km_list = []
    idx_list = []
    x_start = 0

    for cs_idx, count in enumerate(cs_count):
        x_coords = cs_coords[x_start:x_start + int(count)]
        if len(x_coords) > 0:
            mean_x = np.mean(x_coords)
            km_pos = mean_x / 1000.0

            if estuary_only:
                if km_range[0] <= km_pos <= km_range[1]:
                    km_list.append(km_pos)
                    idx_list.append(cs_idx)
            else:
                km_list.append(km_pos)
                idx_list.append(cs_idx)
        x_start += int(count)

    if len(km_list) > 0:
        sorted_order = np.argsort(km_list)
        plot_km = np.array(km_list)[sorted_order]
        plot_indices = np.array(idx_list)[sorted_order]
    else:
        raise ValueError("No cross-sections found matching the specified criteria")

    # --- Load variable data across file parts ---
    if isinstance(his_file_path, (list, tuple)) and len(his_file_path) > 1:
        var_list = []
        t_list = []
        if datasets is None:
            datasets = [ds_first] + [xr.open_dataset(p) for p in his_file_path[1:]]
        last_time = None
        last_var_end = None
        for i, ds_part in enumerate(datasets):
            var_part = ds_part[q_var].isel(cross_section=plot_indices)
            t_part = ds_part['time'].values
            # Offset cumulative variables for seamless stitching
            if i > 0 and last_var_end is not None:
                if 'cumulative' in q_var or 'bedload_sediment_transport' in q_var or 'suspended_sediment_transport' in q_var:
                    var_part = var_part + last_var_end
            if last_time is not None and len(t_part) > 1:
                dt = t_part[1] - t_part[0]
                offset = (last_time - t_part[0]) + dt
                t_part = t_part + offset
            var_list.append(var_part)
            t_list.append(t_part)
            last_time = t_part[-1] if len(t_part) else last_time
            if var_part.shape[0] > 0:
                last_var_end = var_part[-1].values
        var_data = xr.concat(var_list, dim='time')
        times = np.concatenate(t_list)
        if not use_cache:
            for ds_part in datasets:
                ds_part.close()
        ds = ds_for_coords
    else:
        ds = ds_for_coords
        var_data = ds[q_var].isel(cross_section=plot_indices)
        times = ds['time'].values

    if exclude_last_timestep and len(times) > 1:
        var_data = var_data.isel(time=slice(0, -1))
        times = times[:-1]

    if exclude_last_n_days and len(times) > 1:
        dt = times[1] - times[0]
        dt_seconds = dt / np.timedelta64(1, 's')
        timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
        drop_steps = int(exclude_last_n_days) * timesteps_per_day
        if drop_steps > 0 and len(times) > drop_steps:
            var_data = var_data.isel(time=slice(0, -drop_steps))
            times = times[:-drop_steps]

    max_flood_km = None
    flood_sign_used = flood_sign

    def _flip_sign(sign):
        return 1 if sign == -1 else -1

    if selected_time_indices is not None:
        selected_time_indices = np.asarray(selected_time_indices, dtype=int)
        var_data = var_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'external'
    elif select_max_flood_per_cycle:
        print("  Selecting max flood timestep for each cycle...")
        selected_time_indices = select_max_flood_indices_per_cycle(times, var_data, plot_km, flood_sign=flood_sign)
        if len(selected_time_indices) == 0:
            alt_sign = _flip_sign(flood_sign)
            alt_indices = select_max_flood_indices_per_cycle(times, var_data, plot_km, flood_sign=alt_sign)
            if len(alt_indices) > 0:
                selected_time_indices = alt_indices
                flood_sign_used = alt_sign
        var_data = var_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'max_flood_per_cycle'
    elif select_max_flood:
        print("  Selecting maximum flood penetration timestep...")
        try:
            t_idx, max_flood_km = select_max_flood_timestep(var_data, plot_km, flood_sign=flood_sign)
        except ValueError:
            alt_sign = _flip_sign(flood_sign)
            t_idx, max_flood_km = select_max_flood_timestep(var_data, plot_km, flood_sign=alt_sign)
            flood_sign_used = alt_sign
        var_data = var_data.isel(time=[t_idx])
        times_selected = np.array([times[t_idx]])
        selected_time_indices = np.array([t_idx])
        n_timesteps_original = len(times)
        selection_mode = 'max_flood'
    elif select_cycles_hydrodynamic:
        print("  Selecting one hydrodynamic day from each period...")
        selected_time_indices = select_representative_days(times, n_periods=n_periods)
        dt = times[1] - times[0]
        dt_seconds = dt / np.timedelta64(1, 's')
        timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
        print(f"  Timesteps per day: {timesteps_per_day}")
        print(f"  Selecting {len(selected_time_indices)} total timesteps (~{len(selected_time_indices) // timesteps_per_day} complete days)")
        var_data = var_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'representative_days'
    else:
        times_selected = times
        selected_time_indices = np.arange(len(times))
        n_timesteps_original = len(times)
        selection_mode = 'all'

    time_hours = (times_selected - times[0]) / np.timedelta64(1, 'h')
    times_datetime = pd.to_datetime(times_selected)

    return {
        'ds': ds,
        q_var: var_data,              # keyed by actual variable name
        'km_positions': plot_km,
        't': times,
        'times': times_selected,
        'times_datetime': times_datetime,
        'time_hours': time_hours,
        'n_timesteps': len(times_selected),
        'n_timesteps_original': n_timesteps_original,
        'selected_indices': selected_time_indices,
        'cross_section_indices': plot_indices,
        'selection_mode': selection_mode,
        'max_flood_km': max_flood_km,
        'flood_sign_used': flood_sign_used,
    }


def load_cross_section_data_from_cache(cache_file, q_var='cross_section_discharge',
                                        select_cycles_hydrodynamic=False, n_periods=3,
                                        select_max_flood=False, flood_sign=-1,
                                        select_max_flood_per_cycle=False,
                                        exclude_last_timestep=False,
                                        exclude_last_n_days=0,
                                        selected_time_indices=None):
    """
    Load cross-section data from a pre-populated .nc cache file
    (as created by load_and_cache_scenario / extract_cache_his.py) and apply
    the same time-selection logic as load_cross_section_data.

    The cache must contain: km_positions (km), t (time), and q_var (time × km).
    Falls back gracefully: callers should check cache_file.exists() and that
    q_var is present before calling.
    """
    with xr.open_dataset(cache_file) as ds:
        km_positions = ds['km_positions'].values
        times_raw = ds['t'].values
        var_numpy = ds[q_var].values  # shape: (time, km)

    plot_km = km_positions
    plot_indices = np.arange(len(km_positions))

    var_data = xr.DataArray(
        var_numpy,
        dims=['time', 'cross_section'],
        coords={'time': times_raw},
    )
    times = times_raw

    if exclude_last_timestep and len(times) > 1:
        var_data = var_data.isel(time=slice(0, -1))
        times = times[:-1]

    if exclude_last_n_days and len(times) > 1:
        dt = times[1] - times[0]
        dt_seconds = dt / np.timedelta64(1, 's')
        timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
        drop_steps = int(exclude_last_n_days) * timesteps_per_day
        if drop_steps > 0 and len(times) > drop_steps:
            var_data = var_data.isel(time=slice(0, -drop_steps))
            times = times[:-drop_steps]

    max_flood_km = None
    flood_sign_used = flood_sign

    def _flip_sign(sign):
        return 1 if sign == -1 else -1

    if selected_time_indices is not None:
        selected_time_indices = np.asarray(selected_time_indices, dtype=int)
        var_data = var_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'external'
    elif select_max_flood_per_cycle:
        print("  Selecting max flood timestep for each cycle (from cache)...")
        selected_time_indices = select_max_flood_indices_per_cycle(
            times, var_data, plot_km, flood_sign=flood_sign)
        if len(selected_time_indices) == 0:
            alt_sign = _flip_sign(flood_sign)
            alt_indices = select_max_flood_indices_per_cycle(
                times, var_data, plot_km, flood_sign=alt_sign)
            if len(alt_indices) > 0:
                selected_time_indices = alt_indices
                flood_sign_used = alt_sign
        var_data = var_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'max_flood_per_cycle'
    elif select_max_flood:
        print("  Selecting maximum flood penetration timestep (from cache)...")
        try:
            t_idx, max_flood_km = select_max_flood_timestep(var_data, plot_km, flood_sign=flood_sign)
        except ValueError:
            alt_sign = _flip_sign(flood_sign)
            t_idx, max_flood_km = select_max_flood_timestep(var_data, plot_km, flood_sign=alt_sign)
            flood_sign_used = alt_sign
        var_data = var_data.isel(time=[t_idx])
        times_selected = np.array([times[t_idx]])
        selected_time_indices = np.array([t_idx])
        n_timesteps_original = len(times)
        selection_mode = 'max_flood'
    elif select_cycles_hydrodynamic:
        print("  Selecting one hydrodynamic day from each period (from cache)...")
        selected_time_indices = select_representative_days(times, n_periods=n_periods)
        dt = times[1] - times[0]
        dt_seconds = dt / np.timedelta64(1, 's')
        timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
        print(f"  Timesteps per day: {timesteps_per_day}")
        print(f"  Selecting {len(selected_time_indices)} total timesteps (~{len(selected_time_indices) // timesteps_per_day} complete days)")
        var_data = var_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'representative_days'
    else:
        times_selected = times
        selected_time_indices = np.arange(len(times))
        n_timesteps_original = len(times)
        selection_mode = 'all'

    time_hours = (times_selected - times[0]) / np.timedelta64(1, 'h')
    times_datetime = pd.to_datetime(times_selected)

    return {
        'ds': None,
        q_var: var_data,
        'km_positions': plot_km,
        't': times,
        'times': times_selected,
        'times_datetime': times_datetime,
        'time_hours': time_hours,
        'n_timesteps': len(times_selected),
        'n_timesteps_original': n_timesteps_original,
        'selected_indices': selected_time_indices,
        'cross_section_indices': plot_indices,
        'selection_mode': selection_mode,
        'max_flood_km': max_flood_km,
        'flood_sign_used': flood_sign_used,
    }



def load_and_cache_scenario(scenario_dir, his_file_paths, cache_file, boxes, var_name):
    """
    Load one scenario variable from cache or HIS files.

    Supports:
    - cross-section variables (dims include: time, cross_section)
    - station/point variables (dims include: time, station)

    Buffer volumes are only computed for selected cumulative cross-section
    transport variables.
    """
    
    BUFFER_VOLUME_VARS = {'cross_section_bedload_sediment_transport'}
    
    compute_buffers = var_name in BUFFER_VOLUME_VARS

    # --- Check if this variable is already cached ---
    var_cached = False
    if cache_file.exists():
        with xr.open_dataset(cache_file) as ds_check:
            var_cached = var_name in ds_check

    if var_cached:
        print(f"Loading '{var_name}' from cache: {cache_file}")
        with xr.open_dataset(cache_file) as ds:
            variable_data = ds[var_name].values
            time = ds['t'].values if 't' in ds else ds['time'].values
            buffer_volumes = {}
            if compute_buffers:
                buffer_volumes = {
                    (box_start, box_end): ds[f'buffer_{var_name}_{int(box_start)}_{int(box_end)}'].values
                    for box_start, box_end in boxes
                    if f'buffer_{var_name}_{int(box_start)}_{int(box_end)}' in ds
                }

            if 'km_positions' in ds:
                return scenario_dir, {
                    'km_positions': ds['km_positions'].values,
                    var_name: variable_data,
                    't': time,
                    'buffer_volumes': buffer_volumes,
                }

            if 'station_name' in ds:
                return scenario_dir, {
                    'station_name': ds['station_name'].values,
                    var_name: variable_data,
                    't': time,
                    'buffer_volumes': {},
                }

            return scenario_dir, {
                var_name: variable_data,
                't': time,
                'buffer_volumes': buffer_volumes,
            }

    # --- Load from HIS files ---
    print(f"Loading '{var_name}' from HIS files: {scenario_dir}")
    with xr.open_dataset(his_file_paths[0]) as ds0:
        if var_name not in ds0:
            raise KeyError(f"Variable '{var_name}' not found in {his_file_paths[0]}")
        var_dims = ds0[var_name].dims

    buffer_volumes = {}

    if 'cross_section' in var_dims:
        data = load_cross_section_data(
            his_file_path=his_file_paths,
            q_var=var_name,
            estuary_only=True,
            km_range=(20, 45),
            select_cycles_hydrodynamic=False,
        )

        km_positions = np.array(data['km_positions'])
        time = data['t']

        print(f"  Reading '{var_name}' data into memory...")
        variable_data = data[var_name].values  # shape: (time, km)

        if 'ds' in data and data['ds'] is not None:
            try:
                data['ds'].close()
            except Exception:
                pass

        # For a box [box_start, box_end] km:
        #   buffer = transport_upstream - transport_downstream
        # i.e. cumulative sediment that entered the box minus what left it.
        if compute_buffers:
            for box_start, box_end in boxes:
                idx_up = np.argmin(np.abs(km_positions - box_start))
                idx_down = np.argmin(np.abs(km_positions - box_end))
                buf = variable_data[:, idx_up] - variable_data[:, idx_down]
                buffer_volumes[(box_start, box_end)] = buf

        ds_add = xr.Dataset(
            {
                var_name: (['time', 'km'], variable_data),
                'km_positions': (['km'], km_positions),
                't': (['time'], time),
                **{
                    f'buffer_{var_name}_{int(box_start)}_{int(box_end)}': (['time'], buf)
                    for (box_start, box_end), buf in buffer_volumes.items()
                },
            },
            coords={'time': time, 'km': km_positions},
        )

        result_payload = {
            'km_positions': km_positions,
            var_name: variable_data,
            't': time,
            'buffer_volumes': buffer_volumes,
        }

    elif 'station' in var_dims:
        with xr.open_dataset(his_file_paths[0]) as ds0:
            if 'station_name' in ds0:
                station_name_raw = ds0['station_name'].values
                if len(station_name_raw) > 0 and isinstance(station_name_raw[0], bytes):
                    station_names = np.array([s.decode('utf-8', errors='ignore').strip() for s in station_name_raw])
                else:
                    station_names = np.array([str(s).strip() for s in station_name_raw])
            else:
                n_station = ds0.sizes.get('station', 0)
                station_names = np.array([f'station_{i}' for i in range(n_station)])

        var_parts = []
        time_parts = []
        last_time = None
        last_var_end = None

        for i, p in enumerate(his_file_paths):
            with xr.open_dataset(p) as ds_part:
                var_part = ds_part[var_name].values
                t_part = ds_part['time'].values

            if i > 0 and last_var_end is not None:
                if 'cumulative' in var_name or 'bedload_sediment_transport' in var_name or 'suspended_sediment_transport' in var_name:
                    var_part = var_part + last_var_end

            if last_time is not None and len(t_part) > 1:
                dt = t_part[1] - t_part[0]
                offset = (last_time - t_part[0]) + dt
                t_part = t_part + offset

            var_parts.append(var_part)
            time_parts.append(t_part)
            if len(t_part) > 0:
                last_time = t_part[-1]
                last_var_end = var_part[-1]

        variable_data = np.concatenate(var_parts, axis=0)
        time = np.concatenate(time_parts)

        ds_add = xr.Dataset(
            {
                var_name: (['time', 'station'], variable_data),
                'station_name': (['station'], station_names),
                't': (['time'], time),
            },
            coords={'time': time, 'station': np.arange(len(station_names))},
        )

        result_payload = {
            'station_name': station_names,
            var_name: variable_data,
            't': time,
            'buffer_volumes': {},
        }

    else:
        raise ValueError(
            f"Unsupported dims for '{var_name}': {var_dims}. "
            "Expected a variable with 'cross_section' or 'station' dimension."
        )

    # --- Append to (or create) cache ---
    if cache_file.exists():
        with xr.open_dataset(cache_file) as ds_existing:
            ds_existing = ds_existing.load()

        ds_new = ds_existing
        for coord_name, coord in ds_add.coords.items():
            if coord_name not in ds_new.coords:
                ds_new = ds_new.assign_coords({coord_name: coord})

        for var in ds_add.data_vars:
            # Keep existing core axes/coords stable for backward compatibility.
            if var in {'t', 'km_positions'} and var in ds_new:
                continue
            ds_new[var] = ds_add[var]
    else:
        ds_new = ds_add

    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in ds_new.data_vars}
    ds_new.to_netcdf(cache_file, encoding=encoding)
    print(f"  Saved/updated cache: {cache_file}")

    return scenario_dir, result_payload
