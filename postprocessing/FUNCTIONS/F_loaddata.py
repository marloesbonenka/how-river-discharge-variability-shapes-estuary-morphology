"""Utility functions for loading HIS data and resolving run folders."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import time as timer

from FUNCTIONS.F_tidalriverdominance import *
from FUNCTIONS.F_cache import *

def _cache_file_for_variable(cache_dir, scenario_num, run_id, his_file_paths, var_name):
    """
    Keep legacy cross-section cache filenames unchanged.
    Write station variables to dedicated files to avoid mixing schemas.
    """
    with xr.open_dataset(his_file_paths[0]) as ds0:
        if var_name not in ds0:
            raise KeyError(f"Variable '{var_name}' not found in {his_file_paths[0]}")
        dims = ds0[var_name].dims

    if 'station' in dims:
        return cache_dir / f"hisoutput_stations_{int(scenario_num)}_{run_id}.nc"

    # Default and backward-compatible path for cross-section and other legacy vars.
    return cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

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
        variability_map=variability_map,
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


def find_mf_run_folder(base_dir, mf_number):
	"""Find the run folder matching the MF number (e.g., MF50_sens.8778435)."""
	base_dir = Path(base_dir)
	pattern = f"MF{mf_number}*"
	candidates = [p for p in base_dir.glob(pattern) if p.is_dir()]
	mf_regex = re.compile(rf"^MF{re.escape(str(mf_number))}(?!\d)")
	matches = sorted([p for p in candidates if mf_regex.match(p.name)])
	if not matches:
		raise FileNotFoundError(f"No run folder found for pattern '{pattern}' in {base_dir}")
	return matches[0], matches[0].name


def get_his_paths_for_run(base_dir, run_folder):

	"""
	Return list of HIS files to load. If run folder contains 'restart',
	include the timed-out part(s) (if available) before the main run.
	Supports both MF and scenario number-based folder naming.
	"""
	base_dir = Path(base_dir)
	run_folder = Path(run_folder)
	paths = [run_folder / "output" / "FlowFM_0000_his.nc"]
	debug_msgs = []
	
	timed_out_dir = base_dir / "timed-out"
	
	# Try MF logic first
	mf_match = re.search(r"MF(\d+(?:\.\d+)?)", run_folder.name)
	scenario_match = re.match(r"(\d+)[^\d]?", run_folder.name)
	timed_out_candidates = []
	if timed_out_dir.exists():
		if mf_match:
			mf_prefix = f"MF{int(float(mf_match.group(1)))}"
			timed_out_candidates = [p for p in timed_out_dir.rglob("*")
									if p.is_dir() and p.name.startswith(mf_prefix + "_")]
		elif scenario_match:
			# e.g. 3_Q500_rst_flashy.9094053 → 03_run500_flashy
			scenario_num = scenario_match.group(1)
			scenario_num_padded = scenario_num.zfill(2)
			# Find folders like 03_run500_flashy
			timed_out_candidates = [p for p in timed_out_dir.iterdir()
									if p.is_dir() and p.name.startswith(f"{scenario_num_padded}_")]
		matching = sorted(timed_out_candidates, key=lambda p: (p.name, str(p)))
		if matching:
			timed_out_paths = []
			for timed_out_folder in matching:
				timed_out_path = timed_out_folder / "output" / "FlowFM_0000_his.nc"
				if timed_out_path.exists():
					timed_out_paths.append(timed_out_path)
					debug_msgs.append(f"[DEBUG] Timed-out HIS found: {timed_out_path}")
				else:
					debug_msgs.append(f"[DEBUG] Timed-out HIS missing: {timed_out_path}")
			paths = timed_out_paths + paths
		else:
			if "restart" in run_folder.name.lower():
				debug_msgs.append("[DEBUG] Timed-out folder not found for restart run.")
	else:
		if "restart" in run_folder.name.lower():
			debug_msgs.append("[DEBUG] Timed-out dir missing or MF/scenario not found; skipping timed-out search.")

	for msg in debug_msgs:
		print(msg)
	# Return unique existing paths while preserving order
	seen = set()
	unique_paths = []
	for p in paths:
		if p.exists() and p not in seen:
			unique_paths.append(p)
			seen.add(p)
	return unique_paths

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


def load_cross_section_data_old(his_file_path, q_var='cross_section_discharge',
							estuary_only=True, km_range=(20, 45),
							select_cycles_hydrodynamic=True, n_periods=3,
							select_max_flood=False, flood_sign=-1,
							select_max_flood_per_cycle=False,
							exclude_last_timestep=False,
							exclude_last_n_days=0,
							selected_time_indices=None,
							dataset_cache=None):
	"""
	Load discharge data from HIS file(s) and extract cross-section information.
	
	Parameters
	----------
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
			ds_for_coords = dataset_cache.get_xr(his_file_path if not isinstance(his_file_path, (list, tuple)) else his_file_path[0])
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

	if isinstance(his_file_path, (list, tuple)) and len(his_file_path) > 1:
		q_list = []
		t_list = []
		if datasets is None:
			datasets = [ds_first] + [xr.open_dataset(p) for p in his_file_path[1:]]
		last_time = None
		last_q_end = None
		for i, ds_part in enumerate(datasets):
			q_part = ds_part[q_var].isel(cross_section=plot_indices)
			t_part = ds_part['time'].values
			# Offset cumulative variables for seamless stitching
			if i > 0 and last_q_end is not None:
				# Only apply for cumulative variables (assume if 'cumulative' in q_var or user sets flag)
				if 'cumulative' in q_var or 'bedload_sediment_transport' in q_var or 'suspended_sediment_transport' in q_var:
					# Add last value of previous part to all of this part (per cross-section)
					q_part = q_part + last_q_end
			if last_time is not None and len(t_part) > 1:
				dt = t_part[1] - t_part[0]
				offset = (last_time - t_part[0]) + dt
				t_part = t_part + offset
			q_list.append(q_part)
			t_list.append(t_part)
			last_time = t_part[-1] if len(t_part) else last_time
			# Store last value for offsetting next part
			if q_part.shape[0] > 0:
				last_q_end = q_part[-1].values
		q_data = xr.concat(q_list, dim='time')
		times = np.concatenate(t_list)
		if not use_cache:
			for ds_part in datasets:
				ds_part.close()
		ds = ds_for_coords
	else:
		ds = ds_for_coords
		q_data = ds[q_var].isel(cross_section=plot_indices)
		times = ds['time'].values

	if exclude_last_timestep and len(times) > 1:
		q_data = q_data.isel(time=slice(0, -1))
		times = times[:-1]

	if exclude_last_n_days and len(times) > 1:
		dt = times[1] - times[0]
		dt_seconds = dt / np.timedelta64(1, 's')
		timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
		drop_steps = int(exclude_last_n_days) * timesteps_per_day
		if drop_steps > 0 and len(times) > drop_steps:
			q_data = q_data.isel(time=slice(0, -drop_steps))
			times = times[:-drop_steps]

	max_flood_km = None
	flood_sign_used = flood_sign
	def _flip_sign(sign):
		return 1 if sign == -1 else -1
	if selected_time_indices is not None:
		selected_time_indices = np.asarray(selected_time_indices, dtype=int)
		q_data = q_data.isel(time=selected_time_indices)
		times_selected = times[selected_time_indices]
		n_timesteps_original = len(times)
		selection_mode = 'external'
	elif select_max_flood_per_cycle:
		print("  Selecting max flood timestep for each cycle...")
		selected_time_indices = select_max_flood_indices_per_cycle(times, q_data, plot_km, flood_sign=flood_sign)
		if len(selected_time_indices) == 0:
			alt_sign = _flip_sign(flood_sign)
			alt_indices = select_max_flood_indices_per_cycle(times, q_data, plot_km, flood_sign=alt_sign)
			if len(alt_indices) > 0:
				selected_time_indices = alt_indices
				flood_sign_used = alt_sign
		q_data = q_data.isel(time=selected_time_indices)
		times_selected = times[selected_time_indices]
		n_timesteps_original = len(times)
		selection_mode = 'max_flood_per_cycle'
	elif select_max_flood:
		print("  Selecting maximum flood penetration timestep...")
		try:
			t_idx, max_flood_km = select_max_flood_timestep(q_data, plot_km, flood_sign=flood_sign)
		except ValueError:
			alt_sign = _flip_sign(flood_sign)
			t_idx, max_flood_km = select_max_flood_timestep(q_data, plot_km, flood_sign=alt_sign)
			flood_sign_used = alt_sign
		q_data = q_data.isel(time=[t_idx])
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
		q_data = q_data.isel(time=selected_time_indices)
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
		'discharge': q_data,
		'km_positions': plot_km,
		't': times, # raw time > timesteps
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


def read_discharge_from_bc_files(model_path, bc_pattern="*_Qr*_inflow_sinuous.bc"):
    """
    Read and sum discharge from boundary condition (.bc) files.
    
    The BC files contain discharge for individual river cells (01, 02, 03, 04).
    This function sums them to get the total river discharge.
    
    Parameters
    ----------
    model_path : Path
        Path to the model folder containing .bc files
    bc_pattern : str
        Glob pattern to find BC files
        
    Returns
    -------
    times : list of datetime
        Timestamps
    discharges : list of float
        Total discharge values (sum of all cells)
    """
    import re
    from datetime import datetime, timedelta
    
    # Find BC files
    bc_files = list(model_path.glob(bc_pattern))
    if not bc_files:
        # Also check in parent folder
        bc_files = list(model_path.parent.glob(bc_pattern))
    
    if not bc_files:
        print(f"  No BC files found matching {bc_pattern} in {model_path}")
        return None, None
    
    print(f"  Found {len(bc_files)} BC files: {[f.name for f in bc_files]}")
    
    # Parse all BC files
    all_data = {}
    reference_date = None
    
    for bc_file in bc_files:
        with open(bc_file, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        # Parse header to get reference date
        for line in lines:
            if 'seconds since' in line.lower():
                # Extract reference date: "unit = seconds since 2001-01-01 00:00:00"
                match = re.search(r'seconds since\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if match:
                    reference_date = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                break
        
        # Parse data lines (after header)
        data_started = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('[') or '=' in line:
                continue
            
            # Data lines are: "seconds   discharge"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    seconds = float(parts[0])
                    discharge = float(parts[1])
                    
                    if seconds not in all_data:
                        all_data[seconds] = 0.0
                    all_data[seconds] += discharge
                except ValueError:
                    continue
    
    if not all_data or reference_date is None:
        print("  Could not parse BC files.")
        return None, None
    
    # Convert to timestamps and lists
    sorted_seconds = sorted(all_data.keys())
    times = [reference_date + timedelta(seconds=s) for s in sorted_seconds]
    discharges = [all_data[s] for s in sorted_seconds]
    
    print(f"  Read {len(times)} timesteps from BC files (ref: {reference_date})")
    
    return times, discharges


def extract_discharge_at_x(ds, x_target, y_range, time_indices=None):
    """
    Extract total discharge through edges near a given x-coordinate.
    
    Samples mesh2d_q1 (discharge through flow links) at a cross-section.
    Returns the integrated/summed discharge across the width.
    """
    # Get edge coordinates
    if 'mesh2d_edge_x' in ds:
        edge_x = ds['mesh2d_edge_x'].values
        edge_y = ds['mesh2d_edge_y'].values
    else:
        # Fallback: use face coordinates if edge coords not available
        print("  Warning: mesh2d_edge_x not found, using face coordinates.")
        edge_x = ds['mesh2d_face_x'].values
        edge_y = ds['mesh2d_face_y'].values
    
    # Find edges near the target x-coordinate and within y_range
    x_tolerance = 500  # meters tolerance for x-coordinate
    mask = (np.abs(edge_x - x_target) < x_tolerance) & \
           (edge_y >= y_range[0]) & (edge_y <= y_range[1])
    
    edge_indices = np.where(mask)[0]
    
    if len(edge_indices) == 0:
        print(f"  Warning: No edges found near x={x_target}m.")
        return None, None
    
    print(f"  Found {len(edge_indices)} edges near x={x_target/1000:.0f}km for discharge.")
    
    # Extract discharge time series
    if time_indices is None:
        time_indices = range(len(ds.time))
    
    times = []
    discharges = []
    
    for t in tqdm(time_indices, desc="  Extracting discharge", leave=False):
        q_vals = ds['mesh2d_q1'].isel(time=t).values[edge_indices]
        # Sum absolute discharge across the cross-section (accounts for different flow directions)
        total_q = np.nansum(np.abs(q_vals))
        
        times.append(pd.to_datetime(ds.time.values[t]))
        discharges.append(total_q)
    
    return times, discharges


def split_by_hydrodynamic_cycle(times, values, cycle_days=365.25):
    """
    Split a time series into individual hydrodynamic cycles.
    
    Parameters
    ----------
    times : array-like of datetime
        Time stamps
    values : array-like
        Values corresponding to times
    cycle_days : float
        Duration of one hydrodynamic cycle in days
        
    Returns
    -------
    cycles : list of dict
        Each dict contains: {'hydro_day': array, 'values': array, 'cycle_num': int}
    """
    times = pd.to_datetime(times)
    t0 = times[0]
    
    # Calculate hydro days from start
    hydro_days = (times - t0).total_seconds() / 86400
    
    # Determine cycle number for each point
    cycle_nums = (hydro_days // cycle_days).astype(int)
    
    # Split into cycles
    unique_cycles = np.unique(cycle_nums)
    cycles = []
    
    for cycle_num in unique_cycles:
        mask = cycle_nums == cycle_num
        cycle_hydro_days = hydro_days[mask] - (cycle_num * cycle_days)  # Reset to start of cycle
        cycle_values = np.array(values)[mask]
        
        cycles.append({
            'hydro_day': cycle_hydro_days.values,
            'values': cycle_values,
            'cycle_num': int(cycle_num),
        })
    
    return cycles

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

def load_and_cache_scenario_old(scenario_dir, his_file_paths, cache_file, boxes, var_name):
    """Load one scenario from cache or HIS files, compute buffer volumes, save cache."""

    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        ds = xr.open_dataset(cache_file)
        result = {
            'km_positions': ds['km_positions'].values,
            'discharge': ds['discharge'].values,
            't': ds['t'].values,
            'buffer_volumes': {
                (box_start, box_end): ds[f'buffer_{int(box_start)}_{int(box_end)}'].values
                for box_start, box_end in boxes
                if f'buffer_{int(box_start)}_{int(box_end)}' in ds
            }
        }
        ds.close()
        return scenario_dir, result

    # Load from HIS
    print(f"Loading from HIS files: {scenario_dir}")
    data = load_cross_section_data(
        his_file_path=his_file_paths,
        q_var=var_name,
        estuary_only=True,
        km_range=(20, 45),
        select_cycles_hydrodynamic=False
    )

    km_positions = np.array(data['km_positions'])
    time = data['t']

    print(f"  Reading transport data into memory...")

	# t0 = timer.time()

    transport = data['discharge'].values  # triggers actual file read
	
	# print(f"  Data read in {timer.time() - t0:.1f} seconds")

    if 'ds' in data and data['ds'] is not None:
        try:
            data['ds'].close()
        except Exception:
            pass

    # Compute buffer volumes
    buffer_volumes = {}
    for box_start, box_end in boxes:
        idx_up = np.argmin(np.abs(km_positions - box_start))
        idx_down = np.argmin(np.abs(km_positions - box_end))
        buffer_volumes[(box_start, box_end)] = transport[:, idx_up] - transport[:, idx_down]

    # Save to cache
    buffer_dict = {
        f'buffer_{int(box_start)}_{int(box_end)}': (['time'], buf)
        for (box_start, box_end), buf in buffer_volumes.items()
    }
    ds = xr.Dataset({
        'km_positions': (['km'], km_positions),
        'discharge': (['time', 'km'], transport),
        't': (['time'], time),
        **buffer_dict
    })
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(cache_file, encoding=encoding)
    print(f"  Saved cache: {cache_file}")

    return scenario_dir, {
        'km_positions': km_positions,
        'discharge': transport,
        't': time,
        'buffer_volumes': buffer_volumes
    }