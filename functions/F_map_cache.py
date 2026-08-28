"""NetCDF-based cache helpers for map output (no pickle)."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
import xarray as xr
import xugrid as xu
import dfm_tools as dfmt


def _is_ugrid_dataset(ds) -> bool:
    return hasattr(ds, 'obj') and hasattr(ds, 'grids')


def _merge_preserve_ugrid(parts, compat='no_conflicts', join='outer'):
    if not parts:
        return None

    ugrid_ref = next((p for p in parts if _is_ugrid_dataset(p)), None)
    xr_parts = [p.obj if _is_ugrid_dataset(p) else p for p in parts]
    merged = xr.merge(xr_parts, compat=compat, join=join)

    if ugrid_ref is not None:
        return xu.UgridDataset(obj=merged, grids=ugrid_ref.grids)
    return merged


def cache_tag_from_bbox(bbox, tag_override=None) -> str:
    if tag_override:
        return str(tag_override)
    if bbox is None:
        return "full"
    return f"bbox_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"


def _folder_label(folder_name) -> str:
    try:
        return Path(folder_name).name
    except Exception:
        return str(folder_name)


def get_cache_path(cache_dir: Path, folder_name: str, var_name: str, tag: str | None = None) -> Path:
    tag_suffix = f"_{tag}" if tag and tag != "full" else ""
    folder_label = _folder_label(folder_name)
    return cache_dir / f"mapoutput_{var_name}_{folder_label}{tag_suffix}.nc"


def _write_dataset_atomic(ds_out, cache_path: Path, encoding: dict, ds_existing=None):
    """Write to a temp file first, then replace target to avoid Windows file-lock issues."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{uuid4().hex}.tmp{cache_path.suffix}")
    try:
        try:
            if hasattr(ds_out, 'ugrid'):
                ds_out.ugrid.to_netcdf(tmp_path, encoding=encoding)
            else:
                ds_out.to_netcdf(tmp_path, encoding=encoding)
        except RuntimeError as exc:
            if 'invalid argument' not in str(exc).lower():
                raise
            print(f"[map-cache]     retry after in-memory load ({cache_path.name})")
            if _is_ugrid_dataset(ds_out):
                ds_retry = xu.UgridDataset(obj=ds_out.obj.load(), grids=ds_out.grids)
                ds_retry.ugrid.to_netcdf(tmp_path, encoding=encoding)
            else:
                ds_out.load().to_netcdf(tmp_path, encoding=encoding)

        if ds_existing is not None:
            ds_existing.close()

        tmp_path.replace(cache_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def select_cache_path(cache_dir: Path, folder_name: str, var_name: str, tag: str | None = None) -> Path:
    folder_label = _folder_label(folder_name)
    tagged = get_cache_path(cache_dir, folder_name, var_name, tag)
    legacy = cache_dir / f"assessment_multi_{folder_label}.nc"
    if tagged.exists():
        return tagged
    if legacy.exists():
        return legacy
    return tagged


def _expand_run_path_files(run_paths: Iterable) -> list[Path]:
    files: list[Path] = []
    for run_path in run_paths:
        file_pattern = _to_file_pattern(run_path)
        matches = [Path(p) for p in glob.glob(file_pattern)]
        files.extend([p for p in matches if p.is_file()])
    return files


def _cache_is_fresh(cache_paths: list[Path], run_paths: Iterable) -> bool:
    if not cache_paths or any(not p.exists() for p in cache_paths):
        return False

    source_files = _expand_run_path_files(run_paths)
    if not source_files:
        # If sources are unavailable but caches exist, prefer cache reuse.
        return True

    newest_source_mtime = max(p.stat().st_mtime_ns for p in source_files)
    oldest_cache_mtime = min(p.stat().st_mtime_ns for p in cache_paths)
    return oldest_cache_mtime >= newest_source_mtime


def _target_dates_signature(target_dates) -> str:
    if target_dates is None:
        return "all"
    target_ns = [str(np.datetime64(target_dt, 'ns').astype('int64')) for target_dt in target_dates]
    return ",".join(target_ns)


def _cache_has_target_dates(cache_paths: list[Path], target_dates) -> bool:
    if not cache_paths or any(not p.exists() for p in cache_paths):
        return False

    # If no specific dates requested, accept whatever is in the cache.
    if target_dates is None:
        return True

    expected_signature = _target_dates_signature(target_dates)
    for cache_path in cache_paths:
        ds_cache = xu.open_dataset(cache_path)
        try:
            actual_signature = ds_cache.attrs.get('target_dates_signature')
            if actual_signature != expected_signature:
                return False
        finally:
            ds_cache.close()
    return True
     
def load_or_update_map_cache_multi(
    cache_dir: Path,
    folder_name: str,
    run_paths: Iterable,
    var_names,
    bbox=None,
    append_time=True,
    append_vars=True,
    chunks=None,
    cache_tag=None,
    target_dates=None,
):
    run_paths = list(run_paths)

    def _select_var_with_topology(ds: xu.UgridDataset, selected: list[str]) -> xu.UgridDataset:
        keep = set(selected)
        if 'time' in ds.variables:
            keep.add('time')
        for name, var in ds.variables.items():
            if getattr(var, 'attrs', {}).get('cf_role') == 'mesh_topology':
                keep.add(name)
                mesh_attrs = getattr(var, 'attrs', {})
                for attr_name in (
                    'node_coordinates',
                    'face_node_connectivity',
                    'edge_node_connectivity',
                    'face_coordinates',
                    'edge_coordinates',
                    'boundary_node_connectivity',
                    'face_face_connectivity',
                ):
                    ref = mesh_attrs.get(attr_name)
                    if isinstance(ref, str):
                        keep.update(ref.split())
        for name in ds.variables.keys():
            if 'domain' in name.lower():
                keep.add(name)
        keep_existing = [v for v in keep if v in ds.variables]
        return ds[keep_existing]

    print(f"[map-cache] Scenario: {folder_name}")

    cache_paths = [select_cache_path(cache_dir, folder_name, var_name, cache_tag) for var_name in var_names]
    if _cache_is_fresh(cache_paths, run_paths) and _cache_has_target_dates(cache_paths, target_dates):
        print("[map-cache]   cache is up-to-date, skipping map-partition reload")
        datasets = [xu.open_dataset(cache_path) for cache_path in cache_paths]
        if len(datasets) == 1:
            return datasets[0]
        return _merge_preserve_ugrid(datasets, compat='no_conflicts', join='outer')

    ds_all = build_masked_map_dataset(run_paths, var_names, bbox=bbox, chunks=chunks, target_dates=target_dates)
    if ds_all is None:
        return None

    datasets = []
    for var_name in var_names:
        print(f"[map-cache]   cache var: {var_name}")
        ds_var = _select_var_with_topology(ds_all, [var_name])
        cache_path = select_cache_path(cache_dir, folder_name, var_name, cache_tag)

        ds_existing = None
        if cache_path.exists():
            print(f"[map-cache]     existing: {cache_path.name}")
            ds_existing = xu.open_dataset(cache_path)

        ds_out = _merge_for_cache(ds_existing, ds_var, append_time, append_vars)

        if ds_out is None:
            print(f"[map-cache]     no new data for {cache_path.name}")
            if ds_existing is not None:
                ds_existing.close()
            continue

        if cache_tag is not None:
            ds_out.attrs['cache_tag'] = str(cache_tag)
        ds_out.attrs['cache_bbox'] = "full" if bbox is None else str(list(bbox))
        ds_out.attrs['target_dates_signature'] = _target_dates_signature(target_dates)

        comp = dict(zlib=True, complevel=4)
        encoding = {v: comp for v in ds_out.data_vars}
        should_write = (not cache_path.exists()) or (ds_out is not ds_existing)
        if should_write:
            print(f"[map-cache]     write: {cache_path.name}")
            _write_dataset_atomic(ds_out, cache_path, encoding, ds_existing=ds_existing)
        else:
            print(f"[map-cache]     no write needed: {cache_path.name}")
            if ds_existing is not None:
                ds_existing.close()

        datasets.append(ds_out)

    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return _merge_preserve_ugrid(datasets, compat='no_conflicts', join='outer')


def _filter_new_times(ds_new: xu.UgridDataset, ds_existing: xu.UgridDataset):
    if 'time' not in ds_new.dims or 'time' not in ds_existing.dims:
        return ds_new
    existing_times = ds_existing['time'].values
    new_times = ds_new['time'].values
    keep_mask = ~np.isin(new_times, existing_times)
    if keep_mask.any():
        return ds_new.isel(time=keep_mask)
    return None


def _merge_for_cache(ds_existing, ds_new, append_time: bool, append_vars: bool):
    if ds_existing is None:
        return ds_new
    if not append_time and not append_vars:
        return ds_new

    if not append_time and append_vars:
        new_only_vars = [v for v in ds_new.data_vars if v not in ds_existing.data_vars]
        if not new_only_vars:
            return ds_existing
        ds_new_only = ds_new[new_only_vars]
        if 'time' in ds_existing.dims and 'time' in ds_new_only.dims:
            ds_new_only = ds_new_only.reindex(time=ds_existing['time'])
        return _merge_preserve_ugrid([ds_existing, ds_new_only], compat='no_conflicts', join='outer')

    shared_vars = [v for v in ds_new.data_vars if v in ds_existing.data_vars]
    new_only_vars = [v for v in ds_new.data_vars if v not in ds_existing.data_vars] if append_vars else []
    existing_only_vars = [v for v in ds_existing.data_vars if v not in ds_new.data_vars]

    ds_new_filtered = _filter_new_times(ds_new, ds_existing)
    if ds_new_filtered is None and not new_only_vars:
        return ds_existing

    parts = []
    if existing_only_vars:
        parts.append(ds_existing[existing_only_vars])

    if shared_vars:
        if ds_new_filtered is not None:
            shared_concat = xu.concat([ds_existing[shared_vars], ds_new_filtered[shared_vars]], dim='time')
        else:
            shared_concat = ds_existing[shared_vars]
        parts.append(shared_concat)

    if new_only_vars:
        parts.append(ds_new[new_only_vars])

    if not parts:
        return ds_existing

    return _merge_preserve_ugrid(parts, compat='no_conflicts', join='outer')


def _to_file_pattern(path_like) -> str:
    path_str = str(path_like)
    if '*' in path_str or path_str.endswith('.nc'):
        return path_str
    return str(Path(path_like) / "output" / "*_map.nc")


def build_masked_map_dataset(run_paths: Iterable, var_names, bbox=None, chunks=None, target_dates=None):
    def _keep_topology_and_selected(ds: xr.Dataset) -> xr.Dataset:
        if var_names is None:
            return ds

        keep = set(var_names)
        if 'time' in ds.variables:
            keep.add('time')

        for name, var in ds.variables.items():
            if getattr(var, 'attrs', {}).get('cf_role') == 'mesh_topology':
                keep.add(name)
                mesh_attrs = getattr(var, 'attrs', {})
                for attr_name in (
                    'node_coordinates',
                    'face_node_connectivity',
                    'edge_node_connectivity',
                    'face_coordinates',
                    'edge_coordinates',
                    'boundary_node_connectivity',
                    'face_face_connectivity',
                ):
                    ref = mesh_attrs.get(attr_name)
                    if isinstance(ref, str):
                        keep.update(ref.split())

        for name in ds.variables.keys():
            if 'domain' in name.lower():
                keep.add(name)

        keep_existing = [v for v in keep if v in ds.variables]
        return ds[keep_existing]

    datasets = []
    run_paths = list(run_paths)
    print(f"[map-cache] Loading map parts: {len(run_paths)}")
    for run_path in run_paths:
        file_pattern = _to_file_pattern(run_path)
        print(f"[map-cache]   open: {file_pattern}")
        part_ds = dfmt.open_partitioned_dataset(
            file_pattern,
            chunks=chunks or {'time': 100},
            preprocess=_keep_topology_and_selected
        )
        datasets.append(part_ds)
    if not datasets:
        return None

    print("[map-cache]   concatenate parts")
    full_ds = xu.concat(datasets, dim="time")
    if bbox is not None:
        print(f"[map-cache]   apply bbox: {bbox}")
        full_ds = full_ds.ugrid.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))

    if target_dates is not None and 'time' in full_ds.dims:
        print(f"[map-cache]   selecting {len(target_dates)} snapshot timesteps")
        time_values = full_ds['time'].values
        time_ns = np.array(time_values, dtype='datetime64[ns]').astype('int64')
        idx_to_target = {}
        for target_dt in target_dates:
            target_ns = np.datetime64(target_dt, 'ns').astype('int64')
            idx = int(np.argmin(np.abs(time_ns - target_ns)))
            idx_to_target.setdefault(idx, target_dt)

        unique_indices = sorted(idx_to_target.keys())

        full_ds = full_ds.isel(time=unique_indices)
        actual_times = full_ds['time'].values
        for idx, t_actual in zip(unique_indices, actual_times):
            t_target = idx_to_target.get(idx)
            print(f"[map-cache]     target {np.datetime_as_string(np.datetime64(t_target, 'D'), unit='D')} "
                  f"-> actual {np.datetime_as_string(np.datetime64(t_actual, 'D'), unit='D')}")

    return full_ds


def load_or_update_map_cache(
    cache_path: Path,
    run_paths: Iterable,
    var_names,
    bbox=None,
    append_time=True,
    append_vars=True,
    chunks=None,
    cache_tag=None,
):
    run_paths = list(run_paths)

    if cache_path.exists() and _cache_is_fresh([cache_path], run_paths):
        return xu.open_dataset(cache_path)

    ds_existing = None
    if cache_path.exists():
        ds_existing = xu.open_dataset(cache_path)

    ds_new = build_masked_map_dataset(run_paths, var_names, bbox=bbox, chunks=chunks)
    if ds_new is None:
        if ds_existing is not None:
            return ds_existing
        return None

    ds_out = _merge_for_cache(ds_existing, ds_new, append_time, append_vars)

    if ds_out is None:
        if ds_existing is not None:
            ds_existing.close()
        return None

    if cache_tag is not None:
        ds_out.attrs['cache_tag'] = str(cache_tag)
    ds_out.attrs['cache_bbox'] = "full" if bbox is None else str(list(bbox))

    should_write = (not cache_path.exists()) or (ds_out is not ds_existing)
    if should_write:
        comp = dict(zlib=True, complevel=4)
        encoding = {v: comp for v in ds_out.data_vars}
        _write_dataset_atomic(ds_out, cache_path, encoding, ds_existing=ds_existing)
    elif ds_existing is not None:
        ds_existing.close()

    return ds_out


# Extract face-center coordinates from a cached xugrid dataset across version variants.
def _get_face_coords(ds):
    """Robustly extract face_x and face_y from a xugrid UgridDataset.
    
    Tries multiple access patterns in order of preference to handle
    differences across xugrid versions:
      1. ds.grids[0].face_x / face_y          (xugrid >= 0.9)
      2. ds.grid.face_x / face_y              (some older versions)
      3. ds.coords['mesh2d_face_x'] etc.      (always registered as coords)
    """
    # Option 1: ds.grids[0].face_x / face_y (most common in recent xugrid)
    try:
        return (
            np.asarray(ds.grids[0].face_x),
            np.asarray(ds.grids[0].face_y),
        )
    except Exception:
        pass

    # Option 2: ds.grid.face_x / face_y
    try:
        return (
            np.asarray(ds.grid.face_x),
            np.asarray(ds.grid.face_y),
        )
    except Exception:
        pass

    # Option 3: coordinates registered automatically by xugrid
    try:
        return (
            np.asarray(ds.coords['mesh2d_face_x']),
            np.asarray(ds.coords['mesh2d_face_y']),
        )
    except Exception:
        pass

    raise RuntimeError(
        "Could not extract face_x / face_y from the xugrid dataset. "
        "Check that face_coordinates are preserved in the cache topology."
    )
