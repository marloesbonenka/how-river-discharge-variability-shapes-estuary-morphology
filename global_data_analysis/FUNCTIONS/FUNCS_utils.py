"""Utility functions for data analysis in estuary project."""
#%% --- IMPORTS ---
import numpy as np
import h5py
from datetime import datetime, timedelta
import rasterio

#%% Define utility functions

def transform_coordinates(lon, lat, return_indices=False, lon_convention='180'):
    """
    Transforms geographic coordinates to grid coordinates or indices.
    
    Parameters:
        lon (float): Longitude in decimal degrees
        lat (float): Latitude in decimal degrees, range [-90, 90]
        return_indices (bool): If True, return row/col indices; if False, return rm coordinates
        lon_convention (str): '180' if lon is in [-180, 180] (default)
                              '360' if lon is in [0, 360] (e.g. GlobalDeltaData MouthLon)
    """
    # Normalize to -180:180 if needed
    if lon_convention == '360':
        lon = np.where(lon > 180, lon - 360, lon)

    # Calculate rm coordinates
    rm_lon_val = (lon + 180) * 10
    rm_lat_val = (lat + 90) * 10
    
    if return_indices:
        row = 1800 - int(rm_lat_val)
        col = int(rm_lon_val)
        return row, col
    else:
        return rm_lon_val, rm_lat_val

def matlab_datenum_to_datetime(datenum):
    """
    Converts Matlab datenum into Python datetime
    
    Parameters:
        datenum (float): Matlab datenum value
        
    Returns:
        datetime: Corresponding Python datetime object
    """
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

def load_basin_area_lookup(global_delta_path):
    """
    Loads GlobalDeltaData.mat and builds a lookup of basin areas in rm-space,
    matching the coordinate system used by qs_timeseries.

    Parameters:
        global_delta_path (str or Path): Path to GlobalDeltaData.mat

    Returns:
        tuple: (gd_rm_lon, gd_rm_lat, gd_basin_area) as numpy arrays
               All in rm grid coordinate space for direct comparison with qs rm_lon/rm_lat.
    """
    with h5py.File(global_delta_path, 'r') as f:
        gd_mouth_lat  = np.array(f['MouthLat']).flatten()
        gd_mouth_lon  = np.array(f['MouthLon']).flatten()   # 0-360 convention
        gd_basin_area = np.array(f['BasinArea']).flatten()  # km²

    # Convert MouthLon (0-360) and MouthLat to rm grid coordinates
    gd_rm_lon, gd_rm_lat = transform_coordinates(gd_mouth_lon, gd_mouth_lat, lon_convention='360')

    return gd_rm_lon, gd_rm_lat, gd_basin_area


def load_basin_area_raster(tif_path):
    """
    Loads basin area from bqart_a.tif raster (used as fallback or cross-check).

    Parameters:
        tif_path (str or Path): Path to bqart_a.tif

    Returns:
        np.ndarray: 2D basin area array (flipped, NaN-cleaned)
    """
    with rasterio.open(tif_path) as src:
        basin_area = np.flipud(src.read(1)).astype(float)
        basin_area[basin_area < 0] = np.nan
    return basin_area

def load_data(mat_file_path, tif_path=None, global_delta_path=None):
    """
    Load all necessary data from files.
    
    Parameters:
        mat_file_path (str): Path to the .mat file containing estuary data
        tif_path (str, optional): Path to bqart_a.tif for basin area raster
        global_delta_path (str, optional): Path to GlobalDeltaData.mat for basin area lookup
        
    Returns:
        dict: Dictionary containing all required data
    """
    data_cache = {}
    
    # Load mat file data
    with h5py.File(mat_file_path, 'r') as d:
        data_cache['rm_lat']          = np.array(d['rm_lat']).flatten()
        data_cache['rm_lon']          = np.array(d['rm_lon']).flatten()
        data_cache['lon_grid']        = np.array(d['lon_grid']).flatten()
        data_cache['lat_grid']        = np.array(d['lat_grid']).flatten()
        data_cache['discharge_series'] = np.array(d['discharge_series'])
        data_cache['sed_series']       = np.array(d['sed_series'])
        data_cache['t']               = np.array(d['t']).flatten()
    
    # Load basin area from TIF raster if provided
    if tif_path is not None:
        data_cache['basin_area_raster'] = load_basin_area_raster(tif_path)
        print("Basin area raster (bqart_a.tif) loaded.")

    # Load GlobalDeltaData basin area lookup if provided
    if global_delta_path is not None:
        gd_rm_lon, gd_rm_lat, gd_basin_area = load_basin_area_lookup(global_delta_path)
        data_cache['gd_rm_lon']      = gd_rm_lon
        data_cache['gd_rm_lat']      = gd_rm_lat
        data_cache['gd_basin_area']  = gd_basin_area
        print(f"GlobalDeltaData basin areas loaded: {len(gd_basin_area)} deltas.")

    # Create datetime objects from MATLAB date numbers
    data_cache['datetimes'] = [matlab_datenum_to_datetime(dn) for dn in data_cache['t']]
    
    print("Data loaded successfully.")
    return data_cache
