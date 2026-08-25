# -*- coding: utf-8 -*-
"""Functions for plotting discharge time series for estuaries based on WBMsed data."""
#%% --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\02_Data_analysis")
from FUNCTIONS.FUNCS_utils import transform_coordinates

# -------------------------------------------------------------------------
# --- AGU figure settings (matches run_estuary_discharge_analysis.py) ---
_MM_TO_IN = 1 / 25.4
_FW = 170 * _MM_TO_IN   # full-width figure  ≈ 6.69 in  (AGU: 50–170 mm)
_CW = 84  * _MM_TO_IN   # single-column      ≈ 3.31 in

_AGU_RCPARAMS = {
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35,
    'ytick.minor.width': 0.35,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}
plt.rcParams.update(_AGU_RCPARAMS)

#%%

def plot_global_delta_distribution(rm_lon, rm_lat, mean_discharge=None, savefig=False, output_dir=None):
    """
    Plot global distribution of deltas using different visualization methods.
    
    Parameters:
        rm_lon (np.ndarray): Array of longitude coordinates
        rm_lat (np.ndarray): Array of latitude coordinates
        mean_discharge (np.ndarray, optional): Mean discharge values for color coding
        savefig (bool): Whether to save figures
        output_dir (str, optional): Directory to save figures
    """
    outdir = Path(output_dir) if output_dir else None

    plt.figure(figsize=(_FW, _FW * 0.5))
    plt.scatter(rm_lon, rm_lat, alpha=0.5, s=5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Global Distribution of Deltas based on Nienhuis et al. 2020')
    plt.grid(True, alpha=0.3)
    if savefig and outdir:
        plt.savefig(outdir / 'global_delta_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(_FW, _FW * 0.5))
    hb = plt.hexbin(rm_lon, rm_lat, gridsize=50, cmap='viridis')
    plt.colorbar(hb, label='Number of Deltas')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Density of Deltas Worldwide based on Nienhuis et al. 2020')
    if savefig and outdir:
        plt.savefig(outdir / 'delta_density_hexbin.png', dpi=300, bbox_inches='tight')
    plt.show()

    if mean_discharge is not None:
        vmin = np.percentile(mean_discharge, 5)
        vmax = np.percentile(mean_discharge, 95)
        plt.figure(figsize=(_FW, _FW * 0.5))
        scatter = plt.scatter(rm_lon, rm_lat, c=mean_discharge, cmap='viridis',
                              alpha=0.7, s=10, edgecolors='none', vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, label='Average Discharge')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Global Distribution of Deltas with Average Discharge Values')
        plt.grid(True, alpha=0.3)
        if savefig and outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / 'delta_discharge_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()


def extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series,
                                  gd_rm_lon=None, gd_rm_lat=None, gd_basin_area=None,
                                  basin_area_raster=None, lat_grid=None, lon_grid=None,
                                  dist_threshold=0.5):
    """
    Extracts discharge and sediment time series for each estuary using the same
    logic as the original MATLAB get_Qriver_timeseries function:

      1. Look up target estuary basin area from GlobalDeltaData 
      2. Find the best-matching WBMsed grid point using a cost function that
         combines geographic distance AND basin area similarity
      3. Compute a scaling factor: fac = WBMsed_basin_area / estuary_basin_area
      4. Scale the extracted discharge and sediment series by fac.

    Parameters:
        estuary_coords (dict): {name: (lat, lon, basin_area_km2)} 
                               basin_area_km2 is optional; if not provided, looked up from GlobalDeltaData.
                               Accepts both (lat, lon) and (lat, lon, basin_area) tuples.
        rm_lon (np.ndarray): WBMsed grid longitude indices (1-based, like MATLAB)
        rm_lat (np.ndarray): WBMsed grid latitude indices (1-based, like MATLAB)
        discharge_series (np.ndarray): Discharge array (n_points x n_timesteps)
        sed_series (np.ndarray): Sediment array (n_points x n_timesteps)
        gd_rm_lon (np.ndarray, optional): GlobalDeltaData mouth longitudes in rm space
        gd_rm_lat (np.ndarray, optional): GlobalDeltaData mouth latitudes in rm space
        gd_basin_area (np.ndarray, optional): GlobalDeltaData basin areas in km²
        basin_area_raster (np.ndarray, optional): 2D basin area array from bqart_a.tif
        lat_grid (np.ndarray, optional): Latitude grid values for converting indices to degrees
        lon_grid (np.ndarray, optional): Longitude grid values for converting indices to degrees
        dist_threshold (float): Maximum distance in degrees to accept GlobalDeltaData match

    Returns:
        tuple: (estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries, estuary_scaling_factors)
    """
    estuary_discharge_timeseries = {}
    estuary_sed_timeseries       = {}
    estuary_rm_coords            = {}
    estuary_scaling_factors      = {}

    has_global_delta = (gd_rm_lon is not None and 
                        gd_rm_lat is not None and 
                        gd_basin_area is not None)
    
    has_raster = (basin_area_raster is not None and 
                  lat_grid is not None and 
                  lon_grid is not None)

    # Pre-compute WBM basin areas from TIF raster (like MATLAB: wbm_basinarea = a(sub2ind(...)))
    if has_raster:
        # rm_lat and rm_lon are 1-based indices; Python uses 0-based, so subtract 1
        wbm_basin_areas = basin_area_raster[rm_lat.astype(int) - 1, rm_lon.astype(int) - 1]
        # Get actual lat/lon in degrees for each WBM point
        wbm_lat_deg = lat_grid[rm_lat.astype(int) - 1]
        wbm_lon_deg = lon_grid[rm_lon.astype(int) - 1]
    else:
        wbm_basin_areas = None
        wbm_lat_deg = None
        wbm_lon_deg = None

    # Convert GlobalDeltaData from rm space to degrees for distance calculation
    if has_global_delta:
        gd_lon_deg = (gd_rm_lon / 10) - 180
        gd_lat_deg = (gd_rm_lat / 10) - 90

    for estuary, coords in estuary_coords.items():
        # Unpack coords — support both (lat, lon) and (lat, lon, basin_area)
        if len(coords) == 3:
            lat, lon, estuary_basin_area = coords
        else:
            lat, lon = coords
            estuary_basin_area = None

        # --- Step 1: Look up basin area from GlobalDeltaData (if not provided) ---
        if estuary_basin_area is None and has_global_delta:
            # Adjust search longitude to match GlobalDeltaData convention (0-360)
            search_lon = lon + 360 if lon < 0 else lon
            
            # Find closest delta in GlobalDeltaData (in degree space)
            gd_search_lon = gd_lon_deg + 360  # Convert to 0-360 for comparison
            gd_search_lon = np.where(gd_search_lon > 360, gd_search_lon - 360, gd_search_lon)
            
            gd_dist = np.sqrt((gd_search_lon - search_lon)**2 + (gd_lat_deg - lat)**2)
            nearest_gd = np.argmin(gd_dist)
            
            if gd_dist[nearest_gd] <= dist_threshold:
                estuary_basin_area = gd_basin_area[nearest_gd]
                print(f"  {estuary}: basin area looked up from GlobalDeltaData = "
                      f"{estuary_basin_area:.1f} km² (dist={gd_dist[nearest_gd]:.3f} deg)")
            else:
                print(f"  {estuary}: WARNING - no GlobalDeltaData match within {dist_threshold} deg "
                      f"(closest: {gd_dist[nearest_gd]:.2f} deg). No basin area scaling applied.")

        # --- Step 2: Find best WBMsed grid point using MATLAB cost function ---
        if estuary_basin_area is not None and estuary_basin_area > 0 and has_raster:
            # MATLAB: delta_coor = rm_lat + 1i * rm_lon (using actual lat/lon degrees)
            # MATLAB: wbm_coor = lat_grid(wbm_lat) + 1i * lon_grid(wbm_lon)
            # Geographic distance in degrees (abs of complex difference)
            geo_dist = np.sqrt((wbm_lat_deg - lat)**2 + (wbm_lon_deg - lon)**2)
            
            # MATLAB cost function:
            # blub = abs(wbm_coor - delta_coor) + 
            #        abs((wbm_basinarea - basinarea) / basinarea * log10(basinarea)) +
            #        (abs(wbm_coor - delta_coor) > 8) * 10
            basin_term = np.abs((wbm_basin_areas - estuary_basin_area) / 
                               estuary_basin_area * 
                               np.log10(estuary_basin_area))
            # Handle NaN basin areas from raster
            basin_term = np.nan_to_num(basin_term, nan=np.inf)
            
            hard_penalty = np.where(geo_dist > 1, 10, 0)
            
            cost = geo_dist + basin_term + hard_penalty
            best_index = np.argmin(cost)
            
            # Compute scaling factor: fac = WBMsed_basin / estuary_basin
            wbm_basin_area_best = wbm_basin_areas[best_index]
            matched_dist = geo_dist[best_index]
            if not np.isnan(wbm_basin_area_best) and wbm_basin_area_best > 0:
                fac = wbm_basin_area_best / estuary_basin_area
                print(f"  {estuary}: WBMsed basin={wbm_basin_area_best:.1f} km², "
                      f"estuary basin={estuary_basin_area:.1f} km², fac={fac:.4f}, "
                      f"dist={matched_dist:.3f}°")
            else:
                fac = 1.0
                print(f"  {estuary}: Could not determine WBMsed basin area for best point. "
                      f"fac=1.0 (no scaling).")
        else:
            # No basin area available: use simple distance-based selection
            if has_raster:
                geo_dist = np.sqrt((wbm_lat_deg - lat)**2 + (wbm_lon_deg - lon)**2)
            else:
                # Fall back to rm-space distance
                target_rm_lon, target_rm_lat = transform_coordinates(lon, lat)
                geo_dist = np.sqrt((rm_lon - target_rm_lon)**2 + (rm_lat - target_rm_lat)**2)
            
            best_index = np.argmin(geo_dist)
            fac = 1.0
            print(f"  {estuary}: No basin area available, using closest WBM point (fac=1.0)")

        # --- Step 3: Extract, clean, and scale ---
        q_data = np.maximum(np.nan_to_num(discharge_series[best_index, :]), 0) * fac
        s_data = np.maximum(np.nan_to_num(sed_series[best_index, :]),       0) * fac

        estuary_discharge_timeseries[estuary] = q_data
        estuary_sed_timeseries[estuary]       = s_data
        estuary_rm_coords[estuary]            = (rm_lon[best_index], rm_lat[best_index])
        estuary_scaling_factors[estuary]      = fac

    return estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries, estuary_scaling_factors


def validate_estuary_location(estuary_name, original_coords, grid_coords, savefig=False, output_dir=None):
    """
    Validate estuary locations by plotting both original coordinates and extracted grid point.
    
    Parameters:
        estuary_name (str): Name of the estuary
        original_coords (tuple): Original (lat, lon) or (lat, lon, basin_area) coordinates
        grid_coords (tuple): Grid coordinates (rm_lon, rm_lat)
        savefig (bool): Whether to save figures
        output_dir (str, optional): Base directory to save figures
    """
    # Support both (lat, lon) and (lat, lon, basin_area) tuples
    lat, lon = original_coords[0], original_coords[1]
    rm_lon, rm_lat = grid_coords
    
    lon_grid = (rm_lon / 10) - 180
    lat_grid = (rm_lat / 10) - 90
    
    fig = plt.figure(figsize=(_CW, _CW))
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [lon - 5, lon + 5, lat - 5, lat + 5]
    ax.set_extent(extent)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.stock_img()
    ax.plot(lon, lat, 'bo', markersize=8, transform=ccrs.PlateCarree(), label='Input coordinates')
    ax.plot(lon_grid, lat_grid, 'ro', markersize=8, transform=ccrs.PlateCarree(), label='Extracted WBMsed grid point')
    ax.set_title(f'{estuary_name}')
    ax.legend()
    
    outdir = Path(output_dir) if output_dir else None
    if savefig and outdir:
        save_path_l = outdir / '00_Estuary_location_validation'
        save_path_l.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_l / f'Location_check_{estuary_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_estuary_timeseries(estuary_name, discharge_series, sed_series, datetimes,
                             savefig=False, output_dir=None, scaling_factor=1.0,
                             window_ma=7, color_discharge='#056C89',
                             color_sediment='tab:green', color_moving_avg='tab:blue'):
    """
    Generate time series plots for a specific estuary's discharge and sediment data.
    
    Parameters:
        estuary_name (str): Name of the estuary
        discharge_series (np.ndarray): Discharge time series data
        sed_series (np.ndarray): Sediment time series data
        datetimes (list): List of datetime objects
        savefig (bool): Whether to save figures
        output_dir (str, optional): Base directory to save figures
        scaling_factor (float): fac value applied during extraction (shown in title)
        window_ma (int): Window size for moving average calculation
    """
    mean_discharge = np.mean(discharge_series)
    mean_sediment  = np.mean(sed_series)
    fac_label = f"  [fac={scaling_factor:.3f}]" if scaling_factor != 1.0 else ""

    outdir = Path(output_dir) if output_dir else None

    def _style_ax(ax):
        """Remove top/right spines and apply date formatting."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    # --- Discharge plot ---
    fig, ax = plt.subplots(figsize=(_FW, _FW * 0.4))
    ax.plot(datetimes, discharge_series, color=color_discharge)
    ax.set_xlabel('time')
    ax.set_ylabel('$Q_{river}$ [m³/s]')
    ax.set_title(f'{estuary_name}')
    ax.grid(True, alpha=0.2, linewidth=0.4)
    _style_ax(ax)
    fig.tight_layout()

    if savefig and outdir:
        save_path = outdir / '01_River_discharge_Qriver_per_estuary'
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f'Q_{estuary_name}.png')
        fig.savefig(save_path / f'Q_{estuary_name}.pdf')
    plt.show()

    # --- Sediment plot ---
    fig, ax = plt.subplots(figsize=(_FW, _FW * 0.4))
    ax.plot(datetimes, sed_series, color=color_sediment)
    ax.axhline(mean_sediment, linestyle='--', linewidth=0.75, color='red',
               label=f'mean = {mean_sediment:.2f}')
    ax.set_xlabel('time')
    ax.set_ylabel('Sediment load [kg/s]')
    ax.set_title(f'Sediment load for {estuary_name} based on WBMsed + BQART{fac_label}')
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(loc='upper right', frameon=False)
    _style_ax(ax)
    fig.tight_layout()

    if savefig and outdir:
        save_path_s = outdir / '02_Sediment_load_per_estuary'
        save_path_s.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_s / f'Sediment_{estuary_name}.png')
        fig.savefig(save_path_s / f'Sediment_{estuary_name}.pdf')
    plt.show()

    # --- Combined plot ---
    fig, ax1 = plt.subplots(figsize=(_FW, _FW * 0.45))
    ax1.set_xlabel('time')
    ax1.set_ylabel('$Q_{river}$ [m³/s]', color=color_discharge)
    ax1.plot(datetimes, discharge_series, color=color_discharge, label='Discharge')
    ax1.axhline(mean_discharge, linestyle='--', linewidth=0.75, color='orange',
                label=f'Mean Q = {mean_discharge:.2f}')
    ax1.tick_params(axis='y', labelcolor=color_discharge)
    ax1.spines['top'].set_visible(False)
    _style_ax(ax1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sediment load [kg/s]', color=color_sediment)
    ax2.plot(datetimes, sed_series, color=color_sediment, label='Sediment')
    ax2.axhline(mean_sediment, linestyle='--', linewidth=0.75, color='red',
                label=f'Mean S = {mean_sediment:.2f}')
    ax2.tick_params(axis='y', labelcolor=color_sediment)
    ax2.spines['top'].set_visible(False)

    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))

    ax1.set_title(f'Combined discharge and sediment for {estuary_name}{fac_label}')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()

    if savefig and outdir:
        save_path_combined = outdir / '03_Combined_Qriver_qs'
        save_path_combined.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_combined / f'Combined_{estuary_name}.png')
        fig.savefig(save_path_combined / f'Combined_{estuary_name}.pdf')
    plt.show()

    # --- Moving Average plot ---
    q_df = pd.Series(discharge_series, index=datetimes)
    moving_avg = q_df.rolling(window=window_ma, center=True).mean()

    fig, ax = plt.subplots(figsize=(_FW, _FW * 0.4))
    ax.plot(datetimes, discharge_series, color='lightgrey', alpha=0.6,
            linewidth=0.5, label='Daily discharge')
    ax.plot(datetimes, moving_avg, color=color_moving_avg, linewidth=1.5,
            label=f'{window_ma}-day moving average')
    ax.axhline(mean_discharge, linestyle='--', linewidth=0.75, color='orange',
               label=f'mean = {mean_discharge:.2f}')
    ax.set_xlabel('time')
    ax.set_ylabel('$Q_{river}$ [m³/s]')
    ax.set_title(f'Long-term trend ($Q_{{river}}$) for {estuary_name}{fac_label}')
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(loc='upper right', frameon=False)
    _style_ax(ax)
    fig.tight_layout()

    if savefig and outdir:
        save_path_ma = outdir / '05_Moving_Average_Discharge'
        save_path_ma.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_ma / f'Trend_{estuary_name}_MA{window_ma}.png')
        fig.savefig(save_path_ma / f'Trend_{estuary_name}_MA{window_ma}.pdf')

    plt.show()

    return mean_discharge, mean_sediment, moving_avg


def plot_annual_max_peaks(estuary_name, discharge_series, datetimes,
                          savefig=False, output_dir=None,
                          color_discharge='#044457', color_annual_max='tab:orange'):
    """
    Plot the discharge time series with annual maximum peaks highlighted as dots
    and dashed lines at the mean discharge and mean annual maximum, to visualise
    how R_peak is derived. Lines are labelled directly on the right-hand side.

    Parameters:
        estuary_name (str): Name of the estuary.
        discharge_series (np.ndarray): Daily discharge time series.
        datetimes (list): List of datetime objects matching discharge_series.
        savefig (bool): Whether to save the figure.
        output_dir (str or Path, optional): Base directory; plot is saved in the
            subfolder ``01_River_discharge_Qriver_per_estuary/annual_max_peaks/``.
        color_discharge (str): Colour for the time series line and mean-Q dashed line.
        color_annual_max (str): Colour for the annual-max dots and mean-annual-max dashed line.
    """
    datetime_index = pd.DatetimeIndex(datetimes)
    q_series = pd.Series(
        np.array(discharge_series, dtype=float),
        index=datetime_index[:len(discharge_series)]
    ).dropna()

    # Date of annual maximum (one per calendar year)
    peak_dates = q_series.resample('YE').apply(
        lambda x: x.idxmax() if len(x) > 0 else pd.NaT
    ).dropna()
    peak_values = q_series[peak_dates]
    mean_annual_max = peak_values.mean()
    mean_q = q_series.mean()
    r_peak = mean_annual_max / mean_q if mean_q != 0 else np.nan

    def _style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    fig, ax = plt.subplots(figsize=(_FW, _FW * 0.4))
    ax.plot(q_series.index, q_series.values,
            color=color_discharge, linewidth=0.75)
    ax.scatter(peak_dates, peak_values,
               color=color_annual_max, s=20, zorder=5)
    ax.axhline(mean_q, linestyle='--', linewidth=0.75, color=color_discharge)
    ax.axhline(mean_annual_max, linestyle='--', linewidth=0.75, color=color_annual_max)
    ax.set_xlabel('Time')
    ax.set_ylabel('$Q_{river}$ [m³/s]')
    ax.set_title(f'{estuary_name}')
    ax.grid(True, alpha=0.2, linewidth=0.4)
    _style_ax(ax)

    # --- Right-side inline labels ---
    # Shrink axes to leave room for labels on the right
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)

    x_label = q_series.index[-1]
    font_size = plt.rcParams.get('axes.labelsize', 8)

    ax.annotate('Daily $Q_{river}$',
                xy=(x_label, q_series.iloc[-1]), xycoords='data',
                xytext=(4, 5), textcoords='offset points',
                va='bottom', ha='left', fontsize=font_size, color=color_discharge,
                annotation_clip=False)
    ax.annotate(f'Mean $Q$ = {mean_q:.0f} m\u00b3/s',
                xy=(x_label, mean_q), xycoords='data',
                xytext=(4, 5), textcoords='offset points',
                va='bottom', ha='left', fontsize=font_size, color=color_discharge,
                annotation_clip=False)
    ax.annotate(f'Mean annual max \n ($R_{{peak}}$ = {r_peak:.1f})',
                xy=(x_label, mean_annual_max), xycoords='data',
                xytext=(4, 5), textcoords='offset points',
                va='bottom', ha='left', fontsize=font_size, color=color_annual_max,
                annotation_clip=False)

    if savefig and output_dir:
        save_path = Path(output_dir) / '01_River_discharge_Qriver_per_estuary' / 'annual_max_peaks'
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f'Q_annualmax_{estuary_name}.png')
        fig.savefig(save_path / f'Q_annualmax_{estuary_name}.pdf')
    plt.show()
