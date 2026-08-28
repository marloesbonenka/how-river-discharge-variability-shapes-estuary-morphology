import numpy as np

def get_bed_profile_at_x(ds, tree, x_coord, y_range, time_idx, reference_bed=None, detrend=False):
    """
    Extract bed profile at a specific x-coordinate across the y-range.
    Returns distances and bed levels.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing bed level data
    tree : cKDTree
        Spatial tree for nearest neighbor queries
    x_coord : float
        X-coordinate for the cross-section
    y_range : tuple
        (y_min, y_max) range for the cross-section
    time_idx : int
        Time index to extract
    reference_bed : np.ndarray, optional
        Reference bed level for detrending
    detrend : bool
        Whether to apply detrending
    """
    n_points = 100
    y_coords = np.linspace(y_range[0], y_range[1], n_points)
    x_coords = np.full_like(y_coords, x_coord)
    
    query_points = np.vstack([x_coords, y_coords]).T
    _, nearest_indices = tree.query(query_points)
    
    bed_profile = ds['mesh2d_mor_bl'].isel(time=time_idx).values[nearest_indices]
    
    # Apply detrending if requested
    if detrend and reference_bed is not None:
        reference_profile = reference_bed[nearest_indices]
        bed_profile = bed_profile - reference_profile
    
    distances = y_coords - y_range[0]
    
    return distances, bed_profile

def compute_max_channel_width(bed_profile, distances, safety_buffer=0.20):
    """
    Compute the maximum width of individual channels in a cross-section.
    A channel is defined as a continuous stretch below (mean - safety_buffer).
    """
    profile = bed_profile.copy()
    
    if np.all(np.isnan(profile)):
        return 0.0
    
    # Interpolate internal gaps
    nans = np.isnan(profile)
    x = lambda z: z.nonzero()[0]
    if np.any(~nans):
        profile[nans] = np.interp(x(nans), x(~nans), profile[~nans], left=np.nan, right=np.nan)
    
    mean_bl = np.nanmean(profile)
    threshold = mean_bl - safety_buffer
    
    # Identify channel points
    is_channel = np.zeros_like(profile, dtype=bool)
    valid_points = ~np.isnan(profile)
    is_channel[valid_points] = profile[valid_points] < threshold
    
    # Find continuous channel segments
    channel_widths = []
    in_channel = False
    channel_start = 0
    
    for i in range(len(is_channel)):
        if is_channel[i] and not in_channel:
            # Start of a channel
            in_channel = True
            channel_start = i
        elif not is_channel[i] and in_channel:
            # End of a channel
            in_channel = False
            width = distances[i-1] - distances[channel_start]
            channel_widths.append(width)
    
    # Handle case where channel extends to the end
    if in_channel:
        width = distances[-1] - distances[channel_start]
        channel_widths.append(width)
    
    return max(channel_widths) if channel_widths else 0.0
