import numpy as np
from scipy.spatial import cKDTree


def get_nearest_face_indices(tree: cKDTree, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Return nearest face indices for the provided query points.

    Notes
    -----
    This is purely geometric and can be reused across timesteps.
    """
    query_points = np.vstack([x_coords, y_coords]).T
    _, nearest_indices = tree.query(query_points)
    return nearest_indices

def get_bed_profile(ds_map, tree, x_coords, y_coords, time_idx, *, var_name: str = 'mesh2d_mor_bl'):
    """Sample the bedlevel along a line at one timestep.

    This avoids materializing the full mesh array for the timestep by indexing
    the face dimension first.
    """
    nearest_indices = get_nearest_face_indices(tree, x_coords, y_coords)

    # xugrid objects can raise on repeated indices; unwrap to plain xarray first.
    xr_ds = getattr(ds_map, 'obj', ds_map)
    bed_profile = xr_ds[var_name].isel(time=time_idx, mesh2d_nFaces=nearest_indices).values
    return bed_profile

def compute_braiding_index_with_threshold(bed_profile_in, safety_buffer=0.20):
    # Work on a copy so we don't mess up the plot
    profile = bed_profile_in.copy()
    
    if np.all(np.isnan(profile)): return 0
    
    # Interpolate ONLY internal gaps (left/right = np.nan prevents flat lines at edges)
    nans = np.isnan(profile)
    x = lambda z: z.nonzero()[0]
    profile[nans] = np.interp(x(nans), x(~nans), profile[~nans], left=np.nan, right=np.nan)

    mean_bl = np.nanmean(profile)
    threshold = mean_bl - safety_buffer
    
    # Identify channels (Ignore NaNs here)
    is_channel = np.zeros_like(profile)
    # A point is a channel only if it's NOT NaN and below the threshold
    valid_points = ~np.isnan(profile)
    is_channel[valid_points] = (profile[valid_points] < threshold).astype(int)
    
    # 5. Count transitions
    transitions = np.diff(is_channel, prepend=0)
    num_channels = np.sum(transitions == 1)
    
    return num_channels


def _interpolate_internal_nans_2d(values: np.ndarray) -> np.ndarray:
    """Interpolate internal NaNs row-wise, keeping NaNs at edges."""
    out = values.copy()
    x = np.arange(out.shape[1])
    for row_idx in range(out.shape[0]):
        row = out[row_idx]
        nans = np.isnan(row)
        if not np.any(nans):
            continue
        if np.all(nans):
            continue
        row[nans] = np.interp(x[nans], x[~nans], row[~nans], left=np.nan, right=np.nan)
        out[row_idx] = row
    return out


def compute_braiding_index_series(bed_profiles_in: np.ndarray, safety_buffer=0.20, *, interpolate_gaps=True) -> np.ndarray:
    """Compute braiding index for a time series of profiles.

    Parameters
    ----------
    bed_profiles_in : np.ndarray
        Array shaped (time, samples).
    safety_buffer : float
        Buffer below the mean bed level to define channels.
    interpolate_gaps : bool
        When True, interpolate internal NaNs per profile (matches legacy behavior).
    """
    profiles = bed_profiles_in.copy()

    if profiles.ndim != 2:
        raise ValueError("bed_profiles_in must be 2D (time, samples).")

    if interpolate_gaps:
        profiles = _interpolate_internal_nans_2d(profiles)

    mean_bl = np.nanmean(profiles, axis=1)
    threshold = mean_bl - safety_buffer

    valid = ~np.isnan(profiles)
    is_channel = (profiles < threshold[:, None]) & valid

    transitions = np.diff(is_channel.astype(np.int8), axis=1, prepend=0)
    num_channels = np.sum(transitions == 1, axis=1)

    return num_channels