import numpy as np


def classify_intertidal_mask(ds_window, wet_threshold, x_min, x_max,
                              face_x=None, face_x_var='mesh2d_face_x',
                              wet_var='mesh2d_waterdepth'):
    """Classify faces as intertidal (wet at some but not all timesteps within
    the window) and restrict to the tidal zone in x.

    Parameters
    ----------
    ds_window : xarray/xugrid Dataset sliced to the time window of interest.
    wet_threshold : depth [m] above which a cell counts as wet.
    x_min, x_max : along-estuary x-range [m] defining the tidal zone.
    face_x : precomputed face x-coordinates, optional. If omitted, pulled
        from ds_window[face_x_var] (coords or data vars).

    Returns
    -------
    np.ndarray
        Boolean intertidal mask, shape (n_faces,).
    """
    depth_vals = ds_window[wet_var].values   # (n_window, n_faces)
    wet_mask_t = depth_vals > wet_threshold

    always_wet = wet_mask_t.all(axis=0)
    always_dry = (~wet_mask_t).all(axis=0)
    intertidal = ~always_wet & ~always_dry

    if face_x is None:
        face_x = (ds_window.coords[face_x_var].values
                  if face_x_var in ds_window.coords
                  else ds_window[face_x_var].values)

    in_zone = (face_x >= x_min) & (face_x <= x_max)
    return intertidal & in_zone


def intertidal_area_from_mask(mask, ba_da):
    """Compute intertidal area [m2] and face counts from a boolean intertidal
    mask and the mesh2d_flowelem_ba DataArray (static or time-varying).

    Returns
    -------
    area_m2, n_intertidal_faces, n_faces
    """
    ba_vals = ba_da.isel(time=0).values if 'time' in ba_da.dims else ba_da.values
    area_m2 = float(np.nansum(ba_vals[mask]))
    return area_m2, int(mask.sum()), int(mask.shape[0])


def compute_intertidal_hypsometry(bedlev_vals, ba_vals, intertidal_mask):
    """Area-weighted, non-dimensional hypsometric curve for the intertidal
    zone of one scenario.

    Returns
    -------
    elev_sorted    : bed level, ascending [m]
    cum_area_frac  : Ai/Atot, cumulative area up to elev_sorted[i], divided by
                     the total intertidal area of this scenario (not a shared
                     reference area), so curves from scenarios with very
                     different absolute intertidal areas can be compared on
                     the same [0, 1] axis.
    total_area_m2  : Atot for this scenario [m2]
    """
    elev = bedlev_vals[intertidal_mask]
    area = ba_vals[intertidal_mask]

    valid = np.isfinite(elev) & np.isfinite(area)
    elev, area = elev[valid], area[valid]

    if elev.size == 0:
        return np.array([]), np.array([]), 0.0

    order = np.argsort(elev)
    elev_sorted = elev[order]
    area_sorted = area[order]

    cum_area = np.cumsum(area_sorted)
    total_area_m2 = float(cum_area[-1])
    cum_area_frac = cum_area / total_area_m2

    return elev_sorted, cum_area_frac, total_area_m2
