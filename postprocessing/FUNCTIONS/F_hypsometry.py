import numpy as np


# Compute a hypsometric curve as cumulative area versus sorted bed elevation.
def compute_hypsometric_curve(bedlev_data, valid_mask, face_area=None):
    vals = bedlev_data[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([]), 'Cumulative area'

    if face_area is not None:
        area_vals = face_area[valid_mask]
        area_vals = area_vals[np.isfinite(vals)] if area_vals.shape != vals.shape else area_vals
        area_vals = np.asarray(area_vals, dtype=float)
        if area_vals.size != vals.size or np.all(~np.isfinite(area_vals)):
            area_vals = np.ones_like(vals, dtype=float)
            area_label = 'Cumulative area fraction [-]'
            to_plot_area = False
        else:
            area_vals = np.where(np.isfinite(area_vals), area_vals, 0.0)
            area_label = 'Cumulative area [km²]'
            to_plot_area = True
    else:
        area_vals = np.ones_like(vals, dtype=float)
        area_label = 'Cumulative area fraction [-]'
        to_plot_area = False

    order = np.argsort(vals)
    elev_sorted = vals[order]
    area_sorted = area_vals[order]
    cum_area = np.cumsum(area_sorted)

    if to_plot_area:
        cum_area = cum_area / 1e6  # m² -> km²
    else:
        cum_area = cum_area / cum_area[-1]

    return elev_sorted, cum_area, area_label