"""Bed-level metrics (percentiles or mean) per along-estuary x-bin.

Each metric is computed on a population of grid faces per x-bin ("mask mode"),
all built from the raw bed level:
  'frozen'   faces below `threshold` at t=0 (stable, but blind to channel widening)
  'union'    faces below `threshold` at t=0 OR at the snapshot (includes widening;
             same cut-off, so bank slopes barely enter)
  'envelope' faces below water level(x) + wet margin at t=0 OR at the snapshot, plus
             dry stretches narrower than `max_bar_width` bridged by wet faces (keeps
             mid-channel bars above the limit; needs a water-level table, see wet_limits)
"""
import numpy as np

MASK_MODES = ('frozen', 'union', 'envelope')


def metric_value(values, metric):
    """'mean' or 'pNN' (percentile) of a 1-D bed-level array (NaN if there are no valid values)."""
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    if metric == 'mean':
        return values.mean()
    return np.percentile(values, float(metric[1:]))


def wet_limits(x_bins, water_level_km_m, wet_margin):
    """Bed-level limit per x-bin for the 'envelope' mode: water level at the bin centre
    (linearly interpolated from [(x [km], water level [m]), ...]) + wet_margin."""
    wl_km, wl_m = zip(*water_level_km_m)
    centres_km = 0.5 * (x_bins[:-1] + x_bins[1:]) / 1000.0
    return np.interp(centres_km, wl_km, wl_m) + wet_margin


def _threshold_masks(bed_now, init_bl, face_x, width_mask, x_bins, threshold, mode):
    """'frozen' / 'union' face masks per x-bin (bed below threshold)."""
    low = init_bl < threshold
    if mode == 'union':
        low = low | (bed_now < threshold)
    valid = width_mask & low                                   # NaN -> False
    return [valid & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1]) for k in range(len(x_bins) - 1)]


def _envelope_masks(bed_now, init_bl, face_x, face_y, width_mask, x_bins, wet_limit, max_bar_width):
    """'envelope' face masks per x-bin: wet faces plus dry stretches bridged by wet faces."""
    masks = []
    for k in range(len(x_bins) - 1):
        in_bin = width_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
        wet_all = (init_bl < wet_limit[k]) | (bed_now < wet_limit[k])          # NaN -> False
        mask = np.zeros(in_bin.shape, dtype=bool)
        idx = np.flatnonzero(in_bin)
        if idx.size:
            order = idx[np.argsort(face_y[idx])]
            ys, wet = face_y[order], wet_all[order]
            prev_wet_y = np.maximum.accumulate(np.where(wet, ys, -np.inf))
            next_wet_y = np.minimum.accumulate(np.where(wet, ys, np.inf)[::-1])[::-1]
            bridged = (next_wet_y - prev_wet_y) <= max_bar_width               # inf -> False
            mask[order[wet | bridged]] = True
        masks.append(mask)
    return masks


def channel_masks(mode, bed_now, init_bl, face_x, face_y, width_mask, x_bins, threshold,
                  wet_limit=None, max_bar_width=400.0):
    """One boolean face mask per x-bin for the given mask mode (see module docstring)."""
    if mode in ('frozen', 'union'):
        return _threshold_masks(bed_now, init_bl, face_x, width_mask, x_bins, threshold, mode)
    if mode == 'envelope':
        if wet_limit is None:
            raise ValueError("Mask mode 'envelope' needs wet_limit (see wet_limits).")
        return _envelope_masks(bed_now, init_bl, face_x, face_y, width_mask, x_bins, wet_limit, max_bar_width)
    raise ValueError(f"Unknown mask mode {mode!r} (use one of {MASK_MODES}).")


def bedlevel_profiles(bed_now, init_bl, face_x, face_y, width_mask, x_bins, metrics, mask_mode,
                      threshold, wet_limit=None, max_bar_width=400.0):
    """{metric: value per x-bin}, each metric on its own face population, plus the face
    count per bin of the first metric's population.

    mask_mode : a mask mode for all metrics, or {metric: mask mode}.
    """
    mode_of = mask_mode if isinstance(mask_mode, dict) else {m: mask_mode for m in metrics}
    masks = {mode: channel_masks(mode, bed_now, init_bl, face_x, face_y, width_mask, x_bins,
                                 threshold, wet_limit, max_bar_width)
             for mode in {mode_of[m] for m in metrics}}
    profs = {m: np.array([metric_value(bed_now[mk], m) for mk in masks[mode_of[m]]]) for m in metrics}
    counts = [int(np.count_nonzero(mk)) for mk in masks[mode_of[metrics[0]]]]
    return profs, counts
