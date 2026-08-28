"""Functions for computing morphological activity metrics.

Includes:
- Cumulative activity (Σ|Δz|)
- Morph-time conversions
- Activity plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

def cumulative_activity(profiles_time_space: np.ndarray) -> np.ndarray:
    """Compute cumulative activity Σ|Δz| along the time axis.
    
    Parameters
    ----------
    profiles_time_space : np.ndarray
        2D array of shape (time, space) with bed levels.
        
    Returns
    -------
    np.ndarray
        Cumulative activity array, same shape (time, space).
    """
    z = np.asarray(profiles_time_space)
    if z.ndim != 2:
        raise ValueError("Expected 2D array (time, space)")
    dz = np.diff(z, axis=0)
    abs_dz = np.abs(dz)
    zeros = np.zeros((1, z.shape[1]))
    abs_dz0 = np.vstack([zeros, abs_dz])
    return np.cumsum(abs_dz0, axis=0)

def relative_bed_change(profiles_time_space: np.ndarray) -> np.ndarray:
    """
    Compute relative change Δz = z_t - z_{t-1} along the time axis.
    
    Returns
    -------
    np.ndarray
        Relative change array, same shape (time, space). 
        The first timestep is padded with zeros.
    """
    z = np.asarray(profiles_time_space)
    dz = np.diff(z, axis=0)
    # Pad the first row with zeros so the output matches input dimensions
    zeros = np.zeros((1, z.shape[1]))
    return np.vstack([zeros, dz])

def morph_years_from_datetimes(times: pd.DatetimeIndex, *, startdate=None, morfac=1.0) -> np.ndarray:
    """Convert datetime series to morphological years.
    
    Parameters
    ----------
    times : pd.DatetimeIndex
        Series of timestamps.
    startdate : str or pd.Timestamp, optional
        Reference start date. If None, uses first timestamp.
    morfac : float
        Morphological acceleration factor.
        
    Returns
    -------
    np.ndarray
        Morphological years from start.
    """
    if startdate is None:
        t0 = times[0]
    else:
        t0 = pd.Timestamp(startdate)
    hydro_years = np.array([(t - t0).total_seconds() / (365.25 * 24 * 3600) for t in times])
    return hydro_years * float(morfac)


def plot_activity_and_first_profile(
    *,
    dist_m: np.ndarray,
    first_profile: np.ndarray,
    final_profile: np.ndarray,
    cumact: np.ndarray,
    morph_years: np.ndarray,
    title: str,
    outpath: Path,
    show: bool = False,
    profile_xlim: tuple = None,
):
    """Plot cumulative activity heatmap with initial and final bed profiles.
    
    Parameters
    ----------
    dist_m : np.ndarray
        Distance along cross-section in meters.
    first_profile : np.ndarray
        Initial bed level profile (t=0).
    final_profile : np.ndarray
        Final bed level profile (t=end).
    cumact : np.ndarray
        Cumulative activity array (time, space).
    morph_years : np.ndarray
        Morphological years for y-axis.
    title : str
        Plot title.
    outpath : Path
        Output file path.
    show : bool
        If True, display plot interactively.
    profile_xlim : tuple, optional
        Fixed (xmin, xmax) in km for both subplots. If None, auto-scale.
    """
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    x_km = dist_m / 1000.0
    extent = [x_km.min(), x_km.max(), morph_years.min(), morph_years.max()]
    vmax = np.nanpercentile(cumact, 98) if np.any(np.isfinite(cumact)) else 1.0

    im = ax0.imshow(
        cumact,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        vmin=0,
        vmax=vmax,
    )
    # Use make_axes_locatable to keep colorbar from misaligning axes
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\Sigma |\Delta z_b|$ [m]", fontsize=11, fontweight='bold')

    ax0.set_ylabel('Morphological time [years]', fontsize=11, fontweight='bold')
    ax0.set_title(title, fontsize=12, fontweight='bold')

    ax1.plot(x_km, first_profile, color='gray', linewidth=1.5, label='t = 0')
    ax1.plot(x_km, final_profile, color='black', linewidth=2, label='t = final')
    ax1.set_xlabel('Cross-section distance [km]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bed level [m]', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper right', fontsize=9)
    # Add invisible spacer to match colorbar width
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    cax1.axis('off')
    if profile_xlim is not None:
        ax0.set_xlim(profile_xlim)
        ax1.set_xlim(profile_xlim)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_bedlevel_evolution(
    *,
    dist_m: np.ndarray,
    bedlevel_stack: np.ndarray,
    morph_years: np.ndarray,
    title: str,
    outpath: Path,
    cmap,  # Pass the result of create_bedlevel_colormap() here
    show: bool = False,
    profile_xlim: tuple = None,
    vmin: float = -15,
    vmax: float = 15
):
    """Plot heatmap of actual bed levels Z over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x_km = dist_m / 1000.0
    extent = [x_km.min(), x_km.max(), morph_years.min(), morph_years.max()]

    im = ax.imshow(
        bedlevel_stack,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Bed Level [m]', fontsize=11, fontweight='bold')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Cross-section distance [km]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Morphological time [years]', fontsize=11, fontweight='bold')

    if profile_xlim is not None:
        ax.set_xlim(profile_xlim)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)