"""Functions for tidal-river dominance analysis and plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors


def select_max_flood_timestep(discharge, km_positions, flood_sign=-1):
	"""Select the timestep with maximum flood penetration (sea to river)."""
	q = discharge.values
	if flood_sign < 0:
		flood_mask = q < 0
	else:
		flood_mask = q > 0

	km_grid = np.broadcast_to(km_positions, q.shape)
	flood_km = np.where(flood_mask, km_grid, np.nan)
	max_flood_km_per_time = np.nanmax(flood_km, axis=1)

	if np.all(np.isnan(max_flood_km_per_time)):
		raise ValueError("No flood conditions found in the selected discharge data")

	t_idx = int(np.nanargmax(max_flood_km_per_time))
	return t_idx, max_flood_km_per_time[t_idx]


def select_max_flood_indices_per_cycle(times, discharge, km_positions, flood_sign=-1):
	"""Select the max-flood timestep within each tidal day (24h) cycle."""
	dt = times[1] - times[0]
	dt_seconds = dt / np.timedelta64(1, 's')
	timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
	n_total = len(times)
	indices = []

	for start in range(0, n_total, timesteps_per_day):
		end = start + timesteps_per_day
		if end > n_total:
			break
		q_slice = discharge.isel(time=slice(start, end))
		try:
			t_idx_local, _ = select_max_flood_timestep(q_slice, km_positions, flood_sign=flood_sign)
		except ValueError:
			continue
		indices.append(start + t_idx_local)

	return np.array(indices, dtype=int)


def select_max_flood_indices_by_period(times, discharge, km_positions, n_periods=3, flood_sign=-1):
	"""Select the max-flood timestep within each of N simulation periods."""
	n_total = len(times)
	period_size = n_total / n_periods
	indices = []

	for period in range(n_periods):
		start = int(period * period_size)
		end = int(min((period + 1) * period_size, n_total))
		if end - start < 2:
			continue
		q_slice = discharge.isel(time=slice(start, end))
		try:
			t_idx_local, _ = select_max_flood_timestep(q_slice, km_positions, flood_sign=flood_sign)
		except ValueError:
			continue
		indices.append(start + t_idx_local)

	return np.array(indices, dtype=int)


def plot_max_flood_profile(data, figsize=(12, 4), y_label='Discharge [m³/s]'):
	"""Plot profile at maximum flood penetration timestep."""
	discharge = data['discharge']
	km_positions = data['km_positions']
	times = data['times_datetime']
	max_flood_km = data.get('max_flood_km')

	if 'time' in discharge.dims:
		q_profile = discharge.isel(time=0).values
	else:
		q_profile = discharge.values
	time_label = pd.to_datetime(times[0]).strftime('%Y-%m-%d %H:%M') if len(times) else ''

	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(km_positions, q_profile, color='tab:blue', linewidth=2)
	ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
	ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
	ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
	if max_flood_km is not None:
		ax.set_title(f'Max flood profile at {time_label} (penetration to km {max_flood_km:.1f})',
					 fontsize=12, fontweight='bold')
	else:
		ax.set_title(f'Max flood profile at {time_label}', fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.5, linestyle=':')

	return fig, ax


def plot_multiple_max_flood_profiles(data, indices, figsize=(12, 5), y_label='Discharge [m³/s]',
									 title='Max-flood profiles (representative periods)'):
	"""Plot multiple max-flood profiles for selected timesteps."""
	discharge = data['discharge']
	km_positions = data['km_positions']
	times = pd.to_datetime(data['times_datetime'])

	fig, ax = plt.subplots(figsize=figsize)
	colors_list = plt.cm.viridis(np.linspace(0, 1, len(indices)))

	for i, t_idx in enumerate(indices):
		q_profile = discharge.isel(time=t_idx).values
		label = times[t_idx].strftime('%Y-%m-%d')
		ax.plot(km_positions, q_profile, color=colors_list[i], linewidth=2, label=label)

	ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
	ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
	ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.5, linestyle=':')
	ax.legend(loc='best', fontsize=9)

	return fig, ax


def plot_discharge_statistics(data, figsize=(14, 8), quantity_name='Discharge', y_label='Discharge [m³/s]'):
	"""Create subplot showing mean, min, max profiles."""
	discharge = data['discharge']
	km_positions = data['km_positions']

	mean_q = discharge.mean(dim='time').values
	std_q = discharge.std(dim='time').values
	max_q = discharge.max(dim='time').values
	min_q = discharge.min(dim='time').values

	fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

	ax = axes[0]
	ax.plot(km_positions, mean_q, 'b-', linewidth=2.5, label=f'Mean {quantity_name}')
	ax.fill_between(km_positions, mean_q - std_q, mean_q + std_q,
					 alpha=0.3, color='blue', label='±1 Std Dev')
	ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
	ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
	ax.set_title(f'Mean Longitudinal {quantity_name} Profile with Variability', fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.5, linestyle=':')
	ax.legend(loc='upper left')

	ax = axes[1]
	ax.fill_between(km_positions, min_q, max_q, alpha=0.4, color='red', label='Min-Max Range')
	ax.plot(km_positions, mean_q, 'b-', linewidth=2, label='Mean', alpha=0.7)
	ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
	ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
	ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
	ax.set_title(f'{quantity_name} Range Over Time', fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.5, linestyle=':')
	ax.legend(loc='upper left')

	return fig, axes


def plot_upstream_inflow_timeseries(data, figsize=(12, 4), quantity_name='Discharge', y_label='Discharge [m³/s]'):
	"""Plot time series at the most upstream cross-section."""
	discharge = data['discharge']
	km_positions = data['km_positions']
	times = data['times_datetime']

	upstream_idx = int(np.argmax(km_positions))
	q_upstream = discharge.isel(cross_section=upstream_idx).values

	times_dt = pd.to_datetime(times)
	q_vals = np.asarray(q_upstream)

	if len(times_dt) > 1:
		dt_seconds = np.diff(times_dt.values) / np.timedelta64(1, 's')
		median_dt = np.nanmedian(dt_seconds)
		gap_threshold = median_dt * 1.5
		gap_indices = np.where(dt_seconds > gap_threshold)[0]

		times_plot = times_dt.to_numpy()
		q_plot = q_vals.copy()
		for offset, idx in enumerate(gap_indices, start=1):
			insert_at = idx + offset
			times_plot = np.insert(times_plot, insert_at, np.datetime64('NaT'))
			q_plot = np.insert(q_plot, insert_at, np.nan)
	else:
		times_plot = times_dt.to_numpy()
		q_plot = q_vals

	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(times_plot, q_plot, color='tab:red', linewidth=1.5)
	ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
	ax.set_xlabel('Time', fontsize=11, fontweight='bold')
	ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
	ax.set_title(f'Upstream {quantity_name} (most upstream cross-section, km {km_positions[upstream_idx]:.1f})',
				 fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.5, linestyle=':')
	locator = mdates.AutoDateLocator()
	ax.xaxis.set_major_locator(locator)
	ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
	fig.autofmt_xdate()

	return fig, ax


def plot_discharge_heatmap(data, figsize=(14, 6), flood_sign=-1, show_flood_limit=True,
						   percentile_low=2, percentile_high=95, symmetric_scale=False,
						   cbar_label='Discharge [m³/s]',
						   title='Discharge Evolution: Space-Time Heatmap',
						   low_label='sea', high_label='river'):
	"""Create a 2D heatmap showing discharge evolution in space and time."""
	discharge = data['discharge']
	km_positions = data['km_positions']
	times = data['times_datetime']

	fig, ax = plt.subplots(figsize=figsize)

	times_num = mdates.date2num(times)
	q_vals = discharge.values
	finite_vals = q_vals[np.isfinite(q_vals)]
	vmin = vmax = None
	if finite_vals.size > 0:
		p_low, p_high = np.percentile(finite_vals, [percentile_low, percentile_high])
		if symmetric_scale:
			abs_lim = max(abs(p_low), abs(p_high))
			vmin, vmax = -abs_lim, abs_lim
		else:
			vmin, vmax = p_low, p_high
		norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
	else:
		norm = None
	im = ax.pcolormesh(km_positions, times_num, q_vals, cmap='RdBu_r', shading='auto', norm=norm)
	cbar = plt.colorbar(im, ax=ax, label=cbar_label)
	if vmin is not None and vmax is not None:
		cbar.set_ticks([vmin, 0.0, vmax])
		cbar.set_ticklabels([f"{vmin:.0f} ({low_label})", "0", f"{vmax:.0f} ({high_label})"])

	ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
	ax.set_ylabel('Time', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.yaxis_date()
	time_span_days = (pd.to_datetime(times[-1]) - pd.to_datetime(times[0])).days if len(times) > 1 else 0
	if time_span_days > 365 * 2:
		locator = mdates.YearLocator(base=1)
		ax.yaxis.set_major_locator(locator)
		ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
	else:
		locator = mdates.MonthLocator(interval=1)
		ax.yaxis.set_major_locator(locator)
		ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

	if show_flood_limit:
		if flood_sign < 0:
			flood_mask = q_vals < 0
		else:
			flood_mask = q_vals > 0
		km_grid = np.broadcast_to(km_positions, q_vals.shape)
		flood_km = np.where(flood_mask, km_grid, np.nan)
		max_flood_km_per_time = np.nanmax(flood_km, axis=1)
		ax.plot(max_flood_km_per_time, times_num, color='black', linewidth=1.2, label='Flood limit')
		ax.legend(loc='upper right', fontsize=9)

	return fig, ax
