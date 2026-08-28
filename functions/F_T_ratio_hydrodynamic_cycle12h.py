"""
Helper functions for 12-hour-cycle hydrodynamic T-ratio analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from FUNCTIONS.F_loaddata import load_cross_section_data, load_cross_section_data_from_cache


# =============================================================================
# Core helpers
# =============================================================================

def compute_cycle_indices(time_hours, cycle_hours=12.0):
	"""Return list of (start, end) indices for fixed-length cycles."""
	time_hours = np.asarray(time_hours)
	if len(time_hours) < 2:
		return [], np.nan

	dt_hours = np.nanmedian(np.diff(time_hours))
	if not np.isfinite(dt_hours) or dt_hours <= 0:
		return [], np.nan

	steps_per_cycle = int(np.round(float(cycle_hours) / dt_hours))
	if steps_per_cycle < 1:
		return [], dt_hours

	indices = []
	for start in range(0, len(time_hours), steps_per_cycle):
		end = start + steps_per_cycle
		if end > len(time_hours):
			break
		indices.append((start, end))
	return indices, dt_hours


def _riverward_sign(values) -> float:
	sign = float(np.sign(np.nanmean(values)))
	return 1.0 if sign == 0 else sign


def discharge_amplitude(values, method='half_percentile_range', p_low=5, p_high=95):
	"""Estimate tidal discharge amplitude from a time series within one cycle."""
	values = np.asarray(values)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return np.nan

	if method == 'half_range':
		return 0.5 * (np.nanmax(values) - np.nanmin(values))
	if method == 'half_percentile_range':
		lo, hi = np.nanpercentile(values, [p_low, p_high])
		return 0.5 * (hi - lo)
	if method == 'std_sqrt2':
		return float(np.nanstd(values) * np.sqrt(2.0))

	raise ValueError(f"Unknown amplitude method: {method}")


def get_T_mean_over_year(df, year_start, *, col='T_hydro', year_duration=1.0, fallback_to_interp=True):
	"""Average a cycle-based T-series over a morphological-year window."""
	if df.empty or col not in df.columns:
		return np.nan

	sub = df[['morph_year', col]].dropna()
	if sub.empty:
		return np.nan

	lo = float(year_start)
	hi = float(year_start + year_duration)
	mask = (sub['morph_year'].values >= lo) & (sub['morph_year'].values < hi)
	vals = sub.loc[mask, col].values
	if vals.size == 0:
		if not fallback_to_interp:
			return np.nan
		# fallback: nearest-by-time interpolation at year_start
		x = sub['morph_year'].values
		y = sub[col].values
		if year_start <= x.min():
			return float(y[np.argmin(x)])
		if year_start >= x.max():
			return float(y[np.argmax(x)])
		return float(np.interp(year_start, x, y))

	return float(np.nanmean(vals))


def _select_window_values(df, *, year_start=None, year_duration=1.0, col='T_hydro'):
	"""Return finite values for a full-run or a year window."""
	if df.empty or col not in df.columns:
		return np.array([])

	sub = df[['morph_year', col]].dropna()
	if sub.empty:
		return np.array([])

	if year_start is None:
		vals = sub[col].values
		return vals[np.isfinite(vals)]

	lo = float(year_start)
	hi = float(year_start + year_duration)
	mask = (sub['morph_year'].values >= lo) & (sub['morph_year'].values < hi)
	vals = sub.loc[mask, col].values
	return vals[np.isfinite(vals)]


def load_hydro_cross_section_data(
	his_paths,
	*,
	cache_file=None,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
	dataset_cache=None,
):
	q_var = 'cross_section_discharge'

	if cache_file is not None:
		cache_path = Path(cache_file)
		if cache_path.exists():
			try:
				with xr.open_dataset(cache_path) as ds_cache:
					if q_var not in ds_cache:
						print(f"  [cache] {cache_path.name} exists but misses '{q_var}'; falling back to raw HIS.")
					else:
						print(f"  [cache] Using {cache_path.name} ({q_var})")
						return load_cross_section_data_from_cache(
							cache_path,
							q_var=q_var,
							select_cycles_hydrodynamic=False,
							select_max_flood=False,
							select_max_flood_per_cycle=False,
							exclude_last_timestep=exclude_last_timestep,
							exclude_last_n_days=exclude_last_n_days,
						)
			except Exception as exc:
				print(f"  [cache] Failed to load {cache_path.name}: {exc}. Falling back to raw HIS.")

	return load_cross_section_data(
		his_paths,
		q_var=q_var,
		estuary_only=estuary_only,
		km_range=km_range,
		select_cycles_hydrodynamic=False,
		select_max_flood=False,
		select_max_flood_per_cycle=False,
		exclude_last_timestep=exclude_last_timestep,
		exclude_last_n_days=exclude_last_n_days,
		dataset_cache=dataset_cache,
	)


def compute_T_hydro_cycle_based(
	*,
	data,
	morfac,
	cycle_hours=12.0,
	amp_method='half_percentile_range',
	amp_percentiles=(5, 95),
):
	"""Compute hydrodynamic T ratio per fixed-length cycle."""
	# Support both legacy and current loader keys.
	discharge = data.get('cross_section_discharge', data.get('discharge'))
	if discharge is None:
		raise KeyError("Missing discharge data. Expected key 'cross_section_discharge' (or legacy 'discharge').")
	km_positions = np.asarray(data['km_positions'])
	time_hours = np.asarray(data['time_hours'])

	upstream_idx = int(np.argmax(km_positions))
	downstream_idx = int(np.argmin(km_positions))

	q_upstream = discharge.isel(cross_section=upstream_idx).values
	q_mouth = discharge.isel(cross_section=downstream_idx).values

	cycles, _ = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	if len(cycles) == 0:
		return pd.DataFrame(columns=['morph_year', 'T_hydro', 'Qriver_mean', 'Qtide_hat'])

	river_sign = _riverward_sign(q_upstream)
	p_low, p_high = amp_percentiles

	rows = []
	for start, end in cycles:
		q_up = q_upstream[start:end] * river_sign
		q_m = q_mouth[start:end]

		Qriver_mean = float(np.nanmean(q_up))
		q_m_demeaned = q_m - np.nanmean(q_m)
		Qtide_hat = discharge_amplitude(q_m_demeaned, method=amp_method, p_low=p_low, p_high=p_high)

		T = np.nan
		if np.isfinite(Qriver_mean) and Qriver_mean > 0 and np.isfinite(Qtide_hat):
			T = Qtide_hat / Qriver_mean

		mid_time_hours = 0.5 * (time_hours[start] + time_hours[end - 1])
		morph_year = (mid_time_hours / 24.0 / 365.0) * morfac

		rows.append({
			'morph_year': morph_year,
			'T_hydro': T,
			'Qriver_mean': Qriver_mean,
			'Qtide_hat': Qtide_hat,
		})

	return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_Thydro_timeseries(df, *, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(10, 5))
	if not df.empty:
		ax.plot(df['morph_year'], df['T_hydro'], linewidth=2)
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('Morphological time [years]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ (12h-cycle) [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_Thydro_vs_morfac(df, *, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not df.empty:
		ax.plot(df['morfac'], df['T_hydro'], 'o-', linewidth=2)
	ax.set_xscale('log')
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_Thydro_vs_morfac_min_mean_max(*, min_df, mean_df, max_df, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not min_df.empty:
		ax.plot(min_df['morfac'], min_df['T_hydro'], 'o-', linewidth=2, label='min')
	if not mean_df.empty:
		ax.plot(mean_df['morfac'], mean_df['T_hydro'], 'o-', linewidth=2, label='mean')
	if not max_df.empty:
		ax.plot(max_df['morfac'], max_df['T_hydro'], 'o-', linewidth=2, label='max')
	ax.set_xscale('log')
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_Thydro_vs_run(df, *, x_col, xlabel, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not df.empty:
		ax.plot(df[x_col], df['T_hydro'], 'o-', linewidth=2)
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_Thydro_vs_run_min_mean_max(*, min_df, mean_df, max_df, x_col, xlabel, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not min_df.empty:
		ax.plot(min_df[x_col], min_df['T_hydro'], 'o-', linewidth=2, label='min')
	if not mean_df.empty:
		ax.plot(mean_df[x_col], mean_df['T_hydro'], 'o-', linewidth=2, label='mean')
	if not max_df.empty:
		ax.plot(max_df[x_col], max_df['T_hydro'], 'o-', linewidth=2, label='max')
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_exceedance_by_run(results, *, run_values, run_labels, title, output_path: Path, year_start=None, year_duration=1.0, show=False):
	fig, ax = plt.subplots(figsize=(7, 4))
	for run_id in run_values:
		df = results.get(run_id)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		if vals.size == 0:
			continue
		vals = np.sort(vals)
		n = vals.size
		exc = 1.0 - (np.arange(1, n + 1) / n)
		label = run_labels.get(run_id, str(run_id))
		ax.plot(vals, exc, linewidth=2, label=label)
	ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('T [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('Exceedance P(T > x) [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_pdf_by_run(results, *, run_values, run_labels, title, output_path: Path, year_start=None, year_duration=1.0, bins=30, show=False):
	fig, ax = plt.subplots(figsize=(7, 4))
	for run_id in run_values:
		df = results.get(run_id)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		if vals.size == 0:
			continue
		label = run_labels.get(run_id, str(run_id))
		ax.hist(vals, bins=bins, density=True, histtype='step', linewidth=2, label=label)
	ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('T [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('PDF [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_exceedance(results, *, morfac_values, title, output_path: Path, year_start=None, year_duration=1.0, show=False):
	fig, ax = plt.subplots(figsize=(7, 4))
	for mf in morfac_values:
		df = results.get(mf)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		vals = vals[np.isfinite(vals)]
		if vals.size == 0:
			continue
		vals = np.sort(vals)
		n = vals.size
		exc = 1.0 - (np.arange(1, n + 1) / n)
		ax.plot(vals, exc, linewidth=2, label=f"MF{mf}")
	ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('T [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('Exceedance P(T > x) [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_pdf(results, *, morfac_values, title, output_path: Path, year_start=None, year_duration=1.0, bins=30, show=False):
	fig, ax = plt.subplots(figsize=(7, 4))
	for mf in morfac_values:
		df = results.get(mf)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		if vals.size == 0:
			continue
		ax.hist(vals, bins=bins, density=True, histtype='step', linewidth=2, label=f"MF{mf}")
	ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('T [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('PDF [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)

