#%%
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#%%
def generate_river_discharges_fm(grid_info, params, output_dir, start_date_str='2024-01-01 00:00:00'):
    """
    Generates 5 separate CSV files with NEGATIVE discharge values for DFM.
    1 Cumulative + 4 Individual Section Files.
    """
    total_q = params['total_discharge']
    duration_min = params['duration_min']
    time_step = params['time_step']
    pattern_type = params['pattern_type']
    num_cells = len(grid_info['river_cells'])
    
    # 1. Setup Time and Dates
    time_minutes = np.arange(0, duration_min + time_step, time_step)
    num_steps = len(time_minutes)
    t_days = time_minutes / (60 * 24)
    
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    timestamps = [start_dt + timedelta(minutes=int(m)) for m in time_minutes]

    # 2. Pattern Generation (Calculating absolute magnitudes first)
    if pattern_type == "constant":
        discharge_abs = np.ones(num_steps) * total_q
        
    elif pattern_type == "seasonal":
        seasonal_amplitude = 0.4 
        phase_shift = 200  
        seasonal_factor = 1 - seasonal_amplitude * np.cos(2 * np.pi * (t_days - phase_shift) / 365.25)
        discharge_abs = total_q * seasonal_factor
            
    elif pattern_type == "flashy":
        base_discharge = 0.8 * total_q
        discharge_abs = np.full(num_steps, base_discharge)
        rng = np.random.default_rng(0)
        
        flood_magnitude = 2.5 * total_q
        flood_duration_steps = int(3 * 24 * 60 / time_step)
        low_flow_magnitude = 0.4 * total_q
        low_flow_duration_steps = int(2 * 24 * 60 / time_step)
        
        num_years = int(np.floor(duration_min / (365.25 * 24 * 60)))
        
        for year in range(max(1, num_years)):
            # Floods (Wet Season)
            flood_starts = np.linspace(91, 180, 5, dtype=int)
            for d in flood_starts:
                s = int((year * 365.25 + d) * 24 * 60 / time_step)
                if s < num_steps:
                    discharge_abs[s : s + flood_duration_steps] = flood_magnitude

            # Low flows (Dry Seasons)
            low_days = np.linspace(10, 80, 2, dtype=int).tolist() + np.linspace(200, 350, 3, dtype=int).tolist()
            for d in low_days:
                s = int((year * 365.25 + d) * 24 * 60 / time_step)
                if s < num_steps:
                    discharge_abs[s : s + low_flow_duration_steps] = low_flow_magnitude
    
    elif pattern_type == "singlepeak":
        # Single peak per year, no droughts - same magnitude and duration as flashy
        base_discharge = 0.8 * total_q
        discharge_abs = np.full(num_steps, base_discharge)
        
        flood_magnitude = 2.5 * total_q
        flood_duration_steps = int(3 * 24 * 60 / time_step)  # 3 days
        
        num_years = int(np.floor(duration_min / (365.25 * 24 * 60)))
        
        for year in range(max(1, num_years)):
            # Single flood peak around mid-year (day 135, midpoint of original wet season)
            flood_day = 135
            s = int((year * 365.25 + flood_day) * 24 * 60 / time_step)
            if s < num_steps:
                discharge_abs[s : s + flood_duration_steps] = flood_magnitude
        # No low flow/drought periods
    
    # Scaling & Positivity Check
    discharge_abs[discharge_abs < 0.1 * total_q] = 0.1 * total_q
    scaling_factor = total_q / np.mean(discharge_abs)
    discharge_abs *= scaling_factor

    # 3. Quasi-steady, FM-safe partitioning (Schuurman-style)
    num_cells = len(grid_info['river_cells'])
    num_steps = len(time_minutes)

    # Perturbation parameters
    A = 0.25                       # ±25% amplitude of partitioned discharge
    T_years = 500                   # period (very slow, centennial)
    T_minutes = T_years * 365.25 * 24 * 60

    # Normalized boundary coordinates (0–1 along the boundary)
    x = np.linspace(0, 1, num_cells, endpoint=False)

    weights_stack = np.zeros((num_cells, num_steps))

    for t_idx, t_min in enumerate(time_minutes):
        # Temporal factor is extremely slow; quasi-steady over 10 years
        temporal_factor = np.sin(2 * np.pi * t_min / T_minutes)
        
        # Spatial sinusoid along boundary
        spatial_factor = np.sin(2 * np.pi * x)
        
        # Compute fraction weights
        w = (1 / num_cells) * (1 + A * temporal_factor * spatial_factor)
        
        # Safety: enforce positivity and normalize
        w[w < 0] = 0.0
        w /= w.sum()
        
        weights_stack[:, t_idx] = w

    # 4. Save Files
    spinup_duration = 2880 # 2 days
    
    # File 1: Cumulative (Sum of all cells)
    pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': discharge_abs}).to_csv(
        os.path.join(output_dir, 'discharge_cumulative.csv'), index=False)

    # Files 2-5: Individual Sections
    for i in range(num_cells):
        section_q = discharge_abs * weights_stack[i, :]
        
        # Apply Spin-up (constant mean inflow)
        mask = time_minutes <= spinup_duration
        section_q[mask] = total_q / num_cells
        
        # Save
        pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': section_q}).to_csv(
            os.path.join(output_dir, f'river_section_{i+1}.csv'), index=False)

    print(f"Scenario {params.get('name', '')}: Successfully saved 5 CSVs.")


def generate_river_discharges_fm_gaussian(grid_info, params, output_dir, start_date_str='2024-01-01 00:00:00'):
    """
    Generates 5 separate CSV files with discharge values for Delft3D-FM using
    a Gaussian-peak discharge scenario (positive pulses above a low-flow base).

    The yearly pattern is constructed so that:
      - Base flow  = Q_base_fraction * total_discharge  (default: 0.8)
      - Each of the n_peaks Gaussian events is sized to restore the yearly mean
        to total_discharge (same volume-conservation logic as the scenario script).
      - The pattern repeats every year across the full simulation duration.

    Parameters in params dict
    -------------------------
    name             : str   – scenario label used in the print message
    total_discharge  : float – target yearly-mean discharge [m³/s]
    duration_min     : float – total simulation duration [minutes]
    time_step        : float – model time step [minutes]  (1440 = 1 day)
    peak_ratio       : float – Q_peak / total_discharge  (e.g. 3.0 means peak is 3× the mean)
    n_peaks          : int   – number of Gaussian events per year (0 → constant flow)
    Q_base_fraction  : float – fraction of mean used as base flow (default 0.8)

    Output
    ------
    discharge_cumulative.csv  – total discharge at each time step
    river_section_1..N.csv    – per-cell discharge with quasi-steady sinusoidal partitioning
    """
    total_q          = params['total_discharge']
    duration_min     = params['duration_min']
    time_step        = params['time_step']
    peak_ratio       = params['peak_ratio']
    n_peaks          = params['n_peaks']
    Q_base_fraction  = params.get('Q_base_fraction', 0.8)
    cycle_days       = params.get('cycle_days', 365)   # length of one repeating pattern [days]
    num_cells        = len(grid_info['river_cells'])

    # ------------------------------------------------------------------
    # 1. Model time axis
    # ------------------------------------------------------------------
    time_minutes = np.arange(0, duration_min + time_step, time_step)
    num_steps    = len(time_minutes)
    t_days       = time_minutes / (60.0 * 24.0)

    start_dt   = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    timestamps = [start_dt + timedelta(minutes=int(m)) for m in time_minutes]

    # ------------------------------------------------------------------
    # 2. Build the Gaussian pattern (daily resolution, cycle_days long)
    # ------------------------------------------------------------------
    days_year = cycle_days
    t_year    = np.arange(days_year, dtype=float)  # day index 0 … (cycle_days-1)
    Q_base    = Q_base_fraction * total_q

    if n_peaks == 0:
        # Constant flow – no peaks
        q_year = np.full(days_year, total_q)
    else:
        Q_peak         = total_q * peak_ratio
        A              = Q_peak - Q_base                          # pulse amplitude above base
        V_total_excess = (total_q - Q_base) * days_year          # excess volume to distribute
        V_event        = V_total_excess / n_peaks                 # volume per Gaussian event
        sigma          = V_event / (A * np.sqrt(2.0 * np.pi))    # width that gives correct area

        segment        = days_year / n_peaks
        # Snap centers to integer days so each peak falls exactly on a sample
        # point → sampled Q_peak = peak_ratio * total_q regardless of cycle_days.
        # (Poisson summation guarantees discrete Gaussian sum ≈ continuous
        # integral, so volume conservation is unaffected.)
        event_centers  = (np.round(np.linspace(segment / 2.0, days_year - segment / 2.0, n_peaks)).astype(int) % days_year)

        q_year = np.full(days_year, Q_base)
        for t0 in event_centers:
            q_year += A * np.exp(-(t_year - t0) ** 2 / (2.0 * sigma ** 2))

    # ------------------------------------------------------------------
    # 3. Map model time axis onto the yearly pattern (tiled, with wrap)
    # ------------------------------------------------------------------
    # Append a wrap-around point so interpolation covers [0, 365)
    t_wrap = np.append(t_year, float(days_year))
    q_wrap = np.append(q_year, q_year[0])

    t_year_interp = np.mod(t_days, float(days_year))
    discharge_abs = np.interp(t_year_interp, t_wrap, q_wrap)

    # ------------------------------------------------------------------
    # 4. Quasi-steady sinusoidal spatial partitioning (same as original)
    # ------------------------------------------------------------------
    A_spat    = 0.25
    T_years   = 500
    T_minutes = T_years * 365.25 * 24.0 * 60.0

    x = np.linspace(0, 1, num_cells, endpoint=False)
    weights_stack = np.zeros((num_cells, num_steps))

    for t_idx, t_min in enumerate(time_minutes):
        temporal_factor = np.sin(2.0 * np.pi * t_min / T_minutes)
        spatial_factor  = np.sin(2.0 * np.pi * x)
        w = (1.0 / num_cells) * (1.0 + A_spat * temporal_factor * spatial_factor)
        w[w < 0] = 0.0
        w /= w.sum()
        weights_stack[:, t_idx] = w

    # ------------------------------------------------------------------
    # 5. Save CSV files
    # ------------------------------------------------------------------
    spinup_duration = 2880  # 2 days in minutes

    # Cumulative file
    pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': discharge_abs}).to_csv(
        os.path.join(output_dir, 'discharge_cumulative.csv'), index=False)

    # Per-cell files
    for i in range(num_cells):
        section_q = discharge_abs * weights_stack[i, :]

        # Spin-up: force constant mean inflow for first 2 days
        mask = time_minutes <= spinup_duration
        section_q[mask] = total_q / num_cells

        pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': section_q}).to_csv(
            os.path.join(output_dir, f'river_section_{i+1}.csv'), index=False)

    print(f"Scenario {params.get('name', '')} (Gaussian, peak_ratio={peak_ratio}, n_peaks={n_peaks}, "
          f"cycle_days={cycle_days}): Successfully saved {num_cells + 1} CSVs.")


def generate_river_discharges_fm_gaussian_phased(grid_info, params, output_dir, start_date_str='2024-01-01 00:00:00'):
    """
    Identical to generate_river_discharges_fm_gaussian, but the timing of the
    Gaussian peaks can be controlled via ``first_peak_day`` in params.

    Extra entry in params
    ---------------------
    first_peak_day : float
        Day-of-year (0-based) at which the FIRST Gaussian peak is centred.
        Subsequent peaks are evenly spaced after this one.
        Default: same as the original function (segment / 2).

    All other params and the output file format are identical to
    generate_river_discharges_fm_gaussian.
    """
    total_q         = params['total_discharge']
    duration_min    = params['duration_min']
    time_step       = params['time_step']
    peak_ratio      = params['peak_ratio']
    n_peaks         = params['n_peaks']
    Q_base_fraction = params.get('Q_base_fraction', 0.8)
    first_peak_day  = params.get('first_peak_day', None)   # NEW parameter
    num_cells       = len(grid_info['river_cells'])

    # ------------------------------------------------------------------
    # 1. Model time axis
    # ------------------------------------------------------------------
    time_minutes = np.arange(0, duration_min + time_step, time_step)
    num_steps    = len(time_minutes)
    t_days       = time_minutes / (60.0 * 24.0)

    start_dt   = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    timestamps = [start_dt + timedelta(minutes=int(m)) for m in time_minutes]

    # ------------------------------------------------------------------
    # 2. Build the 1-year Gaussian pattern (daily resolution, 365 days)
    # ------------------------------------------------------------------
    days_year = 365
    t_year    = np.arange(days_year, dtype=float)
    Q_base    = Q_base_fraction * total_q

    if n_peaks == 0:
        q_year = np.full(days_year, total_q)
    else:
        Q_peak         = total_q * peak_ratio
        A              = Q_peak - Q_base
        V_total_excess = (total_q - Q_base) * days_year
        V_event        = V_total_excess / n_peaks
        sigma          = V_event / (A * np.sqrt(2.0 * np.pi))

        segment = days_year / n_peaks

        # Use provided first_peak_day, or fall back to original centring
        if first_peak_day is None:
            t0_first = segment / 2.0
        else:
            t0_first = float(first_peak_day)

        # Snap centers to integer days (same reason as in generate_river_discharges_fm_gaussian)
        event_centers = [int(round((t0_first + i * segment) % days_year)) for i in range(n_peaks)]

        q_year = np.full(days_year, Q_base)
        for t0 in event_centers:
            # Wrapped Gaussian: add contributions from t0-365 and t0+365 to
            # handle peaks that are close to the year boundary.
            for offset in (-days_year, 0, days_year):
                q_year += A * np.exp(-(t_year - (t0 + offset)) ** 2 / (2.0 * sigma ** 2))

    # ------------------------------------------------------------------
    # 3. Map model time axis onto the yearly pattern (tiled, with wrap)
    # ------------------------------------------------------------------
    t_wrap = np.append(t_year, float(days_year))
    q_wrap = np.append(q_year, q_year[0])

    t_year_interp = np.mod(t_days, float(days_year))
    discharge_abs = np.interp(t_year_interp, t_wrap, q_wrap)

    # ------------------------------------------------------------------
    # 4. Quasi-steady sinusoidal spatial partitioning (same as original)
    # ------------------------------------------------------------------
    A_spat    = 0.25
    T_years   = 500
    T_minutes = T_years * 365.25 * 24.0 * 60.0

    x = np.linspace(0, 1, num_cells, endpoint=False)
    weights_stack = np.zeros((num_cells, num_steps))

    for t_idx, t_min in enumerate(time_minutes):
        temporal_factor = np.sin(2.0 * np.pi * t_min / T_minutes)
        spatial_factor  = np.sin(2.0 * np.pi * x)
        w = (1.0 / num_cells) * (1.0 + A_spat * temporal_factor * spatial_factor)
        w[w < 0] = 0.0
        w /= w.sum()
        weights_stack[:, t_idx] = w

    # ------------------------------------------------------------------
    # 5. Save CSV files
    # ------------------------------------------------------------------
    spinup_duration = 0  # 2 days in minutes

    pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': discharge_abs}).to_csv(
        os.path.join(output_dir, 'discharge_cumulative.csv'), index=False)

    for i in range(num_cells):
        section_q = discharge_abs * weights_stack[i, :]
        mask = time_minutes <= spinup_duration
        section_q[mask] = total_q / num_cells
        pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': section_q}).to_csv(
            os.path.join(output_dir, f'river_section_{i+1}.csv'), index=False)

    print(f"Scenario {params.get('name', '')} (Gaussian phased, peak_ratio={peak_ratio}, "
          f"n_peaks={n_peaks}, first_peak_day={first_peak_day}): "
          f"Successfully saved {num_cells + 1} CSVs.")