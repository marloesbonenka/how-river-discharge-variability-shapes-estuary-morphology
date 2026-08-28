import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

#%%
def _parse_pm_n(label_str):
    """Extract (pm, n) ints from a label like 'pm3_n5' or 'pm1_n0 (constant)'."""
    m = re.match(r'pm(\d+)_n(\d+)', label_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# --- CHECK VARIABLES IN DELFT3D OUTPUT ---
def check_available_variables_xarray(ds):
    """Updated for xarray/dfm_tools datasets"""
    print("Available variables in dataset:\n")
    # xarray uses ds.data_vars for the main variables
    for var_name in sorted(ds.data_vars):
        var = ds[var_name]
        print(f"  {var_name}:")
        print(f"    shape         = {var.shape}")
        print(f"    dimensions    = {var.dims}")
        
        # xarray stores metadata in the .attrs dictionary
        for attr in ['units', 'long_name', 'standard_name', 'description']:
            if attr in var.attrs:
                print(f"    {attr:13} = {var.attrs[attr]}")
        
        print("") 

    return {'all_vars': list(ds.data_vars)}

# --- EXTRACT MORFAC FROM FOLDER NAME ---
def get_mf_number(folder_name):
    # Accept both pathlib.Path and plain string inputs.
    folder_str = os.fspath(folder_name)
    match = re.search(r'MF_?(\d+)', folder_str)
    return int(match.group(1)) if match else 999


def normalize_scenario_key(scenario):
    """Normalize scenario identifiers so both '1' and '01' map to '1'."""
    try:
        return str(int(str(scenario)))
    except Exception:
        return str(scenario)


def get_variability_map(discharge):
    """Return a normalized variability map with both padded and non-padded keys."""
    try:
        q = int(discharge)
    except Exception as exc:
        raise ValueError(f"Invalid discharge value: {discharge}") from exc

    base_map = {
        '1': f'01_baserun{q}',
        '2': f'02_run{q}_seasonal',
        '3': f'03_run{q}_flashy',
        '4': f'04_run{q}_singlepeak',
    }

    # Support both '1' and '01' lookups to avoid per-script key assumptions.
    normalized_map = {}
    for key, value in base_map.items():
        normalized_map[key] = value
        normalized_map[str(int(key)).zfill(2)] = value
    return normalized_map


def find_variability_model_folders(base_path, discharge, scenarios_to_process=None, analyze_noisy=False):
    """Find variability run folders with discharge-specific naming conventions."""
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    try:
        q = int(discharge)
    except Exception as exc:
        raise ValueError(f"Invalid discharge value: {discharge}") from exc

    if q in (500, 250, 1000):
        if analyze_noisy:
            model_folders = [
                f for f in base_path.iterdir()
                if f.is_dir() and f.name and f.name[0].isdigit() and 'noisy' in f.name.lower()
            ]
        else:
            model_folders = [
                f for f in base_path.iterdir()
                if f.is_dir() and f.name and f.name[0].isdigit()
            ]
    else:
        raise ValueError(f"Unsupported DISCHARGE for variability mode: {q}")

    model_folders.sort(key=lambda x: int(x.name.split('_')[0]))

    if scenarios_to_process:
        scenario_filter = {
            int(normalize_scenario_key(s))
            for s in scenarios_to_process
            if str(normalize_scenario_key(s)).isdigit()
        }
        model_folders = [
            f for f in model_folders
            if int(normalize_scenario_key(f.name.split('_')[0])) in scenario_filter
        ]

    return model_folders

# --- EXTRACT COMPUTATION TIME FROM .dia FILE ---
def extract_computation_time(dia_file_path):
    """
    Extract computation time in days and hours from FlowFM_0000.dia file.
    Returns tuple: (days, hours) or (None, None) if not found.
    """
    try:
        with open(dia_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        comp_time_days = None
        comp_time_hours = None
        
        for line in lines:
            if 'total computation time (d)' in line:
                # Extract the number from the line
                # Example: "** INFO   : total computation time (d)  :             4.6385087087"
                parts = line.split(':')
                if len(parts) >= 3:  # There are multiple colons
                    # Take the last part after the last colon
                    try:
                        comp_time_days = float(parts[-1].strip())
                    except ValueError:
                        pass
                        
            elif 'total computation time (h)' in line:
                parts = line.split(':')
                if len(parts) >= 3:
                    try:
                        comp_time_hours = float(parts[-1].strip())
                    except ValueError:
                        pass
        
        return comp_time_days, comp_time_hours
    
    except Exception as e:
        print(f"    Warning: Could not read .dia file: {e}")
        return None, None
    

# --- CUSTOM COLORMAP ---
def create_terrain_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#fcfcfc"), (0.60, "#f3df91"),
        (0.75, "#ffd000"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

def create_detrended_blev_colormap():
    colors = [
        (0.00, "#000066"),  # deep navy
        (0.15, "#0000ff"),  # pure blue
        (0.35, "#00ccff"),  # cyan-blue
        (0.46, "#aaeeff"),  # pale cyan
        (0.50, "#ffffff"),  # white — zero
        (0.54, "#ffffaa"),  # pale yellow
        (0.65, "#ffdd00"),  # yellow
        (0.85, "#ffcc00"),  # gold
        (1.00, "#cc9900"),  # deep gold
    ]
    return LinearSegmentedColormap.from_list(
        "tidal_diverging", colors
    )

def create_bedlevel_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#ffffcc"), (0.60, "#ffcc00"),
        (0.75, "#cc6600"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_bedlevel", colors)

def create_water_colormap():
    # Highlights shallow areas (light) to deep channels (dark)
    colors = [
        (0.00, "#ffffff"), # Very shallow / Shoreline
        (0.20, "#80deea"), # Shallow water
        (0.40, "#26c6da"), # Mid-depth
        (0.60, "#0097a7"), # Deepening
        (0.80, "#01579b"), # Deep water
        (1.00, "#001b3d"), # Maximum depth / Abyssal
    ]
    return LinearSegmentedColormap.from_list("custom_water", colors)

def create_shear_stress_colormap():
    # Indicates low energy (cool) to high erosive force (hot/bright)
    colors = [
        (0.00, "#f2f2f2"), # Near-zero stress (Light Grey)
        (0.20, "#33ccff"), # Low stress (Blue)
        (0.40, "#ffff00"), # Moderate stress (Yellow)
        (0.60, "#ff9900"), # High stress (Orange)
        (0.80, "#ff0000"), # Critical stress (Red)
        (1.00, "#800000"), # Maximum scour potential (Maroon)
    ]
    return LinearSegmentedColormap.from_list("custom_shear", colors)

# --- MORPHOLOGICAL TIME SELECTION ---
def find_timestep_for_target_morphtime(ds, target_morph_years, start_date):
    """
    Find the timestep where morphological time reaches the target.
    Calculates: morph_time = hydro_time_elapsed * morfac
    """
    start_timestamp = pd.Timestamp(start_date)
    times = pd.to_datetime(ds['time'].values)
    
    # Calculate elapsed hydrodynamic time in years for each timestep
    hydro_elapsed_years = np.array([(t - start_timestamp).days / 365.25 for t in times])
    
    # Get MORFAC values at each timestep
    if 'morfac' in ds:
        morfac_values = ds['morfac'].values
    else:
        raise ValueError("MORFAC variable not found in dataset")
    
    # Calculate morphological time at each timestep: morph_time = hydro_time * morfac
    morph_time_years = hydro_elapsed_years * morfac_values
    
    # Find closest timestep to target morphological time
    time_diffs = np.abs(morph_time_years - target_morph_years)
    closest_idx = int(np.argmin(time_diffs))
    
    actual_morph_years = morph_time_years[closest_idx]
    actual_hydro_years = hydro_elapsed_years[closest_idx]
    actual_morfac = morfac_values[closest_idx]
    actual_time = times[closest_idx]
    
    return closest_idx, actual_time, actual_hydro_years, actual_morph_years, actual_morfac



# Convert a datetime64 value to a compact YYYYMMDD string for filenames.
def _date_to_filename_tag(dt64):
    return str(np.datetime_as_string(dt64, unit='D')).replace('-', '')


# Convert a datetime64 value to a readable date label for titles and logs.
def _date_to_label(dt64):
    return str(np.datetime_as_string(dt64, unit='D'))


# Extract the numeric scenario key from a folder name for consistent sorting and mapping.
def _scenario_key_from_folder(folder_name):
    try:
        return str(int(str(folder_name).split('_')[0]))
    except Exception:
        return str(folder_name).split('_')[0]


# Resolve a human-readable scenario label from a folder name.
def _scenario_label(folder_name, scenario_labels_dict):
    key = _scenario_key_from_folder(folder_name)
    return scenario_labels_dict.get(key, str(folder_name))


# Resolve the plotting color for a scenario based on its folder name.
# Falls back to a deterministic tab20 color (based on scenario number) instead of grey.
def _scenario_color(folder_name, scenario_colors_dict):
    import matplotlib.pyplot as plt
    key = _scenario_key_from_folder(folder_name)
    if key in scenario_colors_dict:
        return scenario_colors_dict[key]
    try:
        idx = int(key)
    except (ValueError, TypeError):
        idx = abs(hash(str(key))) % 20
    return plt.get_cmap('tab20')(idx % 20)


# Build a legend entry that includes both scenario name and folder identifier.
def _scenario_legend_label(folder_name, scenario_labels_dict):
    key = _scenario_key_from_folder(folder_name)
    base = scenario_labels_dict.get(key, key)
    return f"{base} ({folder_name})"


# Generate hydrodynamic target snapshot dates, either explicit or evenly spaced in a range.
def get_target_snapshot_dates(count=4, explicit_dates=None, date_range=None):
    if explicit_dates:
        return [np.datetime64(d).astype('datetime64[ns]') for d in explicit_dates]

    count = max(2, int(count))
    if date_range is None:
        start_dt = np.datetime64('2025-01-01').astype('datetime64[ns]')
        end_dt = np.datetime64('2055-12-31').astype('datetime64[ns]')
    else:
        start_dt = np.datetime64(date_range[0]).astype('datetime64[ns]')
        end_dt = np.datetime64(date_range[1]).astype('datetime64[ns]')

    # Build an even spacing in nanoseconds to avoid index-based alignment.
    ns_grid = np.linspace(start_dt.astype('int64'), end_dt.astype('int64'), count)
    return [np.datetime64(int(ns), 'ns') for ns in ns_grid]


# Match each target date to the nearest available model time index and actual date.
def get_snapshot_matches_by_target_dates(time_values, target_dates):
    if len(time_values) == 0:
        return []

    time_dt = np.array(time_values, dtype='datetime64[ns]')
    time_ns = time_dt.astype('int64')
    matches = []
    for target_dt in target_dates:
        target_ns = np.datetime64(target_dt, 'ns').astype('int64')
        ts_idx = int(np.argmin(np.abs(time_ns - target_ns)))
        actual_dt = time_dt[ts_idx]
        matches.append((target_dt, ts_idx, actual_dt))
    return matches


# Sort scenario keys numerically when possible, otherwise lexicographically.
def sort_scenario_keys(keys):
    return sorted(keys, key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))


# Group per-run snapshot results by scenario key derived from folder names.
def group_snapshot_by_scenario(snapshot_results):
    grouped = {}
    for folder_name, data in snapshot_results.items():
        scenario_key = _scenario_key_from_folder(folder_name)
        grouped.setdefault(scenario_key, []).append((folder_name, data))
    return grouped


# Stack a metric across runs into a 2D array (n_runs, n_points).
def stack_metric_arrays(run_items, metric_key):
    arrays = []
    for _, run_data in run_items:
        arr = run_data.get(metric_key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        arrays.append(arr)
    if not arrays:
        return None
    return np.vstack(arrays)


# Plot a mean line and optional min-max envelope for a stacked metric.
def draw_metric_with_optional_envelope(ax, x, y_stack, color, label, add_envelope=False, marker=None, linestyle='-'):
    if y_stack is None:
        return
    y_center = np.nanmean(y_stack, axis=0)
    if add_envelope and y_stack.shape[0] > 1:
        y_min = np.nanmin(y_stack, axis=0)
        y_max = np.nanmax(y_stack, axis=0)
        ax.fill_between(x, y_min, y_max, color=color, alpha=0.18)
    ax.plot(x, y_center, color=color, linewidth=2, label=label, marker=marker, ms=3, linestyle=linestyle)

