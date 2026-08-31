import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#%%
# =============================================================================
# FIGURE STYLE HELPERS
# =============================================================================

def get_agu_rc(font_size=8, title_delta=1, tick_delta=0, mathtext_font='Calibri'):
    """AGU-compliant rcParams: Calibri font, hairline-free line weights (AGU
    rejects < 0.5pt), editable vector text, and 300-600 ppi export.

    Parameters
    ----------
    font_size : base font size for text, axes labels/titles and ticks.
    title_delta : offset added to font_size for figure.titlesize.
    tick_delta : offset added to font_size for tick/legend labels (e.g. -1 to
        make ticks slightly smaller than axis labels).
    """
    tick_size = font_size + tick_delta
    return {
        'font.size': font_size,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': tick_size,
        'figure.titlesize': font_size + title_delta,
        'mathtext.fontset': 'custom',
        'mathtext.rm': mathtext_font,
        'mathtext.it': f'{mathtext_font}:italic',
        'mathtext.bf': f'{mathtext_font}:bold',

        # --- Line weights: avoid hairlines (AGU rejects anything under 0.5pt) ---
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.75,
        'grid.linewidth': 0.4,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.35,
        'ytick.minor.width': 0.35,

        # --- Keep text as editable text in vector exports (not outlined paths) ---
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',

        # --- Resolution / export ---
        'figure.dpi': 150,          # screen preview only
        'savefig.dpi': 300,         # within AGU's 300-600 ppi raster range
    }


# Transparent figure / white text-and-axes style, for slide/poster backgrounds.
WHITEFIG_RC = {
    'figure.facecolor':    'none',
    'axes.facecolor':      'none',
    'axes.edgecolor':      'white',
    'axes.labelcolor':     'white',
    'xtick.color':         'white',
    'ytick.color':         'white',
    'text.color':          'white',
    'grid.color':          'white',
    'legend.facecolor':    'none',
    'legend.edgecolor':    'white',
    'savefig.transparent': True,
}


def apply_plot_style(style='AGU', font_size=8, title_delta=1, tick_delta=0, **rc_overrides):
    """Reset rcParams to matplotlib defaults, then apply a named figure style.

    Parameters
    ----------
    style : 'default' (matplotlib defaults), 'whitefig' (transparent figure,
        white text/axes), or 'AGU' (Calibri, AGU submission specs).
    font_size, title_delta, tick_delta : forwarded to get_agu_rc() when
        style='AGU'.
    rc_overrides : any extra rcParams to apply on top (e.g. to tweak one or
        two entries of the 'whitefig' style per script).
    """
    plt.rcParams.update(plt.rcParamsDefault)
    if style == 'AGU':
        plt.rcParams.update(get_agu_rc(font_size=font_size, title_delta=title_delta, tick_delta=tick_delta))
    elif style == 'whitefig':
        plt.rcParams.update(WHITEFIG_RC)
    elif style != 'default':
        raise ValueError(f"Unknown style '{style}'. Choose 'default', 'whitefig', or 'AGU'.")
    if rc_overrides:
        plt.rcParams.update(rc_overrides)


def compute_map_figsize(xlim, ylim, width_mm, cbar_frac, mm_to_in=1 / 25.4):
    """Derive (width_in, height_in) from data aspect ratio so an equal-aspect
    map fills the frame at the target print width, with space reserved for
    the colorbar."""
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    aspect = y_span / x_span

    fig_width_in = width_mm * mm_to_in
    map_width_in = fig_width_in * cbar_frac
    fig_height_in = map_width_in * aspect
    return (fig_width_in, fig_height_in)


def add_direct_labels(ax, curves, min_sep_frac=0.09, x_offset=6):
    """Label each line directly to the right of its endpoint instead of using
    a legend.

    Parameters
    ----------
    ax : matplotlib Axes
    curves : list of (x_data, y_data, text, color) tuples
    min_sep_frac : minimum vertical spacing between labels (fraction of y-range)
    x_offset : horizontal distance in points between line end and label
    """
    entries = []
    for x_data, y_data, text, color in curves:
        finite = np.isfinite(y_data)
        if not finite.any():
            continue
        entries.append([x_data[finite][-1], y_data[finite][-1], text, color])

    if not entries:
        return

    # Sort and adjust vertical positions to prevent overlaps
    y_lo, y_hi = ax.get_ylim()
    min_gap = min_sep_frac * (y_hi - y_lo)

    entries.sort(key=lambda e: e[1])
    for i in range(1, len(entries)):
        if entries[i][1] - entries[i - 1][1] < min_gap:
            entries[i][1] = entries[i - 1][1] + min_gap

    label_bbox = dict(
        boxstyle="round,pad=0.15",
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
    )

    for x_end, y_end, text, color in entries:
        ax.annotate(
            text,
            xy=(x_end, y_end),
            xytext=(x_offset, 0),  # Positive offset shifts text to the right
            textcoords="offset points",
            color=color,
            ha="left",  # Left-aligned anchor places text extending rightward
            va="center",
            bbox=label_bbox,
            clip_on=False,  # Allows label to draw outside the main axes boundary
        )

    # Optional: Automatically pad the right x-margin so labels have space inside the figure
    ax.set_xmargin(0.1)


def add_fill_label(ax, x_data, y_lower, y_upper, text, color):
    """Place a label inside a fill_between region, centred on its x-extent."""
    finite = np.isfinite(y_lower) & np.isfinite(y_upper)
    if not finite.any():
        return
    xf = x_data[finite]
    yl = y_lower[finite]
    yu = y_upper[finite]

    x_mid = 0.355 * (xf.min() + xf.max())
    idx = np.argmin(np.abs(xf - x_mid))

    ax.text(
        xf[idx], 0.5 * (yl[idx] + yu[idx]),
        text, color=color, ha='center', va='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7),
        clip_on=True,
    )


def strip_top_right_spines(*axes):
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# =============================================================================
# TIME WINDOW HELPERS
# =============================================================================

def get_last_n_hours_window(time_values, n_hours):
    """Boolean mask selecting the last n_hours of a datetime64 array."""
    t_end = time_values[-1]
    t_start = t_end - np.timedelta64(int(n_hours * 3600), 's')
    return time_values >= t_start


def get_last_n_days_window(time_values, n_days):
    """(t_start, t_end) covering the last n_days of a datetime64 array."""
    t_end = time_values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    return t_start, t_end


# =============================================================================
# SCENARIO / RUN FOLDER DISCOVERY
# =============================================================================

# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}
DHR_FOLDER_RE = re.compile(r'^dhr_(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\d+$')


def discover_variability_scenario_folders(dhr_base, discharge, run_ids_to_include=None, folder_regex=DHR_FOLDER_RE):
    """Find dhr_XX_Qr{discharge}_pm{pm}_n{n}[_mean].{runid}-style folders
    under `dhr_base`, optionally restricted to `run_ids_to_include`.

    Returns a list of (folder_path, run_id, pm_val, n_val) tuples.
    """
    dhr_base = Path(dhr_base)
    results = []
    if not dhr_base.exists():
        return results
    for folder in sorted(dhr_base.iterdir()):
        if not folder.is_dir():
            continue
        m = folder_regex.match(folder.name)
        if not m:
            continue
        run_id, q_val, pm_val, n_val = (int(g) for g in m.groups())
        if run_ids_to_include is not None and run_id not in run_ids_to_include:
            continue
        if q_val != discharge:
            continue
        results.append((folder, run_id, pm_val, n_val))
    return results


def find_run_folder_by_qpmn(search_dir, discharge, pm, n, folder_regex, sort_key=None):
    """Find the single run folder under `search_dir` whose name matches
    `folder_regex` (capture groups 1-3 = discharge, pm, n) for the given
    (discharge, pm, n). Raises FileNotFoundError if none match."""
    search_dir = Path(search_dir)
    candidates = []
    for f in search_dir.iterdir():
        if not f.is_dir():
            continue
        m = folder_regex.match(f.name)
        if not m:
            continue
        q_val, pm_val, n_val = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if q_val == discharge and pm_val == pm and n_val == n:
            candidates.append(f)
    if not candidates:
        raise FileNotFoundError(
            f"No folder found for Q={discharge}, pm={pm}, n={n} in {search_dir}"
        )
    candidates.sort(key=sort_key or (lambda x: x.name))
    return candidates[0]


# =============================================================================
# RUN CONTEXT (base path / cache dir / timed-out dir / model folders)
# =============================================================================

def resolve_timed_out_dir(base_path, timed_out_dir=None):
    """Return the timed-out directory Path if it exists, else None (with a
    warning printed once)."""
    timed_out_dir = Path(timed_out_dir) if timed_out_dir is not None else Path(base_path) / "timed-out"
    if not timed_out_dir.exists():
        print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')
        return None
    return timed_out_dir


def setup_variability_run_context(base_directory, discharge, config_subdir="Model_Output",
                                   scenarios_to_process=None, analyze_noisy=False,
                                   noisy_subdir_template="0_Noise_Q{discharge}", MORFAC=False):
    """Resolve the standard variability-run folder layout used across the
    postprocessing scripts: base_path, cache dir, timed-out dir, variability
    map, and the matching model folders.

    Returns a dict with keys: base_path, cache_dir, timed_out_dir,
    variability_map, model_folders.
    """
    base_directory = Path(base_directory)

    base_path = base_directory / f"{config_subdir}/Q{discharge}"

    if MORFAC:
        base_path = base_directory / f"{config_subdir}/Q{discharge}_MORFAC"

    if analyze_noisy:
        base_path = base_path / noisy_subdir_template.format(discharge=discharge)

    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    cache_dir = base_path / "cached_data"
    timed_out_dir = resolve_timed_out_dir(base_path)
    model_folders = find_variability_model_folders(
        base_path=base_path, discharge=discharge,
        scenarios_to_process=scenarios_to_process, analyze_noisy=analyze_noisy,
    )

    return {
        'base_path': base_path,
        'cache_dir': cache_dir,
        'timed_out_dir': timed_out_dir,
        'model_folders': model_folders,
    }


def _parse_pm_n(label_str):
    """Extract (pm, n) ints from a label like 'pm3_n5' or 'pm1_n0 (constant)'."""
    m = re.match(r'pm(\d+)_n(\d+)', label_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

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

# --- CUSTOM COLORMAP ---
def create_terrain_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#fcfcfc"), (0.60, "#f3df91"),
        (0.75, "#ffd000"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

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

