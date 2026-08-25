"""
Functions for plotting discharge scenarios.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# colorblind friendly
# Fallback palette for Gaussian / unrecognized scenarios (Wong 2008)
_FALLBACK_COLORS = [
    '#0072B2',  # dark blue
    '#D55E00',  # red-orange
    '#009E73',  # teal
    '#CC79A7',  # pink
    '#E69F00',  # orange
    '#56B4E9',  # light blue
    '#F0E442',  # yellow
]
_fallback_color_cache = {}

SCENARIO_CONFIG = {
    "baserun": {"color": '#56B4E9', "label": "Constant discharge"},
    "seasonal": {"color": '#E69F00', "label": "Seasonal discharge"},
    "flashy": {"color": '#009E73', "label": "Flashy discharge"},
    "singlepeak": {"color": '#D55E00', "label": "Single peak discharge"},
}

def get_scenario_type(scenario_name):
    """
    Extract scenario type from folder name.
    
    Handles patterns like:
    - '01_baserun250' -> 'baserun'
    - '02_run250_seasonal' -> 'seasonal'
    - '03_run500_flashy' -> 'flashy'
    - '04_run1000_singlepeak' -> 'singlepeak'
    """
    scenario_lower = scenario_name.lower()
    
    # Check for pattern type at the end (seasonal, flashy, singlepeak)
    if scenario_lower.endswith("_seasonal"):
        return "seasonal"
    elif scenario_lower.endswith("_flashy"):
        return "flashy"
    elif scenario_lower.endswith("_singlepeak"):
        return "singlepeak"
    # Check for baserun (constant discharge)
    elif "baserun" in scenario_lower:
        return "baserun"
    
    return None


def get_scenario_label(scenario_name):
    """Map scenario folder name to display label."""
    scenario_type = get_scenario_type(scenario_name)
    if scenario_type:
        return SCENARIO_CONFIG[scenario_type]["label"]
    return scenario_name


def get_scenario_color(scenario_name):
    """Assign color based on scenario type, with fallback palette for new scenarios."""
    scenario_type = get_scenario_type(scenario_name)
    if scenario_type:
        return SCENARIO_CONFIG[scenario_type]["color"]
    # Assign a consistent fallback color from the palette
    if scenario_name not in _fallback_color_cache:
        idx = len(_fallback_color_cache) % len(_FALLBACK_COLORS)
        _fallback_color_cache[scenario_name] = _FALLBACK_COLORS[idx]
    return _fallback_color_cache[scenario_name]


def plot_discharge_scenarios_first_year(
    scenario_csv_paths,
    output_dir,
    output_filename="discharge_scenarios_first_year.png",
):
    """
    Plot cumulative discharge for multiple scenarios (first year only).

    Parameters
    ----------
    scenario_csv_paths : dict
        Mapping of scenario name to CSV path.
        Expected CSV columns: 'timestamp', 'discharge_m3s'
    output_dir : str or Path
        Directory to save the output figure.
    output_filename : str
        Output image filename.
    """
    if not scenario_csv_paths:
        raise ValueError("No scenario CSV paths provided.")

    scenario_items = list(scenario_csv_paths.items())

    first_year = None
    series_data = []

    for scenario_name, csv_path in scenario_items:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns or "discharge_m3s" not in df.columns:
            raise ValueError(f"CSV missing required columns: {csv_path}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if first_year is None:
            first_year = df["timestamp"].min().year

        df_year = df[df["timestamp"].dt.year == first_year]
        series_data.append((scenario_name, df_year))

    if first_year is None:
        raise ValueError("Could not determine first year from data.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10,6))
    for idx, (scenario_name, df_year) in enumerate(series_data):
        cv_value = compute_CV(df_year)
        mean_q = df_year["discharge_m3s"].mean()
        annualmax_mean_value = df_year["discharge_m3s"].max() / mean_q if mean_q != 0 else float('nan')
        label_name = f"{get_scenario_label(scenario_name)}\n$R_{{\\mathrm{{peak}}}}$={annualmax_mean_value:.1f}, CV={cv_value:.2f}"
        color = get_scenario_color(scenario_name)
        plt.plot(
            df_year["timestamp"],
            df_year["discharge_m3s"],
            label=label_name,
            color=color,
            linewidth=2,
        )

    # plt.title(f"Discharge scenarios {first_year}")
    plt.xlabel("date")
    plt.ylabel("discharge [m³/s]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", labelcolor="linecolor")

    ax = plt.gca()
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, transparent=True)
    plt.close()


def plot_normalized_discharge_variability_one_case(
    scenario_csv_paths,
    output_dir,
    output_filename="discharge_variability_normalized_one_case.png",
):
    """
    Plot normalized discharge variability for one discharge case.

    Normalization is done with the mean discharge of the first simulation year
    for each scenario to provide a dimensionless, one-plot-fits-all comparison.
    """
    if not scenario_csv_paths:
        raise ValueError("No scenario CSV paths provided.")

    scenario_items = list(scenario_csv_paths.items())
    first_year = None
    series_data = []

    for scenario_name, csv_path in scenario_items:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns or "discharge_m3s" not in df.columns:
            raise ValueError(f"CSV missing required columns: {csv_path}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        if first_year is None:
            first_year = df["timestamp"].min().year

        df_year = df[df["timestamp"].dt.year == first_year].copy()
        if df_year.empty:
            continue

        mean_q = df_year["discharge_m3s"].mean()
        if mean_q == 0:
            df_year["discharge_norm"] = 0.0
        else:
            df_year["discharge_norm"] = df_year["discharge_m3s"] / mean_q

        df_year["day_of_year"] = df_year["timestamp"].dt.dayofyear
        series_data.append((scenario_name, df_year))

    if not series_data:
        raise ValueError("No valid first-year data found in the provided CSV files.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario_name, df_year in series_data:
        # compute_p95_mean_value = compute_p95_mean(df_year)
        label_name = (
            f"{get_scenario_label(scenario_name)}"
        )
        color = get_scenario_color(scenario_name)
        ax.plot(
            df_year["day_of_year"],
            df_year["discharge_norm"],
            label=label_name,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("day of year")
    ax.set_ylabel("normalized discharge [-]")
    ax.set_xlim(1, 366)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", labelcolor="linecolor")

    fig.tight_layout()
    fig.savefig(output_dir / output_filename, dpi=300, transparent=True)
    plt.close(fig)


def compute_CV(df):
    """
    Compute the coefficient of variation (CV) as a measure of flashiness.
    
    CV = standard deviation / mean
    """
    if df["discharge_m3s"].mean() == 0:
        return 0.0
    return df["discharge_m3s"].std() / df["discharge_m3s"].mean()

def compute_p90_p10(df):
    """
    Compute the 90th and 10th percentiles of discharge.
    """
    p90 = df["discharge_m3s"].quantile(0.9)
    p10 = df["discharge_m3s"].quantile(0.1)
    return p90/p10

def compute_p95_mean(df):
    """
    Compute the 95th and mean percentiles of discharge.
    """
    p95 = df["discharge_m3s"].quantile(0.95)
    mean = df["discharge_m3s"].mean()
    return p95/mean


def compute_scenario_metrics(scenario_csv_paths):
    """
    Compute CV, R_peak (P95/mean, old method), and R_peak_annualmax
    (mean annual maximum / mean, new method consistent with model parameterization)
    for each scenario over the full timeseries. Prints a summary table.

    Parameters
    ----------
    scenario_csv_paths : dict
        Mapping of scenario name to path of discharge_cumulative.csv.

    Returns
    -------
    pd.DataFrame with columns: Scenario, Mean_Q, CV, R_peak_P95, R_peak_annualmax
    """
    import numpy as np

    rows = []
    for scenario_name, csv_path in scenario_csv_paths.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        q_series = df["discharge_m3s"].dropna()
        mean_q   = q_series.mean()
        std_q    = q_series.std()
        cv       = std_q / mean_q if mean_q != 0 else float('nan')

        # Old method: P95 / mean
        p95      = q_series.quantile(0.95)
        r_peak_p95 = p95 / mean_q if mean_q != 0 else float('nan')

        # New method: mean annual maximum / mean (matches model peak_ratio)
        annual_max      = q_series.resample('YE').max()
        mean_annual_max = annual_max.mean()
        r_peak_annualmax = mean_annual_max / mean_q if mean_q != 0 else float('nan')

        rows.append({
            'Scenario':         scenario_name,
            'Mean_Q':           round(mean_q,          2),
            'CV':               round(cv,               4),
            'R_peak_P95':       round(r_peak_p95,       3),
            'R_peak_annualmax': round(r_peak_annualmax, 3),
        })

    df_metrics = pd.DataFrame(rows)
    print("\n--- Scenario metrics (full timeseries) ---")
    print(df_metrics.to_string(index=False))
    return df_metrics

