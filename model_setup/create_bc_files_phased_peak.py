"""
Delft3D-FM River Boundary Condition Generation – Phased-peak variant
Author: Marloes Bonenkamp
Date: April 2026

Description:
    Generates river boundary conditions for a single scenario using
    generate_river_discharges_fm_gaussian_phased, which lets you control:
        - START_DATE      : when the timeseries begins
        - FIRST_PEAK_DAY  : day-of-year (0-based) at which the first
                            Gaussian peak is centred.

    Example below uses scenario 12 (pm5_n4) starting on 2031-01-01,
    with the first peak placed at day 1 (so the maximum occurs within
    the first two days of the series).
"""

#%% Imports
import os
import sys
from datetime import datetime

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

from FUNCTIONS.FUNCS_create_bc_varyingriver_csv_FM import generate_river_discharges_fm_gaussian_phased
from FUNCTIONS.FUNCS_csv_to_bc_converter import convert_csv_folder_to_bc
from FUNCTIONS.FUNCS_plot_discharge_scenarios import plot_discharge_scenarios_first_year


#%% ── Configuration ──────────────────────────────────────────────────────────

total_discharge = 500          # mean river discharge [m³/s]
nyears          = 7            # length of the generated timeseries
duration_min    = 365.25 * 24 * 60 * nyears
time_step_rcel  = 1440         # 1 day [minutes]

# ── Timing control ───────────────────────────────────────────────────────────
START_DATE     = '2031-01-01 00:00:00'   # timeseries start date
FIRST_PEAK_DAY = 1                        # day-of-year (0-based) for first peak
                                          # day 1 → peak maximum on 2031-01-02

# ── Scenario 12: pm5_n4 ──────────────────────────────────────────────────────
scenario = {
    "name":       f"12_Qr{total_discharge}_pm5_n4_phased",
    "peak_ratio": 5,
    "n_peaks":    4,
    "first_peak_day": FIRST_PEAK_DAY,
}

# ── Grid (same as main script) ────────────────────────────────────────────────
grid_info = {
    'nx': 99,
    'ny': 141,
    'river_cells': [
        (395, 72),
        (395, 71),
        (395, 70),
        (395, 69),
    ],
}

# ── Output location ───────────────────────────────────────────────────────────
output_dir = (
    rf"u:\PhDNaturalRhythmEstuaries\Models"
    rf"\2_RiverDischargeVariability_domain45x15_Gaussian"
    rf"\Model_Input\Q{total_discharge}\detailed-hydro-run"
)
bc_prefix = f"Qr{total_discharge}_inflow_sinuous_Gaussian"


#%% ── Generate CSV files ─────────────────────────────────────────────────────

params = {
    "total_discharge": total_discharge,
    "duration_min":    duration_min,
    "time_step":       time_step_rcel,
    **scenario,
}

scenario_dir  = os.path.join(output_dir, scenario["name"])
boundary_dir  = os.path.join(scenario_dir, "boundaryfiles_csv")
os.makedirs(boundary_dir, exist_ok=True)
#%%
generate_river_discharges_fm_gaussian_phased(
    grid_info=grid_info,
    params=params,
    output_dir=boundary_dir,
    start_date_str=START_DATE,
)


#%% ── Convert CSV → .bc ──────────────────────────────────────────────────────

convert_csv_folder_to_bc(
    csv_dir=boundary_dir,
    output_prefix=bc_prefix,
    reference_date=datetime(2001, 1, 1),
    output_dir=scenario_dir,
)

print(f"CSV and .bc files written to: {scenario_dir}")


#%% ── Quick plot (first year) ────────────────────────────────────────────────

scenario_csv_paths = {
    scenario["name"]: os.path.join(boundary_dir, "discharge_cumulative.csv")
}
plots_dir = os.path.join(output_dir, "plots_river_bct")

plot_discharge_scenarios_first_year(
    scenario_csv_paths,
    plots_dir,
    output_filename=f"discharge_phased_peak_{scenario['name']}.png",
)

print(f"Plot saved to: {plots_dir}")
