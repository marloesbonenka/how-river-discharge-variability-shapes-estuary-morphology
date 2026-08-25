"""
Delft3D-FM River Boundary Condition Generation Script
Author: Marloes Bonenkamp
Date: Janaury 15, 2026
Description: Generates river boundary conditions for an estuary domain, incorporating
             variability, flashiness, climate change scenarios, and a sinusoidal distribution
             of discharge among four river cells. Includes checks for cell-to-cell
             difference limit, preservation of the sinusoidal pattern, and total mean discharge.
"""

#%% Load packages
import os
import sys
from datetime import datetime

#%% Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

# Load functions to create Delft3D-FM boundary files
from FUNCTIONS.FUNCS_create_bc_varyingriver_csv_FM import *
from FUNCTIONS.FUNCS_csv_to_bc_converter import convert_csv_folder_to_bc
from FUNCTIONS.FUNCS_plot_discharge_scenarios import (
    plot_discharge_scenarios_first_year,
    plot_normalized_discharge_variability_one_case,
    compute_scenario_metrics,
)

 #%% Configuration settings
total_discharge = 1000                   # Total river discharge in m³/s
nyears = 52
duration_min    = 365.25 * 24 * 60 * nyears               # Total simulation duration in minutes;  525600 = 1 year;     2629440 = 5 years
time_step_rcel  = 1440                                  # Time step for variations over consecutive river cells in minutes, to force bar formation      

SCENARIO_TYPE = 'new' # 'old' (RCEM 2025, NCK 2026) or 'new' (Gaussian variability scenarios)

# IMPORTANT: Update these values based on your grid
grid_info = {
    'nx': 99,                                           # Number of sea basin cells in x-direction (m-direction)
    'ny': 141,                                          # Number of sea basin cells in y-direction (n-direction)
    'river_cells': [
        (395, 72),
        (395, 71),
        (395, 70),
        (395, 69)
    ]
}

# Specify the directory of model runs
output_dir = rf"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Input\Camille\Q{total_discharge}_2months"

os.makedirs(output_dir, exist_ok=True)

#%%
# Shared parameters injected into every scenario dict at runtime
_shared = {
    "total_discharge": total_discharge,
    "duration_min":    duration_min,
    "time_step":       time_step_rcel,
    "cycle_days":      61,   # length of repeating Gaussian pattern in days (e.g. 365=1yr, 61≈2mo)
}

# --- Old scenarios (RCEM 2025, NCK 2026) – pattern_type based ---
scenarios_old = [
    {"name": f"01_baserun{total_discharge}",     "pattern_type": "constant"},
    {"name": f"02_run{total_discharge}_seasonal", "pattern_type": "seasonal"},
    {"name": f"03_run{total_discharge}_flashy",   "pattern_type": "flashy"},
    {"name": f"04_run{total_discharge}_singlepeak","pattern_type": "singlepeak"},
]

# --- New scenarios – Gaussian variability (peak_ratio × n_peaks) ---
# S1: constant (n_peaks=0); S5–S10: spanned peak/mean × frequency space
#CAMILLE
scenarios_new = [
    {"name": f"01_Qr{total_discharge}_pm1_n0",  "peak_ratio": 1.0, "n_peaks": 0},
    {"name": f"02_Qr{total_discharge}_pm2_n1",  "peak_ratio": 2,   "n_peaks": 1},
    {"name": f"03_Qr{total_discharge}_pm3_n1",  "peak_ratio": 3,   "n_peaks": 1},
    {"name": f"04_Qr{total_discharge}_pm4_n1",  "peak_ratio": 4,   "n_peaks": 1},
    {"name": f"05_Qr{total_discharge}_pm5_n1",  "peak_ratio": 5,   "n_peaks": 1},
]
# scenarios_new = [
#     {"name": f"01_Qr{total_discharge}_pm1_n0",  "peak_ratio": 1.0, "n_peaks": 0},
#     {"name": f"02_Qr{total_discharge}_pm2_n1",  "peak_ratio": 2,   "n_peaks": 1},
#     {"name": f"03_Qr{total_discharge}_pm3_n5",  "peak_ratio": 3,   "n_peaks": 5},
#     {"name": f"04_Qr{total_discharge}_pm3_n1",  "peak_ratio": 3,   "n_peaks": 1},
#     {"name": f"05_Qr{total_discharge}_pm5_n1",  "peak_ratio": 5,   "n_peaks": 1},
#     {"name": f"06_Qr{total_discharge}_pm4_n3",  "peak_ratio": 4,   "n_peaks": 3},
#     {"name": f"07_Qr{total_discharge}_pm3_n4",  "peak_ratio": 3,   "n_peaks": 4},
#     {"name": f"08_Qr{total_discharge}_pm2_n6",  "peak_ratio": 2,   "n_peaks": 6},    # {"name": f"S1_Qr{total_discharge}_pm1.0_n0",  "peak_ratio": 1.0, "n_peaks": 0},
#     {"name": f"09_Qr{total_discharge}_pm5_n3",  "peak_ratio": 5,   "n_peaks": 3},
#     {"name": f"10_Qr{total_discharge}_pm3_n3",  "peak_ratio": 3,   "n_peaks": 3},
#     {"name": f"11_Qr{total_discharge}_pm2_n3",  "peak_ratio": 2,   "n_peaks": 3},
#     {"name": f"12_Qr{total_discharge}_pm5_n4",  "peak_ratio": 5,   "n_peaks": 4},
#     {"name": f"13_Qr{total_discharge}_pm4_n4",  "peak_ratio": 4,   "n_peaks": 4},
#     {"name": f"14_Qr{total_discharge}_pm2_n4",  "peak_ratio": 2,   "n_peaks": 4},
# ]

# --- Select active set based on SCENARIO_TYPE ---
if SCENARIO_TYPE == 'old':
    scenarios     = scenarios_old
    generate_fn   = generate_river_discharges_fm
    bc_prefix     = f"Qr{total_discharge}_inflow_sinuous"
else:
    scenarios     = scenarios_new
    generate_fn   = generate_river_discharges_fm_gaussian
    bc_prefix     = f"Qr{total_discharge}_inflow_sinuous_Gaussian"

#%%
scenario_csv_paths = {}

for scenario in scenarios:
    params        = {**_shared, **scenario}
    scenario_dir  = os.path.join(output_dir, scenario["name"])
    boundary_dir  = os.path.join(scenario_dir, "boundaryfiles_csv")
    os.makedirs(boundary_dir, exist_ok=True)

    generate_fn(
        grid_info=grid_info,
        params=params,
        output_dir=boundary_dir,
        start_date_str='2024-01-01 00:00:00'
    )

    convert_csv_folder_to_bc(
        csv_dir=boundary_dir,
        output_prefix=bc_prefix,
        reference_date=datetime(2001, 1, 1),
        output_dir=scenario_dir,
    )

    scenario_csv_paths[scenario["name"]] = os.path.join(boundary_dir, "discharge_cumulative.csv")

print("Finished --> CSV and .bc files generated successfully!")

#%% --- Plot first year of generated scenarios ---
plots_dir = os.path.join(output_dir, "plots_river_bct")

plot_discharge_scenarios_first_year(
    scenario_csv_paths,
    plots_dir,
    output_filename=f"discharge_scenarios_Qr{total_discharge}_first_year.png",
)

plot_normalized_discharge_variability_one_case(
    scenario_csv_paths,
    plots_dir,
    output_filename=f"discharge_variability_normalized_Qr{total_discharge}.png",
)

print(f"Plots saved to: {plots_dir}")

#%% --- Scenario metrics: compare CV and R_peak against observed WBMsed range ---
df_scenario_metrics = compute_scenario_metrics(scenario_csv_paths)
#%%