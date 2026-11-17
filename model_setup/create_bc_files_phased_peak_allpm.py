"""
Delft3D-FM River Boundary Condition Generation – Phased-peak runs for all pm scenarios
Author: Marloes Bonenkamp
Date: October 2026

Description:
    For every existing detailed-hydro-run '*_n3_mean' folder of one discharge
    (e.g. dhr_06_Qr1000_pm4_n3_mean), creates the matching '*_n3_peak' run:
        1. copies the mean run folder (mdu, grid, restart files, ...) to *_peak
        2. regenerates the four river .bc files with
           generate_river_discharges_fm_gaussian_phased, so that the first
           Gaussian peak is centred on FIRST_PEAK_DAY (day 1 → 2031-01-02),
           i.e. inside the 2-day window of the detailed hydro run
        3. renames the SLURM job in submit_singularity_delftblue.sh

    peak_ratio and n_peaks are parsed from the folder name; n_peaks still
    matters for the peak width (sigma).

    Same setup as dhr_05_Qr500_pm5_n1_peak (Q500).
"""

#%% Imports
import os
import re
import sys
import glob
import shutil
from datetime import datetime

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

from FUNCTIONS.FUNCS_create_bc_varyingriver_csv_FM import generate_river_discharges_fm_gaussian_phased
from FUNCTIONS.FUNCS_csv_to_bc_converter import convert_csv_folder_to_bc
from FUNCTIONS.FUNCS_plot_discharge_scenarios import plot_discharge_scenarios_first_year


#%% ── Configuration ──────────────────────────────────────────────────────────

total_discharge = 500         # mean river discharge [m³/s]
nyears          = 7            # length of the generated timeseries
duration_min    = 365.25 * 24 * 60 * nyears
time_step_rcel  = 1440         # 1 day [minutes]

# ── Timing control ───────────────────────────────────────────────────────────
START_DATE     = '2031-01-01 00:00:00'   # timeseries start date (= restart / StartDateTime)
FIRST_PEAK_DAY = 1                        # day-of-year (0-based) for first peak
                                          # day 1 → peak maximum on 2031-01-02

OVERWRITE = True              # True: regenerate existing *_peak folders

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

# ── Locations ─────────────────────────────────────────────────────────────────
output_dir = (
    rf"u:\PhDNaturalRhythmEstuaries\Models"
    rf"\2_RiverDischargeVariability_domain45x15_Gaussian"
    rf"\Model_Input\Q{total_discharge}\detailed-hydro-run"
)
bc_prefix = f"Qr{total_discharge}_inflow_sinuous_Gaussian"
csv_root  = os.path.join(output_dir, "boundaryfiles_csv_peak")   # kept out of the run folders


#%% ── Find mean runs → peak scenarios ────────────────────────────────────────

scenarios = []
for mean_dir in sorted(glob.glob(os.path.join(output_dir, "*_n3_mean"))):
    mean_name = os.path.basename(mean_dir)
    match = re.search(r"pm(\d+)_n(\d+)_mean$", mean_name)
    if match is None:
        print(f"Skipping {mean_name}: cannot parse pm/n from name")
        continue

    peak_ratio, n_peaks = int(match.group(1)), int(match.group(2))

    scenarios.append({
        "name":           mean_name[:-len("_mean")] + "_peak",
        "mean_dir":       mean_dir,
        "peak_ratio":     peak_ratio,
        "n_peaks":        n_peaks,
        "first_peak_day": FIRST_PEAK_DAY,
    })

for s in scenarios:
    print(f"{s['name']}: peak_ratio={s['peak_ratio']}, n_peaks={s['n_peaks']}")


#%% ── Build peak runs ────────────────────────────────────────────────────────

scenario_csv_paths = {}

for scenario in scenarios:
    peak_dir = os.path.join(output_dir, scenario["name"])

    if os.path.exists(peak_dir):
        if not OVERWRITE:
            print(f"{scenario['name']} already exists, skipping (set OVERWRITE = True to redo)")
            continue
    else:
        # Copy everything except the river .bc files, which are regenerated below
        shutil.copytree(
            scenario["mean_dir"], peak_dir,
            ignore=shutil.ignore_patterns(f"??_{bc_prefix}.bc"),
        )

    # ── Generate CSV files ───────────────────────────────────────────────────
    params = {
        "total_discharge": total_discharge,
        "duration_min":    duration_min,
        "time_step":       time_step_rcel,
        "name":            scenario["name"],
        "peak_ratio":      scenario["peak_ratio"],
        "n_peaks":         scenario["n_peaks"],
        "first_peak_day":  scenario["first_peak_day"],
    }

    boundary_dir = os.path.join(csv_root, scenario["name"])
    os.makedirs(boundary_dir, exist_ok=True)

    generate_river_discharges_fm_gaussian_phased(
        grid_info=grid_info,
        params=params,
        output_dir=boundary_dir,
        start_date_str=START_DATE,
    )

    # ── Convert CSV → .bc (written straight into the peak run folder) ───────
    convert_csv_folder_to_bc(
        csv_dir=boundary_dir,
        output_prefix=bc_prefix,
        reference_date=datetime(2001, 1, 1),
        output_dir=peak_dir,
    )

    # ── Rename SLURM job ────────────────────────────────────────────────────
    submit_file = os.path.join(peak_dir, "submit_singularity_delftblue.sh")
    if os.path.exists(submit_file):
        with open(submit_file, newline="") as f:
            text = f.read()
        text = re.sub(r"(#SBATCH --job-name=)\S+", rf"\g<1>{scenario['name']}", text)
        with open(submit_file, "w", newline="") as f:
            f.write(text)

    scenario_csv_paths[scenario["name"]] = os.path.join(boundary_dir, "discharge_cumulative.csv")
    print(f"Peak run ready: {peak_dir}")


#%% ── Quick plot (first year) ────────────────────────────────────────────────

if scenario_csv_paths:
    plots_dir = os.path.join(output_dir, "plots_river_bct")
    plot_discharge_scenarios_first_year(
        scenario_csv_paths,
        plots_dir,
        output_filename=f"discharge_phased_peak_Qr{total_discharge}_allpm.png",
    )
    print(f"Plot saved to: {plots_dir}")
