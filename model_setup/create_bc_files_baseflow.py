"""
Delft3D-FM River Boundary Condition Generation – Base-flow detailed-hydro runs
Author: Marloes Bonenkamp
Date: November 2026

Description:
    For each detailed-hydro-run '*_mean' folder listed in MEAN_RUNS
    (e.g. dhr_05_Qr500_pm5_n1_mean), creates the matching '*_baseflow' run:
        1. copies the mean run folder (mdu, grid, restart files, ...) to *_baseflow
        2. regenerates the four river .bc files with a CONSTANT discharge equal to
           the base flow of the Gaussian scenarios, Q_base = Q_BASE_FRACTION * Qmean
           (0.8 Qmean, the flow during most of the year), via
           generate_river_discharges_fm_gaussian_phased with n_peaks = 0
        3. renames the SLURM job in submit_singularity_delftblue.sh

    Same final morphology and restart as the mean/peak runs; only the river
    discharge differs. Used for the tidal range at base flow in Fig. 3f
    (postprocessing/fig_morphology_overview.py).
"""

#%% Imports
import os
import re
import sys
import shutil
from datetime import datetime

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

from FUNCTIONS.FUNCS_create_bc_varyingriver_csv_FM import generate_river_discharges_fm_gaussian_phased
from FUNCTIONS.FUNCS_csv_to_bc_converter import convert_csv_folder_to_bc


#%% ── Configuration ──────────────────────────────────────────────────────────

total_discharge = 500          # mean river discharge [m³/s]
Q_BASE_FRACTION = 0.8          # base flow / mean flow (default of the Gaussian generator)
Q_base          = Q_BASE_FRACTION * total_discharge
nyears          = 7            # length of the generated timeseries (as the peak runs)
duration_min    = 365.25 * 24 * 60 * nyears
time_step_rcel  = 1440         # 1 day [minutes]

START_DATE = '2031-01-01 00:00:00'   # timeseries start date (= restart / StartDateTime)

# mean runs to derive a base-flow run from (folder names in Model_Input)
MEAN_RUNS = [
    "dhr_05_Qr500_pm5_n1_mean",
]

OVERWRITE = False              # True: regenerate existing *_baseflow folders

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
csv_root  = os.path.join(output_dir, "boundaryfiles_csv_baseflow")   # kept out of the run folders


#%% ── Build base-flow runs ───────────────────────────────────────────────────

for mean_name in MEAN_RUNS:
    mean_dir = os.path.join(output_dir, mean_name)
    if not mean_name.endswith("_mean") or not os.path.isdir(mean_dir):
        print(f"Skipping {mean_name}: not an existing '*_mean' folder in {output_dir}")
        continue

    run_name = mean_name[:-len("_mean")] + "_baseflow"
    run_dir = os.path.join(output_dir, run_name)

    if os.path.exists(run_dir):
        if not OVERWRITE:
            print(f"{run_name} already exists, skipping (set OVERWRITE = True to redo)")
            continue
    else:
        # Copy everything except the river .bc files, which are regenerated below
        shutil.copytree(
            mean_dir, run_dir,
            ignore=shutil.ignore_patterns(f"??_{bc_prefix}.bc"),
        )

    # ── Generate CSV files: constant Q_base (n_peaks = 0 → flat hydrograph) ──
    params = {
        "total_discharge": Q_base,
        "duration_min":    duration_min,
        "time_step":       time_step_rcel,
        "name":            run_name,
        "peak_ratio":      1,
        "n_peaks":         0,
    }

    boundary_dir = os.path.join(csv_root, run_name)
    os.makedirs(boundary_dir, exist_ok=True)

    generate_river_discharges_fm_gaussian_phased(
        grid_info=grid_info,
        params=params,
        output_dir=boundary_dir,
        start_date_str=START_DATE,
    )

    # ── Convert CSV → .bc (written straight into the run folder) ────────────
    convert_csv_folder_to_bc(
        csv_dir=boundary_dir,
        output_prefix=bc_prefix,
        reference_date=datetime(2001, 1, 1),
        output_dir=run_dir,
    )

    # ── Rename SLURM job ────────────────────────────────────────────────────
    submit_file = os.path.join(run_dir, "submit_singularity_delftblue.sh")
    if os.path.exists(submit_file):
        with open(submit_file, newline="") as f:
            text = f.read()
        text = re.sub(r"(#SBATCH --job-name=)\S+", rf"\g<1>{run_name}", text)
        with open(submit_file, "w", newline="") as f:
            f.write(text)

    print(f"Base-flow run ready ({Q_base:g} m3/s = {Q_BASE_FRACTION:g} Qmean): {run_dir}")
