"""Analyze effect of MORFAC on the model results. This script is used to create the figures in the appendix of the paper."""
#%% 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, one level up from appendix/

from functions.F_general import apply_plot_style, setup_variability_run_context
#%%
apply_plot_style("AGU")

BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")

run_context = setup_variability_run_context(BASE_DIR, 
                                            discharge=500,
                                            MORFAC = True)


run_context
# %%
