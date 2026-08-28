"""Analyze effect of MORFAC on the model results. This script is used to create the figures in the appendix of the paper."""
#%% 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

from functions.F_general import create_bedlevel_colormap, create_terrain_colormap, create_water_colormap, create_shear_stress_colormap
from functions.F_general import get_variability_map, find_variability_model_folders
from functions.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from functions.F_loaddata import get_stitched_map_run_paths
