""""Plot a world map with the locations of the estuaries 
    in the dataset, for visualisation purposes."""

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

#%%
estuary_coords = {
    # 'Mississippi': (29.15, -89.25),  # USA, for validation
    # 'Amazon': (-0.1, -50.4),         # Brazil, for validation

    'Chao Phraya': (13.55, 100.59),         #Thailand
    'Colorado (MX)': (31.83, -114.82),      #Mexico
    'Cacipore': (3.6, -51.2),               #French Guiana
    'Cromary Firth': (57.69, -4.02),        #UK
    'Demerara': (6.79, -58.18),             #Guyana
    'Firth of Tay': (56.45, -2.83),         #UK
    'Fly': (-8.62, 143.70),                 #Papua New Guinea
    'Gironde': (45.58, -1.05),              #France
    'Guayas': (-2.55, -79.88),              #Ecuador
    'Humber': (53.62, -0.11),               #UK
    'Kumbe': (-8.36, 140.23),               #Indonesia
    'Ord': (-15.5, 128.35),                 #Australia
    'Purna': (20.91, 72.78),                #India
    'Sokyosen': (36.9, 126.9),              #South Korea 
    'Sungai Merauke': (-8.47, 140.35),      #Indonesia
    'Suriname': (5.84, -55.11),             #Suriname
    'Taeryong': (39.63, 125.48),            #North Korea
    'Tapi': (21.15, 72.75),                 #India
    'Yangon': (16.52, 96.29),               #Myanmar
    'Bian': (-8.10, 139.97),                #Indonesia   
    'Western Scheldt': (51.42, 3.57),       #Netherlands

    # #Excluded                                                         because:
    # 'Eel': (40.63, -124.31),                #USA                      - River-dominated
    # 'Klamath': (41.54, -124.08),            #USA                      - River-dominated
    # 'Thames': (51.5, 0.6),                  #UK                       - Too tide-dominated
    # 'Columbia': (46.25, -124.05),           #USA                      - Reason unclear
    # 'Rio de la Plata': (-35.00, -56.00),    #Argentina/Uruguay        - Reason unclear
    # 'Mengha (Ganges-Brahmaputra)': (21.47, 91.06), #Bangladesh        - Reason unclear
}

# Label priority in crowded regions: higher values are kept first.
# Example below prefers plotting Humber and de-prioritizes Cromary Firth.
label_priority = {
    'Humber': 10,
    'Cromary Firth': -10,
    'Sungai Merauke': -10,
    'Ord': 10,
    'Tapi': -10,
    'Purna': 10,
    'Taeryong': -10,
    'Bian': -10
}
# %% --- SETTINGS: toggle style and dot color here ---
STYLE = 'transparent'        # 'default'      →  white background, black outlines, no land fill
                     # 'whitefig'     →  transparent background, white outlines, no land fill, white globe outline
                     # 'transparent'  →  transparent background, black outlines, white land fill, no globe outline
dot_color = '#BE5A38'#023653'

# --- AGU figure sizing (figures must be 50–170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170   # full-width figure; use ~84 for a single-column figure
CBAR_WIDTH_FRACTION = 0.85  # fraction of total width reserved for the map itself (rest = colorbar + label)

# -------------------------------------------------------------------------
_AGU_RCPARAMS = {
    # --- Typography ---
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # fallback if Arial unavailable
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',

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
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}

STYLES = {
    'default': {
        **_AGU_RCPARAMS,
        'figsize': (14, 7),
        'fig_facecolor': 'white',
        'ax_facecolor': 'white',
        'transparent_save': False,
        'land_facecolor': 'none',
        'land_edge': 'black',
        'coast_edge': 'black',
        'border_edge': 'black',
        'grid_color': 'grey',
        'label_color': 'black',
        'spine_visible': True,
        'spine_color': 'black',
        'savename': 'worldmap_estuary_locations.png',
    },
    'whitefig': {
        **_AGU_RCPARAMS,
        'figsize': (14, 7),
        'fig_facecolor': 'none',
        'ax_facecolor': 'none',
        'transparent_save': True,
        'land_facecolor': 'none',
        'land_edge': 'white',
        'coast_edge': 'white',
        'border_edge': 'white',
        'grid_color': 'white',
        'label_color': 'white',
        'spine_visible': True,
        'spine_color': 'white',
        'savename': 'worldmap_estuary_locations_whitefig.png',
    },
    'transparent': {
        **_AGU_RCPARAMS,
        'figsize': (14, 7),
        'fig_facecolor': 'none',
        'ax_facecolor': 'none',
        'transparent_save': True,
        'land_facecolor': '#EEF0F2', #'#c3c3c3',  # '#c3e4e9',
        'land_edge': 'black',
        'coast_edge': 'black', 
        'border_edge': 'black',
        'grid_color': 'grey',
        'label_color': 'black',
        'spine_visible': True,
        'spine_color': 'black',
        'savename': 'worldmap_estuary_locations_transparent.png',
    },
    'AGU': {
        **_AGU_RCPARAMS,
        # --- AGU figure size (50–170 mm wide) ---
        'figsize': (FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_WIDTH_MM * MM_TO_IN * 0.5),
        'fig_facecolor': 'white',
        'ax_facecolor': 'white',
        'transparent_save': False,
        'land_facecolor': 'none',
        'land_edge': 'black',
        'coast_edge': 'black',
        'border_edge': 'black',
        'grid_color': 'grey',
        'label_color': 'black',
        'spine_visible': True,
        'spine_color': 'black',
        'savename': 'worldmap_estuary_locations_AGU.png',
    },
}
cfg = STYLES[STYLE]
plt.rcParams.update({k: v for k, v in cfg.items() if '.' in k})
# -------------------------------------------------------------------------

#%%
lats = [v[0] for v in estuary_coords.values()]
lons = [v[1] for v in estuary_coords.values()]

# Figure
fig = plt.figure(figsize=cfg['figsize'])
fig.patch.set_facecolor(cfg['fig_facecolor'])
if cfg['fig_facecolor'] == 'none':
    fig.patch.set_alpha(0)
ax = plt.axes(projection=ccrs.Robinson())
ax.set_facecolor(cfg['ax_facecolor'])
ax.spines['geo'].set_visible(cfg['spine_visible'])
if cfg['spine_visible']:
    ax.spines['geo'].set_edgecolor(cfg['spine_color'])
    ax.spines['geo'].set_linewidth(1.0)

# Land and coastlines
ax.add_feature(cfeature.LAND, facecolor=cfg['land_facecolor'], edgecolor=cfg['land_edge'], linewidth=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor=cfg['coast_edge'])
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7, edgecolor=cfg['border_edge'])

# Gridlines (only visible in 'whitefig' style)
if STYLE == 'whitefig':
    ax.gridlines(color=cfg['grid_color'], linewidth=0.5, linestyle='--', alpha=0.7)

# --- POINTS ---
ax.scatter(
    lons, lats,
    transform=ccrs.PlateCarree(),
    color=dot_color,
    s=75,
    zorder=2
)

# --- LABELS ---
def _distance_km(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in km (haversine)."""
    r = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


names = list(estuary_coords.keys())

# Prefer high-priority labels first; these win in local conflicts.
order = sorted(
    names,
    key=lambda n: (-label_priority.get(n, 0), n),
)

# If a lower-priority estuary sits near a higher-priority one, skip it early.
neighbor_priority_radius_km = 700.0
filtered_names = []
for name in order:
    lat, lon = estuary_coords[name]
    pri = label_priority.get(name, 0)

    overshadowed = False
    for other in names:
        if other == name:
            continue
        other_pri = label_priority.get(other, 0)
        if other_pri <= pri:
            continue
        o_lat, o_lon = estuary_coords[other]
        if _distance_km(lat, lon, o_lat, o_lon) < neighbor_priority_radius_km:
            overshadowed = True
            break

    if not overshadowed:
        filtered_names.append(name)

# True overlap avoidance: place annotations and test rendered text bounding boxes.
offset_candidates = [
    (2, 2), (2, -2), (-2, 2), (-2, -2),
    (3, 1), (3, -1), (-3, 1), (-3, -1),
    (1, 3), (1, -3), (-1, 3), (-1, -3),
    (0, 4), (0, -4),
]

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
accepted_bboxes = []
drawn_annotations = []

for name in filtered_names:
    lat, lon = estuary_coords[name]
    placed = False

    for dx, dy in offset_candidates:
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ann = ax.annotate(
            name,
            xy=(lon, lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=15,
            ha=ha,
            va=va,
            color=cfg['label_color'],
            zorder=4,
        )

        fig.canvas.draw()
        bbox = ann.get_window_extent(renderer=renderer).expanded(1.02, 1.08)

        if any(bbox.overlaps(b) for b in accepted_bboxes):
            ann.remove()
            continue

        accepted_bboxes.append(bbox)
        drawn_annotations.append(ann)
        placed = True
        break

    if not placed:
        continue

ax.set_global()
# plt.title("Global Distribution of Estuaries", fontsize=14)

savepath = r"u:\PhDNaturalRhythmEstuaries\Data\\" + cfg['savename']
plt.savefig(savepath, dpi=300, bbox_inches='tight', transparent=cfg['transparent_save'])
plt.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight', transparent=cfg['transparent_save'])
plt.show()
# %%
