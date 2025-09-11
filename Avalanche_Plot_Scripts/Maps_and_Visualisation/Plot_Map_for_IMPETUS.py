#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a map for IMPETUS visulisation.
"""

#%% imports
import sys
import copy
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import pylab as pl
from matplotlib import pyplot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib.colors import ListedColormap

from ava_functions.Lists_and_Dictionaries.Paths import obs_path, path_par


#%% paths
nora3_path = f"{path_par}/IMPETUS/NORA3/"
shp_path = f"{obs_path}/IMPETUS/Debmita/AvalancheRegion_shapefiles/"
warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% load the list of Norwegian mainland warning regions with daily ADLs
a_regs_mainl = pd.read_csv(warn_path + "A_Regs_Mainland.csv", index_col=0)


#%% load the warning regions
warn_nc = Dataset(nora3_path + "Warn_Region_Raster.nc")


#%% load the values
warn = warn_nc.variables["warn_regions"][:].astype(float)
x_full = warn_nc.variables["x"][:]
y_full = warn_nc.variables["y"][:]
lons_full = warn_nc.variables["longitude"][:]
lats_full = warn_nc.variables["latitude"][:]


#%% replace the zeros in warn with NaN
warn[warn == 0] = np.nan


#%% get the number of individual regions
regs_uniq = np.unique(warn)[:-1]  # [:-1] for dropping NaN
regs_uniq = np.array(a_regs_mainl).squeeze()


#%% loop over the regions, sum up the number of grid cells per region and multiply it by 3x3km^2
reg_size = {}

for reg_code in regs_uniq:
    reg_size[reg_code]  = np.sum(warn == reg_code) * 3 * 3
# end for reg_code

print(f"Mean size: {np.mean(np.array(list(reg_size.values())))} km^2")


#%% load the shape file and the NORA3 file for the grid
# shp = gpd.read_file(shp_path + "avalanche_regions.shp")
nora3 = Dataset(nora3_path + "fc2021030100_003_fp.nc")


#%% load the avalanche warning region info
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg")


#%% reproject the shape
orth_crs = ccrs.Orthographic(central_latitude=69, central_longitude=20)
warn_info = warn_info.to_crs(orth_crs)


#%% calculate the area of the regions
warn_info["area_new"] = warn_info["geometry"].area * 1e-6


#%% exclude the Svalbard regions and extract A and B
warn_info = warn_info[warn_info["reg_code"] > 3004]
warn_a_info = warn_info[warn_info["regionType"] == "A"]
warn_b_info = warn_info[warn_info["regionType"] == "B"]


#%% extract the North
warn_a_info_north = warn_a_info[(warn_a_info.reg_code < 3014) & (warn_a_info.reg_code > 3008)]
warn_b_info_north = warn_b_info[(warn_b_info.reg_code < 3014) & (warn_b_info.reg_code > 3008)]


#%%
df = pd.DataFrame({"reg_code":warn_info["reg_code"].iloc[np.argsort(warn_info["reg_code"])],
                   "name":warn_info["safeName"].iloc[np.argsort(warn_info["reg_code"])],
                   "type":warn_info["regionType"].iloc[np.argsort(warn_info["reg_code"])],
#                    "area":warn_info["areal_km2"].iloc[np.argsort(warn_info["reg_code"])],
                   "area_new":np.round(warn_info["area_new"].iloc[np.argsort(warn_info["reg_code"])]).astype(int)})
latex_table = df.to_latex(index=False, column_format='|c|c|c|', caption='Sample Table', label='tab:sample_table',
                          escape=False)
print(latex_table)


#%% calculate the area means
warn_a_amean = warn_a_info["area_new"].mean()
warn_b_amean = warn_b_info["area_new"].mean()
warn_amean = np.mean([warn_a_amean, warn_b_amean])
warn_north_amean = warn_a_info_north["area_new"].mean()


#%% define a list of colors
colors = ['lightgrey', 'yellow', 'orange', 'red']

# create the colormap
custom_cmap = ListedColormap(colors)


#%% set the warning regions for colours
warn_pl = copy.deepcopy(warn)

warn_pl[warn == 3007] = 0
warn_pl[warn == 3008] = 0
warn_pl[warn == 3009] = 2
warn_pl[warn == 3010] = 3
warn_pl[warn == 3011] = 0
warn_pl[warn == 3012] = 0
warn_pl[warn == 3013] = 1
warn_pl[warn >= 3014] = 0

warn_pl[warn > 3017] = np.nan
warn_pl[warn < 3007] = np.nan


#%% plot
# pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Supplementary_Paper_1_Revision/Figures/"

cmap = pyplot.get_cmap('gist_rainbow', len(regs_uniq))

fig = pl.figure()

ax00 = fig.add_subplot(111, projection=orth_crs)

p00 = ax00.pcolormesh(lons_full, lats_full, warn_pl, cmap=custom_cmap, transform=ccrs.PlateCarree(), alpha=0.5)
# cb00 = fig.colorbar(p00, ax=ax00)
# cb00.set_label("Warning region code")

# add the regions shp
"""
shp_orth.plot(ax=ax00, column="N&R", legend=False, zorder=101, linewidth=1.8,
              legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
              edgecolor="black", cmap="Set1")
"""
warn_b_info.plot(ax=ax00, legend=False, zorder=101, linewidth=1,
                 legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                 edgecolor="gray", cmap="Set1")

warn_a_info.plot(ax=ax00, legend=False, zorder=101, linewidth=1.2,
                 legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                 edgecolor="black", cmap="Set1")

ax00.set_extent([10, 35.5, 65.75, 71.5])

ax00.add_feature(cfeature.OCEAN, zorder=0)
# ax00.add_feature(cfeature.LAND, zorder=0)
ax00.add_feature(cfeature.LAKES, zorder=0)
ax00.add_feature(cfeature.COASTLINE, zorder=1)
ax00.add_feature(cfeature.BORDERS, edgecolor="black")

# gl = ax00.gridlines(draw_labels=True, linewidth=0.5, zorder=101)
# gl.top_labels = False
# gl.right_labels = False
# gl.bottom_labels = True

# ax00.set_title("Warning regions Norway")

ax00.axis("off")

# pl.savefig(pl_path + "Norwegian_Warning_Regions.png", dpi=200, bbox_inches="tight")

pl.show()
pl.close()
