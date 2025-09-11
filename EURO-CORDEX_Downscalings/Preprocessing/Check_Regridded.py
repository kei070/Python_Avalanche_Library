#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the regridded EURO-CORDEX files.
"""

#%% imports
import numpy as np
import xarray as xr
import geopandas as gpd
import pylab as pl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib import gridspec
from ava_functions.Lists_and_Dictionaries.Paths import path_par3, obs_path, path_par


#%% set the variable name dictionary
var_ns = {"TM":"air_temperature__map_hist_daily", "TX":"max_air_temperature__map_hist_daily",
          "TN":"min_air_temperature__map_hist_daily",
          "RR":"precipitation__map_hist_daily",
          "SWE":"snow_water_equivalent__map_hist_daily"}

var = "SWE"


#%% set paths
data_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Regridded/hist/MPI_RCA/"
fn = f"MPI_RCA_{var}_daily_NORA3Grid_1971-2000_v4.nc"
warn_path = f"/{obs_path}/IMPETUS/Avalanches_Danger_Files/"

no3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"


#%% load the avalanche warning region info
crs = ccrs.Orthographic(central_latitude=69, central_longitude=20)
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg").to_crs(crs=crs.proj4_init)


#%% extract the North
warn_info_north = warn_info[(warn_info.reg_code < 3014) & (warn_info.reg_code > 3008)]


#%% load the file
no3_nc = xr.open_dataset(no3_path + "NORA3_NorthNorway_Sub_2024.nc")
nc = xr.open_dataset(data_path + fn)


#%% extract some data
lons = np.array(no3_nc["longitude"])
lats = np.array(no3_nc["latitude"])

tmean = np.array(nc[var_ns[var]][10, :, :]) / 10 - 273.15 if var[0] == "T" else np.array(nc[var_ns[var]][10, :, :])


#%% plot a map of the subset
fig = pl.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)

axes = fig.add_subplot(gs[0, 0], projection=crs)

axes.axis("off")

p_tmean = axes.pcolormesh(lons, lats, tmean, vmin=None, vmax=None, transform=ccrs.PlateCarree(), cmap="Blues")
cb = fig.colorbar(p_tmean, ax=axes, shrink=0.7)
cb.set_label(f"{var} in {nc[var_ns[var]].units}")

# axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

# add the regions shp
warn_info_north.plot(ax=axes, legend=False, zorder=101, linewidth=1.8,
                    legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                    edgecolor="black", cmap="Set1")

pl.show()
pl.close()