#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take a first look into the EURO-CORDEX data.
"""


#%% imports
import numpy as np
import xarray as xr
import geopandas as gpd
import pylab as pl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib import gridspec
from ava_functions.Lists_and_Dictionaries.Paths import path_par3


#%% set paths
data_path = f"{path_par3}/IMPETUS/EURO-CORDEX/hist/MPI_RCA/TM/"
fn = "hist_MPI_RCA_TM_daily_1971_v4.nc"
warn_path = "/home/kei070/Documents/IMPETUS/Avalanches_Danger_Files/"


#%% load the avalanche warning region info
crs = ccrs.Orthographic(central_latitude=69, central_longitude=20)
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg").to_crs(crs=crs.proj4_init)


#%% extract the North
warn_info_north = warn_info[(warn_info.reg_code < 3014) & (warn_info.reg_code > 3008)]


#%% load the file
nc = xr.open_dataset(data_path + fn)


#%% extract some data
lons = np.array(nc["lon"])
lats = np.array(nc["lat"])

tmean = np.array(nc["air_temperature__map_hist_daily"][10, :, :]) / 10 - 273.15


#%% plot a map
x1, x2 = 600, 890
y1, y2 = 190, 420

fig = pl.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)

axes = fig.add_subplot(gs[0, 0], projection=crs)

axes.axis("off")

p_tmean = axes.pcolormesh(lons[y1:y2, x1:x2], lats[y1:y2, x1:x2], tmean[y1:y2, x1:x2],
                          vmin=-15, vmax=15, transform=ccrs.PlateCarree(), cmap="RdBu_r")
cb = fig.colorbar(p_tmean, ax=axes, shrink=0.7)
cb.set_label("Elevation in m")

# axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

# add the regions shp
warn_info_north.plot(ax=axes, legend=False, zorder=101, linewidth=1.8,
                    legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                    edgecolor="black", cmap="Set1")

pl.show()
pl.close()


#%% blanket map
x1, x2 = 600, 890
y1, y2 = 190, 420

pl.imshow(tmean[y1:y2, x1:x2])


#%% select the indices
nc_sub = nc.isel(Xc=slice(x1, x2), Yc=slice(y1, y2))


#%% extract the new subset variables
lons_sub = np.array(nc_sub["lon"])
lats_sub = np.array(nc_sub["lat"])

tmean_sub = np.array(nc_sub["air_temperature__map_hist_daily"][10, :, :]) / 10 - 273.15


#%% plot a map of the subset
fig = pl.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)

axes = fig.add_subplot(gs[0, 0], projection=crs)

axes.axis("off")

p_tmean = axes.pcolormesh(lons_sub, lats_sub, tmean_sub,
                          vmin=-15, vmax=15, transform=ccrs.PlateCarree(), cmap="RdBu_r")
cb = fig.colorbar(p_tmean, ax=axes, shrink=0.7)
cb.set_label("Elevation in m")

# axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

# add the regions shp
warn_info_north.plot(ax=axes, legend=False, zorder=101, linewidth=1.8,
                    legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                    edgecolor="black", cmap="Set1")

pl.show()
pl.close()


#%% fill
nc_sub_filled = nc_sub.interpolate_na(dim="Xc", method="nearest")
nc_sub_filled["air_temperature__map_hist_daily"] = nc_sub_filled["air_temperature__map_hist_daily"].fillna(2731.5)


#%% load values
tmean_sub_filled = np.array(nc_sub_filled["air_temperature__map_hist_daily"][10, :, :]) / 10 - 273.15


#%%
# [i if len(nc_sub[i].dims) == 3 else "" for i in nc_sub.data_vars]


#%%
pl.imshow(tmean_sub_filled)


#%% plot a map of the subset
fig = pl.figure(figsize=(8, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)

axes = fig.add_subplot(gs[0, 0], projection=crs)

axes.axis("off")

p_tmean = axes.pcolormesh(lons_sub, lats_sub, tmean_sub_filled,
                          vmin=-15, vmax=15, transform=ccrs.PlateCarree(), cmap="RdBu_r")
cb = fig.colorbar(p_tmean, ax=axes, shrink=0.7)
cb.set_label("Elevation in m")

# axes.add_feature(cfeature.OCEAN, zorder=101)
axes.add_feature(cfeature.COASTLINE, zorder=101)
axes.add_feature(cfeature.BORDERS, edgecolor="black")

# add the regions shp
warn_info_north.plot(ax=axes, legend=False, zorder=101, linewidth=1.8,
                    legend_kwds={"ncol":1, "loc":"upper left"}, facecolor="none",
                    edgecolor="black", cmap="Set1")

pl.show()
pl.close()

