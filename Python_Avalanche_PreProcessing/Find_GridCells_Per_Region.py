#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the NORA3 grid cells for the individual warning regions.
"""

#%% imports
import pylab as pl
import numpy as np
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset
from shapely.geometry import Point
from joblib import dump
from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set paths
no3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
rs_path =  f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Rain_and_Snow/"
warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
out_path = f"{path_par}/IMPETUS/NORA3/Cell_Selection/"


#%% load the avalanche warning region info
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg")
warn_info = warn_info[(warn_info.reg_code < 3014) & (warn_info.reg_code > 3008)]


#%% load file
no3_nc = Dataset(no3_path + "NORA3_NorthNorway_Sub_2024.nc")
rs_nc = Dataset(rs_path + "Rain_Snow_NORA3_NorthNorway_Sub_1970.nc")


#%% extract the projection and convert the warn_info file to it
crs = no3_nc.variables["projection_lambert"].proj4
warn_info = warn_info.to_crs(crs)


#%% load lat and lon
lons = no3_nc.variables["longitude"][:]
lats = no3_nc.variables["latitude"][:]

x, y = no3_nc.variables["x"][:], no3_nc.variables["y"][:]
xs, ys = np.meshgrid(x, y)

arr1x = xs.ravel()
arr1y = ys.ravel()


#%% load the variables
ws = no3_nc.variables["wind_speed"][0, 0, :, :].squeeze()
snow = rs_nc.variables["snow"][0, :, :].squeeze()


#%% using the X and Y columns, build a dataframe, then the geodataframe
df1 = pd.DataFrame({'X':arr1x, 'Y':arr1y, "lon":lons.ravel(), "lat":lats.ravel(), "ws":np.ravel(ws),
                    "snow":np.ravel(snow)})
df1['coords'] = list(zip(df1['X'], df1['Y']))
df1['coords'] = df1['coords'].apply(Point)
gdf1 = gpd.GeoDataFrame(df1, geometry='coords')
gdf1 = gdf1.set_crs(crs)


#%%  try to find the intersect of the NORA3 grid with one of the avalanche region shapes
reg_indices = {}
for reg_code in regions.keys():
    df_join = gpd.sjoin(left_df=warn_info[warn_info["reg_code"] == reg_code], right_df=gdf1, how="left",
                        predicate="intersects")


    #% try to convert this dataframe to one that can be plotted
    df2 = pd.DataFrame({'X':np.array(df_join.X), 'Y':np.array(df_join.Y), "lon":np.array(df_join.lon),
                        "lat":np.array(df_join.lat), "ws":np.array(df_join.ws)})

    df2['coords'] = list(zip(df2['X'], df2['Y']))
    df2['coords'] = df2['coords'].apply(Point)
    gdf2 = gpd.GeoDataFrame(df2, geometry='coords')
    gdf2 = gdf2.set_crs(crs)


    #% attempt a plot
    gdf2.plot("ws", cmap="Reds", legend=True)
    pl.show()
    pl.close()


    #% get the latitudes and longitudes of the individual region
    lat_intersect = np.intersect1d(df_join.lat, lats.ravel(), return_indices=True)
    # lon_intersect = np.intersect1d(df_join.lon, lons.ravel(), return_indices=True)

    # --> note that these are the same for lat and lon but you will only really see that when you sort th indices, i.e.:
    #     np.sort(lat_intersect[2]) == np.sort(lon_intersect[2])

    reg_indices[reg_code] = np.sort(lat_intersect[2])

# end for reg_code


#%% dump the indices
dump(reg_indices, out_path + "Region_GridCell_Indices_NORA3.joblib")







