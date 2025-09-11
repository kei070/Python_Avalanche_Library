#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:12:49 2025

@author: kei070
"""
#%% imports
import geopandas as gpd
import cartopy.crs as ccrs
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% store
store_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
warn_info = gpd.read_file(store_path + "Warning_Region_Info.gpkg", driver="GPKG")


#%% reproject the shape
orth_crs = ccrs.Orthographic(central_latitude=69, central_longitude=20)
warn_info = warn_info.to_crs(orth_crs)


#%% calculate the area of the regions
warn_info["area_new"] = warn_info["geometry"].area * 1e-6