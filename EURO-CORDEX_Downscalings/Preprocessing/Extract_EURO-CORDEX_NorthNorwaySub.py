#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a northern-Norwegian subset from the EURO-CORDEX data. Indices were found in First_Look_EURO-CORDEX.py.
"""


#%% imports
import sys
import glob
import numpy as np
import xarray as xr
import geopandas as gpd
import pylab as pl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib import gridspec
from ava_functions.Lists_and_Dictionaries.Paths import path_par3
from dask.diagnostics import ProgressBar


#%% set paths
model = sys.argv[1]  # "MPI_RCA"
scen = sys.argv[2]  # "rcp85"
var = sys.argv[3]  # "RR"

data_path = f"{path_par3}/IMPETUS/EURO-CORDEX/{scen}/{model}/{var}/"
out_path = f"{path_par3}/IMPETUS/EURO-CORDEX/{scen}/{model}/"


#%% set the variable name dictionary
var_ns = {"TM":f"air_temperature__map_{scen}_daily", "TX":f"max_air_temperature__map_{scen}_daily",
          "TN":f"min_air_temperature__map_{scen}_daily",
          "RR":f"precipitation__map_{scen}_daily",
          "SWE":f"snow_water_equivalent__map_{scen}_daily"}


#%% file list
f_l = glob.glob(data_path + "*.nc")


#%% load the files
with ProgressBar():
    nc = xr.open_mfdataset(f_l)


    #% extract the years
    sta_yr = f_l[0].split("_")[-2]
    end_yr = f_l[-1].split("_")[-2]


    #% set indices
    x1, x2 = 600, 890
    y1, y2 = 190, 420


    #% select the indices
    nc_sub = nc.isel(Xc=slice(x1, x2), Yc=slice(y1, y2))


    #% fill the NA values
    if var in ["TM"]:
        nc_sub = nc_sub.interpolate_na(dim="Xc", method="nearest")
        nc_sub[var_ns[var]] = nc_sub[var_ns[var]].fillna(2731.5)
        nc_sub[var_ns[var]] = nc_sub[var_ns[var]] * 0.1
    if var in ["TN", "TX"]:
        nc_sub = nc_sub.interpolate_na(dim="Xc", method="nearest")
        nc_sub[var_ns[var]] = nc_sub[var_ns[var]].fillna(273.15)
    if var in ["SWE"]:
        nc_sub[var_ns[var]] = nc_sub[var_ns[var]].fillna(0)
    if var in ["RR"]:
        nc_sub[var_ns[var]] = nc_sub[var_ns[var]].fillna(0)
        nc_sub[var_ns[var]] = nc_sub[var_ns[var]] * 0.1
    # end if


    #% store
    nc_sub.to_netcdf(out_path + f"{model}_{var}_daily_{sta_yr}-{end_yr}_v4.nc")

# end with

