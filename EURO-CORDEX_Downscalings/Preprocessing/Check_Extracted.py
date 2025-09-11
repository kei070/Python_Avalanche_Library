#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:53:01 2025

@author: kei070
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the extracted northern-Norwegian subsets from the EURO-CORDEX data. The extraction was performed with
Extract_EURO-CORDEX.py
"""


#%% imports
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
scen = "hist"
model = "MPI_RCA"

var1 = "SWE"
var2 = "RR"

data_path = f"{path_par3}/IMPETUS/EURO-CORDEX/{scen}/{model}/"


#%% set the variable name dictionary
var_ns = {"TM":"air_temperature__map_hist_daily", "TX":"max_air_temperature__map_hist_daily",
          "TN":"min_air_temperature__map_hist_daily",
          "RR":"precipitation__map_hist_daily",
          "SWE":"snow_water_equivalent__map_hist_daily"}


#%% file names
fn1 = f"MPI_RCA_{var1}_daily_1971-2000_v4.nc"
fn2 = f"MPI_RCA_{var2}_daily_1971-2000_v4.nc"


#%% load the files
nc1 = xr.open_dataset(data_path + fn1)
nc2 = xr.open_dataset(data_path + fn2)


#%% do some plotting
fig = pl.figure(figsize=(10, 6))
ax00 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)

p00 = ax00.pcolormesh(nc1[var_ns[var1]][10, :, :])
fig.colorbar(p00, ax=ax00)

p01 = ax01.pcolormesh(nc2[var_ns[var2]][10, :, :])
fig.colorbar(p00, ax=ax01)

pl.show()
pl.close()