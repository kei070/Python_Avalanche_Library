#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to generate a (linear?) model to infer the surface temperature from 2m temperature, radiation, and potentially
some other quantities like RH etc.
"""

#%% imports
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pylab as pl
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths
# n3_path = "/media/kei070/One Touch/IMPETUS/NORA3/"
# n3_fn = "fc2021030100_003_fp.nc"

n3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
n3_fn = "NORA3_NorthNorway_Sub_2000.nc"


#%% load the file
n3_nc = xr.open_dataset(n3_path + n3_fn)
# n3_nc = Dataset(n3_path + n3_fn)


#%% load some more or less random time searies
"""
i, j = 500, 1000

tas = np.array(n3_nc["air_temperature_2m"][:, 0, j, i])
ts = np.array(n3_nc["air_temperature_0m"][:, 0, j, i])
snlw = np.array(n3_nc["integral_of_surface_net_downward_longwave_flux_wrt_time"][:, 0, j, i])
snsw = np.array(n3_nc["integral_of_surface_net_downward_shortwave_flux_wrt_time"][:, 0, j, i])
rh = np.array(n3_nc["relative_humidity_2m"][:, 0, j, i])
"""

i, j = 50, 50

tas = np.array(n3_nc["air_temperature_2m"][:, 0, j, i])
# ts = np.array(n3_nc["air_temperature_0m"][:, 0, j, i])
snlw = np.array(n3_nc["surface_net_longwave_radiation"][:, 0, j, i])
snsw = np.array(n3_nc["surface_net_shortwave_radiation"][:, 0, j, i])
rh = np.array(n3_nc["relative_humidity_2m"][:, 0, j, i])


#%%
ts = 0.8 * tas + 0.005 * snlw - 0.002 * snlw + 0.1 * rh + 5
