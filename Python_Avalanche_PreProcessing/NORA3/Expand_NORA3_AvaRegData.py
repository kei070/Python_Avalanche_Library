#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOT FUNCTIONAL!
Expand the existing avalanche region data with newly available files.
"""


#%% imports
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set parameters
s_yr = 2024
e_yr = 2024
h_low = 400
h_hi = 900

region = "Tromsoe"


#%% paths
data_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m/"
new_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m/{s_yr}-{e_yr}/"
fn = f"NORA3_{region}_Between{h_low}_and_{h_hi}m.nc"
new_n = f"NORA3_{region}_{s_yr}-{e_yr}.nc"


#%% load the data
data = xr.open_dataset(data_path + fn)

data_add = xr.open_dataset(new_path + new_n)


#%% merge
with ProgressBar():
    data_temp = xr.merge([data, data_add])
# end with


#%% loop over the variables to expand them
"""
for var in ["air_temperature_2m"]:  # data_add.variables:
    if np.ndim(data_add[var]) < 2:
        # jump over the coordinate variables (time, loc)
        continue
    # end if
    print(var)

    data_temp = xr.merge([data[var], data_add[var]])

# end for var
"""