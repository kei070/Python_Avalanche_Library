#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to update the NORA3 subsets with the predicted surface temperature.
"""

#%% imports
import glob
import numpy as np
from netCDF4 import Dataset
from ava_functions.Progressbar import print_progress_bar
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% paths
ts_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Ts_Files/"
n3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"


#%% load the files
list_fn_ts = sorted(glob.glob(ts_path + "*.nc"))
list_fn_n3 = sorted(glob.glob(n3_path + "*.nc"))


#%% loop over and update the files
l = len(list_fn_ts)

count = 0
print()
print_progress_bar(count, l)
for fn_ts, fn_n3 in zip(list_fn_ts[:], list_fn_n3[:]):

    ts = Dataset(fn_ts)["ts"][:]

    n3_nc = Dataset(fn_n3, mode="r+")

    try:
        nc_ts = n3_nc.createVariable("ts", np.float64, ('time', 'y', 'x'))
        nc_ts[:] = ts
    except:
        # print("\nts already exists")
        print_progress_bar(count, l, suffix="ts already exists")
        nc_ts = n3_nc.variables["ts"]
    # end try except

    nc_ts.description = "Surface temperature as derived with a linear regression model that was generated from ERA5 " +\
                        "data using at2m, wind speed, short- and long-wave radiation at surface."

    nc_ts.long_name = "derived_surface_temperature"

    nc_ts.units = "K"

    n3_nc.close()

    count += 1

    print_progress_bar(count, l, suffix="                 ")
# end for fn_ts, fn_n3
