#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the daily maxs of NORA3.
"""

#%% imports
import os
import sys
import glob
import xarray as xr

from ava_functions.Progressbar import print_progress_bar
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set path
data_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
out_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files_DailyMax/"


#%% load file names
fn_list = sorted(glob.glob(data_path + "*.nc"))

fn = fn_list[-1]
nc = xr.open_dataset(fn)

# sys.exit()


#%% iterate over the file names, open the file and calculate the daily means, and store the resulting daily mean file
l = len(fn_list)

print_progress_bar(0, l)
for i, fn in enumerate(fn_list):

    # extract the year from the file name
    yr = fn[-7:-3]
    suff = f"processing {yr}"
    print_progress_bar(i, l, suffix=f"{suff:20}")

    # load the file
    nc = xr.open_dataset(fn)

    # calculate daily means
    nc_daily = nc.resample(time="1D").max()

    # calculate the daily precipitation sum
    nc_daily["precipitation_amount_hourly"] = nc_daily["precipitation_amount_hourly"] * 24

    # store the daily means
    nc_daily.to_netcdf(path=out_path + f"NORA3_NorthNorway_Sub_DailyMax_{yr}.nc", mode='w')

    # print(f"NORA3_NorthNorway_Sub_DailyMean_{yr}.nc produced.")
    suff = f"{yr} done"
    print_progress_bar(i+1, l, suffix=f"{suff:20}")

# end for fn