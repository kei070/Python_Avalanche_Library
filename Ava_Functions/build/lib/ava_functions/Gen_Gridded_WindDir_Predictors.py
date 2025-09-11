#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the gridded wind direction predictors. --> Still somewhat unsure about this, does the standard deviation make
sense, and even the rolling standard deviation?
"""

#%% imports
import os
import glob
import numpy as np
import xarray as xr


#%% define the function
def gen_wdir_daily(data_path, out_path, fn, var_dic, agg="mean", yrs=np.arange(2017, 2024, 1), exist_ok=False):

    # perform checks if the outputs exist already
    wdir_exists = False
    dwdir_exists = False
    if os.path.exists(out_path + f"{fn}_wdir1.nc"):
        wdir_exists = True
    if os.path.exists(out_path + f"{fn}_dwdir1.nc"):
        dwdir_exists = True
    # end if

    if (wdir_exists & dwdir_exists) & (not exist_ok):
        print("\nBoth wdir1 and dwdir1 exist. Continuing with next...\n")
    # end if

    # set the variable name
    varn = var_dic["wdir"]

    fn_list = []
    for yr in yrs:
        fn_list.append(glob.glob(data_path + f"*{yr}*.nc")[0])
    # end for yr

    #% load the dataset
    wdir_1h = xr.open_mfdataset(fn_list)[varn].squeeze()
    print("Data loaded. Proceeding to calculations...\n")

    #% generate daily means
    if (not wdir_exists) | exist_ok:
        wdir1 = wdir_1h.resample(time="1D").mean()
        wdir1 = wdir1.rename("wdir1")
        print("Storing wdir1...\n")
        wdir1.to_netcdf(out_path + f"{fn}_wdir1.nc")
    else:
        print("\nwdir1 exists. Continuning with next...\n")
    if (not dwdir_exists) | exist_ok:
        dwdir1 = wdir_1h.resample(time="1D").std()
        dwdir1 = dwdir1.rename("dwdir1")
        print("Storing dwdir1...\n")
        dwdir1.to_netcdf(out_path + f"{fn}_dwdir1.nc")  # std of the wind direction the day before the event
    else:
        print("\ndwdir1 exists. Continuning with next...\n")
    # end if else

# end def


def gen_wdir_shift(data_path, fn, var_dic, shift=[2, 3], exist_ok=False):

    dwdir1 = xr.open_mfdataset(data_path + f"{fn}_dwdir1.nc")["dwdir1"].squeeze()

    for ndays in shift:
        # perform checks if the outputs exist already
        dwdir_exists = False
        if os.path.exists(data_path + f"{fn}_dwdir{ndays}.nc"):
            dwdir_exists = True
        # end if
        if (not dwdir_exists) | exist_ok:
            dwdirx = dwdir1.shift(time=ndays)
            dwdirx = dwdirx.rename(f"dwdir{ndays}")
            dwdirx.to_netcdf(data_path + f"{fn}_dwdir{ndays}.nc")
        else:
            print(f"\ndwdir{ndays} exists. Continuing with next...\n")
    # end for ndays

# end def


def gen_wdir_roll(data_path, fn, var_dic, roll=[2, 3], exist_ok=False):

    wdir1 = xr.open_mfdataset(data_path + f"{fn}_wdir1.nc")["wdir1"].squeeze()

    #% calculate the 2- and 3-day means
    ndays_l = [2, 3]

    for ndays in ndays_l:
        # perform checks if the outputs exist already
        dwdir_exists = False
        if os.path.exists(data_path + f"{fn}_dwdird{ndays}.nc"):
            dwdir_exists = True
        # end if

        if (not dwdir_exists) | exist_ok:
            dwdir_x = wdir1.rolling(time=ndays, center=False).std()
            dwdir_x = dwdir_x.rename(f"dwdird{ndays}")
            print(f"\n{ndays}-days calculation started. Storing...\n")
            dwdir_x.to_netcdf(data_path + f"{fn}_dwdird{ndays}.nc")
        else:
            print(f"\ndwdird{ndays} exists. Continuing with next...\n")
    # end for ndays

    wdir3_exists = False
    if os.path.exists(data_path + f"{fn}_wdir3.nc"):
        wdir3_exists = True
    # end if

    if (not wdir3_exists) | exist_ok:
        wdir3 = wdir1.rolling(time=3, center=False).mean()
        wdir3 = wdir3.rename("wdir3")
        print("\n3-days calculation started. Storing...\n")
        wdir3.to_netcdf(data_path + f"{fn}_wdir3.nc")
    # end if

# end def
