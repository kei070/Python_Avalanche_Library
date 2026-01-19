#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the gridded precipitation predictors.
--> This is based on the files generated with Calc_r1_s1_Gridded.py
"""


#%% imports
import os
import glob
import numpy as np
import xarray as xr


#%% define the function
def gen_precip_daily(data_path, out_path, fn, var_dic, yrs=np.arange(2016, 2024+1, 1), exist_ok=False):

    # perform checks if the outputs exist already
    r1_exists = False
    s1_exists = False
    if os.path.exists(out_path + f"{fn}_r1.nc"):
        r1_exists = True
    if os.path.exists(out_path + f"{fn}_s1.nc"):
        s1_exists = True
    # end if

    if (r1_exists & s1_exists) & (not exist_ok):
        print("\nr1 and s1 exist. Stopping those calculations and returning from function.\n")
        return
    # end if

    # set the variable name
    s_n = var_dic["snow"]
    r_n = var_dic["rain"]

    fn_list = []
    for yr in yrs:
        fn_list.append(glob.glob(data_path + f"*_{yr}*.nc")[0])
    # end for yr

    #% load the dataset
    rain_1h = xr.open_mfdataset(fn_list)[r_n].squeeze()
    snow_1h = xr.open_mfdataset(fn_list)[s_n].squeeze()
    print("Data loaded. Proceeding to calculations...\n")

    #% generate daily sums
    r1 = rain_1h.resample(time="1D").sum()
    r1 = r1.rename("r1")
    s1 = snow_1h.resample(time="1D").sum()
    s1 = s1.rename("s1")
    print("\n1-day calculation started. Storing...\n")
    r1.to_netcdf(out_path + f"{fn}_r1.nc")
    s1.to_netcdf(out_path + f"{fn}_s1.nc")

# end def


def gen_precip_roll(data_path, fn, var_dic, roll=[3, 7], exist_ok=False):

    r1 = xr.open_mfdataset(data_path + f"{fn}_r1.nc")["r1"].squeeze()
    s1 = xr.open_mfdataset(data_path + f"{fn}_s1.nc")["s1"].squeeze()

    #% calculate the 3 and 7-day means
    ndays_l = roll

    for ndays in ndays_l:
        # perform checks if the outputs exist already
        r_exists = False
        s_exists = False
        if os.path.exists(data_path + f"{fn}_r{ndays}.nc"):
            r_exists = True
        if os.path.exists(data_path + f"{fn}_s{ndays}.nc"):
            s_exists = True
        # end if

        if (not r_exists) | exist_ok:
            r_x = r1.rolling(time=ndays, center=False).sum()
            r_x = r_x.rename(f"r{ndays}")
            print(f"\n{ndays}-days for rain calculation started. Storing...\n")
            r_x.to_netcdf(data_path + f"{fn}_r{ndays}.nc")
        else:
            print(f"\nr{ndays} exists. Continuing with next...\n")
        if (not s_exists) | exist_ok:
            s_x = s1.rolling(time=ndays, center=False).sum()
            s_x = s_x.rename(f"s{ndays}")
            print(f"\n{ndays}-days for snow calculation started. Storing...\n")
            s_x.to_netcdf(data_path + f"{fn}_s{ndays}.nc")
        else:
            print(f"\ns{ndays} exists. Continuing with next...\n")
        # end if else
    # end for ndays

# end def