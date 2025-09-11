#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the gridded wind-drift predictors.
"""

#%% imports
import os
import xarray as xr


#%% define the function
def gen_wdrift(grid_path, fn, exist_ok=False):

    pred_exist = False
    if os.path.exists(grid_path + f"{fn}_wdrift.nc"):
        pred_exist = True
    # end if

    if pred_exist & (not exist_ok):
        print("\nwdrift predictors exist. Continuing with next...\n")
        return
    # end if

    #% load the w_mean, w3, s1, and s3 predictors
    w1 = xr.open_dataset(grid_path + f"{fn}_w1.nc")
    w3 = xr.open_dataset(grid_path + f"{fn}_w3.nc")
    s1 = xr.open_dataset(grid_path + f"{fn}_s1.nc")
    s3 = xr.open_dataset(grid_path + f"{fn}_s3.nc")


    #% calculate the wind-drift parameters
    wdrift = w1.w1.squeeze() * s1.s1.squeeze()
    wdrift3 = w1.w1.squeeze()**3 * s1.s1.squeeze()

    wdrift_3 = w3.w3.squeeze() * s3.s3.squeeze()
    wdrift3_3 = w3.w3.squeeze()**3 * s3.s3.squeeze()


    #% merge the parameters
    ds_dict = {"wdrift":wdrift, "wdrift3":wdrift3, "wdrift_3":wdrift_3, "wdrift3_3":wdrift3_3}
    ds = xr.Dataset(ds_dict)


    #% store the file
    ds.to_netcdf(grid_path + f"{fn}_wdrift.nc")

# end def