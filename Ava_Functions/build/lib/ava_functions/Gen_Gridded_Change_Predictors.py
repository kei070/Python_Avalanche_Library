#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate some change predictors (temperature range, wind-speed variation, ...).
"""


#%% imports
import os
import xarray as xr


#%% define the script as a function
def gen_change_preds(grid_path, fn, exist_ok=False):

    pred_exist = False
    if os.path.exists(grid_path + f"{fn}_dt_dws.nc"):
        pred_exist = True
    # end if

    if pred_exist & (not exist_ok):
        print("\nChange predictors exist. Continuing with next...\n")
        return
    # end if

    #% load the w_mean, w3, s1, and s3 predictors
    tmax1 = xr.open_dataset(grid_path + f"{fn}_tmax1.nc")
    tmin1 = xr.open_dataset(grid_path + f"{fn}_tmin1.nc")

    tmax3 = xr.open_dataset(grid_path + f"{fn}_tmax3.nc")
    tmin3 = xr.open_dataset(grid_path + f"{fn}_tmin3.nc")

    tmax7 = xr.open_dataset(grid_path + f"{fn}_tmax7.nc")
    tmin7 = xr.open_dataset(grid_path + f"{fn}_tmin7.nc")

    wmax1 = xr.open_dataset(grid_path + f"{fn}_wmax1.nc")
    wmin1 = xr.open_dataset(grid_path + f"{fn}_wmin1.nc")

    wmax3 = xr.open_dataset(grid_path + f"{fn}_wmax3.nc")
    wmin3 = xr.open_dataset(grid_path + f"{fn}_wmin3.nc")


    #% calculate the variation predictors
    t1_range = tmax1.tmax1.squeeze() - tmin1.tmin1.squeeze()
    t3_range = tmax3.tmax3.squeeze() - tmin3.tmin3.squeeze()
    t7_range = tmax7.tmax7.squeeze() - tmin7.tmin7.squeeze()

    w1_range = wmax1.wmax1.squeeze() - wmin1.wmin1.squeeze()
    w3_range = wmax3.wmax3.squeeze() - wmin3.wmin3.squeeze()


    #% calculate the variation ON THE DAY before or two days before the event
    dtr1 = t1_range.shift(time=1)
    dtr2 = t1_range.shift(time=2)

    dws2 = w1_range.shift(time=1)
    dws3 = w1_range.shift(time=2)


    #% rename the variables
    t1_range = t1_range.rename("dtr1")
    t3_range = t3_range.rename("dtrd3")
    t7_range = t7_range.rename("dtrd7")

    w1_range = w1_range.rename("dws1")
    w3_range = w3_range.rename("dwsd3")

    dtr2 = dtr1.rename("dtr2")
    dtr3 = dtr2.rename("dtr3")

    dws2 = dws2.rename("dws2")
    dws3 = dws3.rename("dws3")


    #% merge them into a data set
    ds = xr.merge([t1_range, t3_range, t7_range, w1_range, w3_range, dtr2, dtr3, dws2, dws3])


    #% store the data set
    ds.to_netcdf(grid_path + f"{fn}_dt_dws.nc")

# end def



