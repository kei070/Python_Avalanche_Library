#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the gridded wind predictors.
"""

#%% imports
import os
import glob
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from .Lists_and_Dictionaries.Variable_Name_NORA3_NorCP import pred_names


#%% define the function
def gen_daily_pred(data_path, out_path, fn, var_name, var_dic,
                   agg="mean", yrs=np.arange(2016, 2024+1, 1), exist_ok=False):

    # set the variable name
    varn = var_dic[var_name]

    # perform checks if the outputs exist already
    mean_exists = False
    min_exists = False
    max_exists = False
    if agg == "all":
        if os.path.exists(out_path + f"{fn}_{pred_names[var_name]['mean']}1.nc"):
            mean_exists = True
        if os.path.exists(out_path + f"{fn}_{pred_names[var_name]['max']}1.nc"):
            max_exists = True
        if os.path.exists(out_path + f"{fn}_{pred_names[var_name]['min']}1.nc"):
            min_exists = True
        # end if
    else:
        if os.path.exists(out_path + f"{fn}_{pred_names[var_name][agg]}1.nc"):
            mean_exists = True
    # end if else

    if (mean_exists & max_exists & min_exists) & (not exist_ok):
        print(f"\nAll {var_name} 1-day predictors exits. Aborting {var_name} calculations.\n")
        return
    # end if

    fn_list = []
    # f_list = []
    for yr in yrs:
        fn_list.append(glob.glob(data_path + f"*{yr}*.nc")[0])
        # f_list.append(xr.open_dataset(glob.glob(data_path + f"*{yr}*.nc")[0])[varn].squeeze())
    # end for yr

    #% load the dataset
    x_1h = xr.open_mfdataset(fn_list, combine="by_coords", data_vars="minimal")[varn].squeeze()
    with ProgressBar():
        # x_1h = xr.merge(f_list)
        print("Data loaded. Proceeding to calculations...\n")

        #% generate daily means
        if agg == "mean":
            if (not mean_exists) | exist_ok:
                x1 = x_1h.resample(time="1D").mean()
                x1 = x1.rename(pred_names[var_name][agg] + "1")
                print("\n1-day calculation started. Storing...\n")
                x1.to_netcdf(out_path + f"{fn}_{pred_names[var_name][agg]}1.nc")
            else:
                print(f"\n{pred_names[var_name][agg]}1 exists. Continuing with next...\n")
            # end if else
        elif agg == "max":
            if (not max_exists) | exist_ok:
                max1 = x_1h.resample(time="1D").max()
                max1 = max1.rename(pred_names[var_name][agg] + "1")
                print("\n1-day calculation started. Storing...\n")
                max1.to_netcdf(out_path + f"{fn}_{pred_names[var_name][agg]}1.nc")
            else:
                print(f"\n{pred_names[var_name][agg]}1 exists. Continuing with next...\n")
            # end if else
        elif agg == "min":
            if (not min_exists) | exist_ok:
                min1 = x_1h.resample(time="1D").min()
                min1 = min1.rename(pred_names[var_name][agg] + "1")
                print("\n1-day calculation started. Storing...\n")
                min1.to_netcdf(out_path + f"{fn}_{pred_names[var_name][agg]}1.nc")
            else:
                print(f"\n{pred_names[var_name][agg]}1 exists. Continuing with next...\n")
            # end if else
        elif agg == "all":
            if (not mean_exists) | exist_ok:
                x1 = x_1h.resample(time="1D").mean()
                x1 = x1.rename(pred_names[var_name]["mean"] + "1")
                print("\n1-day calculation mean started. Storing...\n")
                x1.to_netcdf(out_path + f"{fn}_{pred_names[var_name]['mean']}1.nc")
            else:
                print(f"\n{pred_names[var_name]['mean']}1 exists. Continuing with next...\n")
            # end if else
            if (not max_exists) | exist_ok:
                max1 = x_1h.resample(time="1D").max()
                max1 = max1.rename(pred_names[var_name]["max"] + "1")
                print("\n1-day calculation max started. Storing...\n")
                max1.to_netcdf(out_path + f"{fn}_{pred_names[var_name]['max']}1.nc")
            else:
                print(f"\n{pred_names[var_name]['max']}1 exists. Continuing with next...\n")
            # end if else
            if (not min_exists) | exist_ok:
                min1 = x_1h.resample(time="1D").min()
                min1 = min1.rename(pred_names[var_name]["min"] + "1")
                print("\n1-day calculation min started. Storing...\n")
                min1.to_netcdf(out_path + f"{fn}_{pred_names[var_name]['min']}1.nc")
            else:
                print(f"\n{pred_names[var_name]['max']}1 exists. Continuing with next...\n")
            # end if else
        # end if elif
    # end with
# end def


def gen_pred_roll(data_path, var_name, fn, roll=[3, 7], agg="mean", exist_ok=False):

    # open the 1-day dataset
    if agg == "all":
        x1 = xr.open_mfdataset(data_path +
                               f"{fn}_{pred_names[var_name]['mean']}1.nc")[f"{pred_names[var_name]['mean']}1"].squeeze()
        min1 = xr.open_mfdataset(data_path +
                                 f"{fn}_{pred_names[var_name]['min']}1.nc")[f"{pred_names[var_name]['min']}1"].squeeze()
        max1 = xr.open_mfdataset(data_path +
                                 f"{fn}_{pred_names[var_name]['max']}1.nc")[f"{pred_names[var_name]['max']}1"].squeeze()
    if agg == "mean":
        x1 = xr.open_mfdataset(data_path +
                               f"{fn}_{pred_names[var_name][agg]}1.nc")[f"{pred_names[var_name][agg]}1"].squeeze()
    if agg == "min":
        min1 = xr.open_mfdataset(data_path +
                                 f"{fn}_{pred_names[var_name][agg]}1.nc")[f"{pred_names[var_name][agg]}1"].squeeze()
    if agg == "max":
        max1 = xr.open_mfdataset(data_path +
                                 f"{fn}_{pred_names[var_name][agg]}1.nc")[f"{pred_names[var_name][agg]}1"].squeeze()
    # end if

    #% calculate the 3-day means
    ndays_l = roll

    for ndays in ndays_l:

        # perfrom checks if the outputs exist already
        mean_exists = False
        min_exists = False
        max_exists = False
        if agg == "all":
            if os.path.exists(data_path + f"{fn}_{pred_names[var_name]['mean']}{ndays}.nc"):
                mean_exists = True
            if os.path.exists(data_path + f"{fn}_{pred_names[var_name]['max']}{ndays}.nc"):
                max_exists = True
            if os.path.exists(data_path + f"{fn}_{pred_names[var_name]['min']}{ndays}.nc"):
                min_exists = True
            # end if
        else:
            if os.path.exists(data_path + f"{fn}_{pred_names[var_name][agg]}{ndays}.nc"):
                min_exists = True
            # end if
        # end if else
        if (mean_exists & max_exists & min_exists) & (not exist_ok):
            print(f"\nAll {var_name} {ndays}-day predictors exits. Continuing with next...\n")
            continue
        # end if

        if agg == "mean":
            if (not mean_exists) | exist_ok:
                x_x = x1.rolling(time=ndays, center=False).mean()
                x_x = x_x.rename(f"{pred_names[var_name][agg]}{ndays}")
                print(f"\n{ndays}-days calculation started. Storing...\n")
                x_x.to_netcdf(data_path + f"{fn}_{pred_names[var_name][agg]}{ndays}.nc")
            else:
                print(f"\n{pred_names[var_name][agg]}{ndays} exists. Continuing with next...\n")
            # end if else
        elif agg == "max":
            if (not max_exists) | exist_ok:
                x_x = max1.rolling(time=ndays, center=False).max()
                x_x = x_x.rename(f"{pred_names[var_name][agg]}{ndays}")
                print(f"\n{ndays}-days calculation started. Storing...\n")
                x_x.to_netcdf(data_path + f"{fn}_{pred_names[var_name][agg]}{ndays}.nc")
            else:
                print(f"\n{pred_names[var_name][agg]}{ndays} exists. Continuing with next...\n")
            # end if else
        elif agg == "min":
            if (not min_exists) | exist_ok:
                x_x = min1.rolling(time=ndays, center=False).min()
                x_x = x_x.rename(f"{pred_names[var_name][agg]}{ndays}")
                print(f"\n{ndays}-days calculation started. Storing...\n")
                x_x.to_netcdf(data_path + f"{fn}_{pred_names[var_name][agg]}{ndays}.nc")
            else:
                print(f"\n{pred_names[var_name][agg]}{ndays} exists. Continuing with next...\n")
            # end if else
        elif agg == "all":
            if (not mean_exists) | exist_ok:
                x_x = x1.rolling(time=ndays, center=False).mean()
                x_x = x_x.rename(f"{pred_names[var_name]['mean']}{ndays}")
                print(f"\n{ndays}-days mean calculation started. Storing...\n")
                x_x.to_netcdf(data_path + f"{fn}_{pred_names[var_name]['mean']}{ndays}.nc")
            else:
                print(f"\n{pred_names[var_name]['mean']}{ndays} exists. Continuing with next...\n")
            # end if else
            if (not max_exists) | exist_ok:
                x_x = max1.rolling(time=ndays, center=False).max()
                x_x = x_x.rename(f"{pred_names[var_name]['max']}{ndays}")
                print(f"\n{ndays}-days max calculation started. Storing...\n")
                x_x.to_netcdf(data_path + f"{fn}_{pred_names[var_name]['max']}{ndays}.nc")
            else:
                print(f"\n{pred_names[var_name]['max']}{ndays} exists. Continuing with next...\n")
            # end if else
            if (not min_exists) | exist_ok:
                x_x = min1.rolling(time=ndays, center=False).min()
                x_x = x_x.rename(f"{pred_names[var_name]['min']}{ndays}")
                print(f"\n{ndays}-days min calculation started. Storing...\n")
                x_x.to_netcdf(data_path + f"{fn}_{pred_names[var_name]['min']}{ndays}.nc")
            else:
                print(f"\n{pred_names[var_name]['min']}{ndays} exists. Continuing with next...\n")
            # end if else
        # end if elif
    # end for ndays
# end def

