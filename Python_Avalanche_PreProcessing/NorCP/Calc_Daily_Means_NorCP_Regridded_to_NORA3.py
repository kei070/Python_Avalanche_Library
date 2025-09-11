#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate monthly means over the NorCP data regriddde to the NORA3 grid.
"""

#%% imports
import os
import sys
import glob
import xarray as xr
from dask.diagnostics import ProgressBar
from netCDF4 import Dataset
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set parameters for the NorCP data path
model = "ERAINT" # sys.argv[1]  # "GFDL-CM3" #  "EC-EARTH"  # EC-EARTH, GFDL
var = "tas"  # sys.argv[2]  # "tas"  # tas, uas, vas, pr, ...
scen = "evaluation" # sys.argv[3]  # "historical"  # historcal, rcp45, rcp85
period = ""  # sys.argv[4]  # ""  # _MC for mid-century, _LC for late-century, empty string for historical

test_load = False


#%% set data path
data_path = f"{path_par}/IMPETUS/NorCP/{model}_{scen}_{var}/{period[1:]}/Regridded_to_NORA3/"
# print(data_path)

#%% list the nc files
f_list = sorted(glob.glob(data_path + "*.nc"))


#%% open the first file to extract the lon-lat info
nc = Dataset(f_list[0])


#%% load the time variable
norcp_time = nc.variables["time"]


#%% extract lat lon info
# lon = nc.variables["lon"][:]
# lat = nc.variables["lat"][:]


#%% open the datasets via xarray

#% write the dataset out as an nc file
out_path = f"{path_par}/IMPETUS/NorCP/Regridded_to_NORA3/Daily_Mean/{model}/"
os.makedirs(out_path, exist_ok=True)
out_fn = f"{var}_Day_{model}_NorCP_NORA3Grid_{scen}{period}.nc"

if not test_load:
    print("\nLoading data...\n")
    norcp_ncs = xr.open_mfdataset(f_list, chunks={'time':'auto'})


    #% calculate monthly means
    with ProgressBar():
        if var == "pr":  # in case of precipitation calculate the sum instead of the mean
            # --> for some reason the sum function does not work; try to do it manually
            norcp_ncs["pr"] = norcp_ncs["pr"] * 3600  # convert to units mm
            # norcp_daily = norcp_ncs.resample(time="1D").sum()
            norcp_daily = norcp_ncs.resample(time="1D").mean()
            norcp_daily = norcp_daily * 24  # convert from mean to sum
        else:
            norcp_daily = norcp_ncs.resample(time="1D").mean()
        # end if else

        print("\nStoring the daily data...\n")
        norcp_daily.to_netcdf(path=out_path + out_fn, mode='w',
                              encoding={'time': {'units':norcp_time.units, 'dtype': 'double'}})
    # end with
# end if


#%% load and look into the new file
# nc_new = Dataset(out_path + out_fn)


#%%
# nc_date = nc_new.variables["time"]
# nc_dates = num2date(nc_date[:], nc_date.units, nc_date.calendar)


#%% load the file to check its fidelity
if test_load:

    # imports
    import pylab as pl

    # load the file
    nc_new = Dataset(out_path + out_fn)

    # lazy-load the variable
    var_nc = nc_new.variables["pr"]

    # load the values
    var = var_nc[:]

    # do some plotting
    fig = pl.figure(figsize=(10, 5))
    ax00 = fig.add_subplot(121)
    ax01 = fig.add_subplot(122)

    ax00.pcolormesh(var[50, :, :])
    ax01.plot(var[:, 50, 50])

    pl.show()
    pl.close()

# end if


