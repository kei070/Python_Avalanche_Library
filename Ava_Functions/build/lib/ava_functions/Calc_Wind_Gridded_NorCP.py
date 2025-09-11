#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the gridded wind speed and direction from NorCP uas and vas data.
"""


#%% imports
import os
import sys
import glob
import xarray as xr
from datetime import date
from netCDF4 import Dataset

"""
from Lists_and_Dictionaries.Paths import path_par
from Functions.Func_Progressbar import print_progress_bar
from Functions.Func_Wind_Dir_from_U_and_V import wind_dir
"""

from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Progressbar import print_progress_bar
from ava_functions.Calc_Preds import wind_dir


#%% define the function
def calc_wspeed_dir_ncp(params=[]):

    model, scen, period = params

    #% set paths
    uas_path = f"/{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}_uas/{period}/Regridded_to_NORA3/"
    vas_path = f"/{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}_vas/{period}/Regridded_to_NORA3/"
    out_path = f"/{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}_Wind_Speed_Dir/{period}/Regridded_to_NORA3/"


    #% generate the directory
    os.makedirs(out_path, exist_ok=True)


    #% load the files
    fns_uas = sorted(glob.glob(uas_path + "*.nc"))
    fns_vas = sorted(glob.glob(vas_path + "*.nc"))


    #% prepare progressbar
    l = len(fns_uas)

    #% load the necessary files
    print(f"\nGenerating wind speed and direction for NorCP {model} {scen} {period}...\n")
    print_progress_bar(0, l)
    count = 0
    for fn_uas, fn_vas in zip(fns_uas, fns_vas):

        #% load the datasets
        nc = Dataset(fn_uas)
        uas_nc = xr.open_dataset(fn_uas)
        vas_nc = xr.open_dataset(fn_vas)

        # check that both files are from the same year
        if int(uas_nc.time.dt.year[0]) != int(vas_nc.time.dt.year[0]):
            print("\nU and V wind files not from the same year. Aborting...\n")
            sys.exit()
        # end if

        # load the values
        uas_1h = uas_nc["uas"].values.squeeze()
        vas_1h = vas_nc["vas"].values.squeeze()

        # print("Data loaded. Proceeding to calculations...\n")


        #% file name
        out_name = "Wind_Speed_Dir_" + "_".join(fn_uas.split("/")[-1].split("_")[1:])


        #% calculate wind speed and direction
        wspeed = (uas_1h**2 + vas_1h**2)**0.5
        wdir = wind_dir(uas_1h, vas_1h)


        #% load the dimensions
        x = nc.variables["x"]
        y = nc.variables["y"]

        # longitude = nc.variables["longitude"]
        # latitude = nc.variables["latitude"]


        # store r1 and s1 as netcdf
        ds = Dataset(out_path + out_name, 'w', format='NETCDF4')

        # define the dimensions of the data
        time = ds.createDimension('time', None)
        x_dim = ds.createDimension('x', len(x))
        y_dim = ds.createDimension('y', len(y))

        # create coordinate variables for 3 dimensions
        times = ds.createVariable('time', 'f4', ('time',))
        x_var = ds.createVariable('x', 'f4', ('x',))
        y_var = ds.createVariable('y', 'f4', ('y',))

        # latitudes = ds.createVariable('latitude', 'f4', ('y', 'x'))
        # longitudes = ds.createVariable('longitude', 'f4', ('y', 'x'))

        # create the actual variables
        wspeed_nc = ds.createVariable('wspeed', 'f4', ('time', 'y', 'x'))
        wdir_nc = ds.createVariable('wdir', 'f4', ('time', 'y', 'x'))

        # fill the variables
        times[:] = nc.variables["time"][:]
        x_var[:] = x[:]
        y_var[:] = y[:]
        # latitudes[:] = latitude[:]
        # longitudes[:] = longitude[:]

        wspeed_nc[:] = wspeed
        wdir_nc[:] = wdir

        # Add global attributes
        ds.description = f"""Wind speed and direction data generated from NorCP {model} {scen}."""
        ds.history = 'Created ' + str(date.today())

        # Add local attributes to variables
        times.units = nc.variables["time"].units
        times.calendar = nc.variables["time"].calendar

        x_var.units = x.units
        y_var.units = y.units
        # latitudes.units = latitude.units
        # longitudes.units = longitude.units
        wspeed_nc.units = 'm/s'
        wdir_nc.units = 'degree'

        ds.close()

        count +=1
        print_progress_bar(count, l)

    # end for
# end def


#%% execute
# params = ["EC-Earth", "rcp45", "LC"]
# calc_rain_snow_ncp(params=params)




