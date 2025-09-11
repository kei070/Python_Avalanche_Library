#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the gridded rain and snow parameters (from which r1, 3, 7 and s1, 3, 7 can be easily generated).
"""


#%% imports
import glob
import numpy as np
import xarray as xr
from datetime import date
from netCDF4 import Dataset

from .Lists_and_Dictionaries.Paths import path_par


#%% define the function
def calc_rain_snow(sta_yr=2016, end_yr=2025):
    #% set paths
    nora3_path = f"/{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
    out_path = f"/{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Rain_and_Snow/"


    #% load the necessary files
    yrs = np.arange(sta_yr, end_yr, 1)
    for yr in yrs:
        fn = glob.glob(nora3_path + f"*{yr}*.nc")[0]


        #% load the dataset
        # t2m_1h = xr.open_mfdataset(fn_list)["air_temperature_2m"].values().squeeze()
        # prec_1h = xr.open_mfdataset(fn_list)["precipitation_amount_hourly"].values().squeeze()
        nc = Dataset(fn)

        t2m_1h = xr.open_dataset(fn)["air_temperature_2m"].values.squeeze()
        prec_1h = xr.open_dataset(fn)["precipitation_amount_hourly"].values.squeeze()
        print(f"Data for {yr} loaded. Proceeding to calculations...\n")


        #% generate s1 and r1 arrays
        s1 = np.zeros(np.shape(t2m_1h))
        r1 = np.zeros(np.shape(t2m_1h))


        #% categorise the precipitation
        s1[t2m_1h <= 273.15] = prec_1h[t2m_1h <= 273.15]
        r1[t2m_1h > 273.15] = prec_1h[t2m_1h > 273.15]


        #% load the dimensions
        x = nc.variables["x"]
        y = nc.variables["y"]

        longitude = nc.variables["longitude"]
        latitude = nc.variables["latitude"]


        #% store r1 and s1 as netcdf
        ds = Dataset(out_path + f'Rain_Snow_NORA3_NorthNorway_Sub_{yr}.nc', 'w', format='NETCDF4')

        # define the dimensions of the data
        time = ds.createDimension('time', None)
        x_dim = ds.createDimension('x', len(x))
        y_dim = ds.createDimension('y', len(y))

        # create coordinate variables for 3 dimensions
        times = ds.createVariable('time', 'f4', ('time',))
        x_var = ds.createVariable('x', 'f4', ('x',))
        y_var = ds.createVariable('y', 'f4', ('y',))

        latitudes = ds.createVariable('latitude', 'f4', ('y', 'x'))
        longitudes = ds.createVariable('longitude', 'f4', ('y', 'x'))

        # create the actual variables
        r1_nc = ds.createVariable('rain', 'f4', ('time', 'y', 'x'))
        s1_nc = ds.createVariable('snow', 'f4', ('time', 'y', 'x'))

        # fill the variables
        times[:] = nc.variables["time"][:]
        x_var[:] = x[:]
        y_var[:] = y[:]
        latitudes[:] = latitude[:]
        longitudes[:] = longitude[:]

        r1_nc[:] = r1
        s1_nc[:] = s1

        # Add global attributes
        ds.description = """Rain and snow data generated from NORA3 northern Norwegian subset precipitation and
                            temperature.
                            The hourly precipitation was classified as rain when the hourly temperature was > 273.15K
                            and as snow otherwise."""
        ds.history = 'Created ' + str(date.today())

        # Add local attributes to variables
        times.units = nc.variables["time"].units
        times.calendar = nc.variables["time"].calendar

        x_var.units = x.units
        y_var.units = y.units
        latitudes.units = latitude.units
        longitudes.units = longitude.units
        r1_nc.units = 'mm'
        s1_nc.units = 'mm'

        ds.close()

    # end for yr
# end def