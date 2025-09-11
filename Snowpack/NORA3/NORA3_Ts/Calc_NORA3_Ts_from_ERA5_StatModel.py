#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive the NORA3 Ts from the ERA5-based statistical model.
"""

#%% imports
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from joblib import load
from datetime import date

from ava_functions.Progressbar import print_progress_bar
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths & names
n3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
out_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Ts_Files/"
model_path = f"{path_par}/IMPETUS/ERA5/Stat_Models/"
model_fn = "Linear_Ts_Model_ERA5.joblib"


#%% create the output directory if necessary
os.makedirs(out_path, exist_ok=True)


#%% load the model to derive the Ts from the other quantities
ts_model = load(model_path + model_fn)


#%% load a first file for the grid
nc_ini = Dataset(n3_path + "NORA3_NorthNorway_Sub_1985.nc")

# load the grid info
x = np.array(nc_ini["x"])
y = np.array(nc_ini["y"])


#%% set the year list
yrs = np.arange(1985, 1985+1, 1)


#%% execute
l = nc_ini.dimensions["x"].size * nc_ini.dimensions["y"].size * len(yrs)

count = 0
print(f"\nDeriving NORA3 surface temperature for {yrs[0]} to {yrs[-1]}\n")
print_progress_bar(count, l)
for yr in yrs:

    # filenames dependent on year
    n3_fn = f"NORA3_NorthNorway_Sub_{yr}.nc"
    out_fn = f"NORA3_modelled_Ts_{yr}.nc"

    nc = Dataset(n3_path + n3_fn)

    #% load the data
    tas = np.array(np.squeeze(nc["air_temperature_2m"]))
    # print(f"tas {yr} loaded...")
    print_progress_bar(count, l, suffix=f"  | {yr} tas loaded  |               ")
    ws = np.array(np.squeeze(nc["wind_speed"]))
    # print(f"ws {yr} loaded...")
    print_progress_bar(count, l, suffix=f"  | {yr} ws loaded   |               ")
    snlw = np.array(np.squeeze(nc["surface_net_longwave_radiation"]))
    # print(f"snlw {yr} loaded...")
    print_progress_bar(count, l, suffix=f"  | {yr} snlw loaded |               ")
    snsw = np.array(np.squeeze(nc["surface_net_shortwave_radiation"]))
    # print(f"snsw {yr} loaded. Predicting...\n")
    print_progress_bar(count, l, suffix=f"  | {yr} snsw loaded | predicting... ")

    #% loop over all cells to apply the model
    ts = np.zeros((nc.dimensions["time"].size, nc.dimensions["y"].size, nc.dimensions["x"].size))
    for i in np.arange(nc.dimensions["x"].size):
        for j in np.arange(nc.dimensions["y"].size):
            # tas = np.array(np.squeeze(nc["air_temperature_2m"])[:, j, i])
            # print_progress_bar(count, l, suffix="  |  tas loaded  |              ")
            # ws = np.array(np.squeeze(nc["wind_speed"])[:, j, i])
            # print_progress_bar(count, l, suffix="  |  ws loaded   |              ")
            # snlw = np.array(np.squeeze(nc["surface_net_longwave_radiation"])[:, j, i])
            # print_progress_bar(count, l, suffix="  |  snlw loaded |              ")
            # snsw = np.array(np.squeeze(nc["surface_net_shortwave_radiation"])[:, j, i])
            # print_progress_bar(count, l, suffix="  |  snsw loaded | predicting...")

            df_x = pd.DataFrame({"tas":tas[:, j, i], "ws":ws[:, j, i], "snlw":snlw[:, j, i], "snsw":snsw[:, j, i]})

            ts[:, j, i] = ts_model.predict(df_x)

            count += 1
            # print_progress_bar(count, l, suffix="  |  snsw loaded | done...      ")
            print_progress_bar(count, l, suffix=f"  | {yr} snsw loaded | predicting... ")
        # end for j
    # end for i

    print_progress_bar(count, l, suffix=f"  | {yr} predicting done | storing...")

    #% store the ts as nc-file
    ds = Dataset(out_path + out_fn, 'w', format='NETCDF4')

    # define the dimensions of the data
    time = ds.createDimension('time', None)
    x_dim = ds.createDimension('x', nc.dimensions["x"].size)
    y_dim = ds.createDimension('y', nc.dimensions["y"].size)

    # create coordinate variables for 3 dimensions
    times = ds.createVariable('time', 'f4', ('time',))
    x_var = ds.createVariable('x', 'f4', ('x',))
    y_var = ds.createVariable('y', 'f4', ('y',))

    # latitudes = ds.createVariable('latitude', 'f4', ('y', 'x'))
    # longitudes = ds.createVariable('longitude', 'f4', ('y', 'x'))

    # create the actual variables
    ts_nc = ds.createVariable('ts', 'f4', ('time', 'y', 'x'))

    # fill the variables
    times[:] = nc.variables["time"][:]
    x_var[:] = nc["x"][:]
    y_var[:] = nc["y"][:]
    # latitudes[:] = latitude[:]
    # longitudes[:] = longitude[:]

    ts_nc[:] = ts

    # Add global attributes
    ds.description = f"""Surface temperature (ts) derived from an ERA5-based linear model generated from tas, net lw and
    sw radiation, as well as wind speed. Data is hourly for the year {yr}."""
    ds.history = 'Created ' + str(date.today())

    # Add local attributes to variables
    times.units = nc.variables["time"].units
    times.calendar = nc.variables["time"].calendar

    x_var.units = nc["x"].units
    y_var.units = nc["y"].units
    # latitudes.units = latitude.units
    # longitudes.units = longitude.units
    ts_nc.units = 'K'

    ds.close()

# end for yr


