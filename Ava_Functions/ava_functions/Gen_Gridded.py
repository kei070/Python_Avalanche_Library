#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All the gridded predictor functions.
"""


#%% imports
import os
import sys
import glob
import numpy as np
import xarray as xr
from datetime import date
from netCDF4 import Dataset
from dask.diagnostics import ProgressBar

from .Lists_and_Dictionaries.Paths import path_par
from .Lists_and_Dictionaries.Variable_Name_NORA3_NorCP import pred_names
from ava_functions.Progressbar import print_progress_bar
from ava_functions.Calc_Preds import wind_dir


#%% Calculate the gridded wind speed and direction from NorCP uas and vas data
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


#%% Calculate the gridded rain and snow parameters (from which r1, 3, 7 and s1, 3, 7 can be easily generated)
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


#%% Generate some change predictors (temperature range, wind-speed variation, ...)
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


#%% Generate the gridded precipitation predictors
#   --> This is based on the files generated with Calc_r1_s1_Gridded.py
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

# rolling-sum precipitation predictors
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


#%% Generate the gridded predictors
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


#%% Generate the gridded seNorge predictors
def gen_sn_preds(sn_path, out_path, fn, roll=[3, 7], syr_in=1970, eyr_in=2024, syr_out=2016, eyr_out=2024,
                 exist_ok=False):

    pred_exist = False
    if os.path.exists(out_path + f"seNorge_{fn}_Features.nc"):
        pred_exist = True
    # end if

    if pred_exist & (not exist_ok):
        print("\nseNorge predictors exist. Continuing with next...\n")
        return
    # end if

    #% load the necessary files
    sn_nc = xr.open_dataset(sn_path + f"seNorge_NorthNorway_{fn}_{syr_in}_{eyr_out}.nc")

    #% load the dataset
    swe = sn_nc["SWE"][sn_nc.time.dt.year >= syr_out]
    swe = swe.rename("SWE1")
    sdp = sn_nc["SnowDepth"][sn_nc.time.dt.year >= syr_out]
    sdp = sdp.rename("SDP1")
    sd = sn_nc["SnowDens"][sn_nc.time.dt.year >= syr_out]
    sd = sd.rename("SD1")
    mr = sn_nc["Melt_Refreeze"][sn_nc.time.dt.year >= syr_out]
    mr = mr.rename("MR1")
    print("Data loaded. Proceeding to calculations...\n")

    # % calculate the 3 and 7-day means
    ndays_l = roll

    swe_x = {}
    sdp_x = {}
    sd_x = {}
    mr_x = {}

    for ndays in ndays_l:
        swe_x[ndays] = swe.rolling(time=ndays, center=False).mean()
        swe_x[ndays] = swe_x[ndays].rename(f"SWE{ndays}")
        sdp_x[ndays] = sdp.rolling(time=ndays, center=False).mean()
        sdp_x[ndays] = sdp_x[ndays].rename(f"SDP{ndays}")
        sd_x[ndays] = sd.rolling(time=ndays, center=False).mean()
        sd_x[ndays] = sd_x[ndays].rename(f"SD{ndays}")
        mr_x[ndays] = mr.rolling(time=ndays, center=False).mean()
        mr_x[ndays] = mr_x[ndays].rename(f"MR{ndays}")
        print(f"\n{ndays}-days calculation started. Storing...\n")
    # end for ndays

    #% merge all the data arrays
    ds = xr.merge([swe, sdp, sd, mr, swe, swe_x[3], swe_x[7], sdp_x[3], sdp_x[7], sd_x[3], sd_x[7], mr_x[3], mr_x[7]])

    #% store the dataset as netcdf file
    ds.to_netcdf(out_path + f"seNorge_{fn}_Features.nc")

# end def


#%% Generate the gridded wind direction predictors. --> Still somewhat unsure about this, does the standard deviation
#   make sense, and even the rolling standard deviation?
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


#%% Generate the gridded wind-drift predictors
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