#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the NorCP gridcell extraction year for year but for all regions in one script.

THIS IS FOR THE FILE REGRIDDED TO THE NORA3 GRID!
"""


#%% imports
import os
import sys
import glob
import time
import argparse
import numpy as np
import xarray as xr
import warnings
from netCDF4 import Dataset
from datetime import datetime

from ava_functions.Progressbar import print_progress_bar
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% suppress UserWarnings to suppress the warning about the empty shp-files (e.g., Soer-Troms does not have data
#   between 900-1200 m)
warnings.filterwarnings('ignore', category=UserWarning)


#%% parser
parser = argparse.ArgumentParser(
                    description="""Extracts the grid cells from the regridded NorCP subset for each of the avalanche
                    regions for a given range of years""")
parser.add_argument('--model', default="EC-Earth", choices=["EC-Earth", "GFDL-CM3", "ERAINT"], type=str,
                    help='Model used in the downscaling.')
parser.add_argument('--scen', default="rcp85", choices=["historical", "evaluation", "rcp45", "rcp85"], type=str,
                    help='Scenario used in the downscaling.')
parser.add_argument('--period', default="", choices=["", "MC", "LC"], type=str,
                    help='Period used in the downscaling.')
parser.add_argument('--low', default=900, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=1200, type=int, help='The upper threshold for the altitude band to extract.')
parser.add_argument("--reg_codes", nargs="*", default=[3009, 3010, 3011, 3012, 3013],
                    help="""The numerical codes of the regions.""")

args = parser.parse_args()


#%% get the arguments from the parser
model = args.model  # "EC-Earth"
scen = args.scen  # "historical"
period = args.period  # ""  # either "", "MC", or "LC"
h_low = args.low  # 400  # args.low
h_hi = args.high  # 900  # args.high
reg_codes = args.reg_codes


#%% set start and end year depending on period
if ((scen == "historical") & (scen == "evaluation")):
    period = ""
# end if
period_str = ""
if period == "":
    if scen == "historical":
        sta_yr = 1985
        end_yr = 2005
    elif scen == "evaluation":
        sta_yr = 1998
        end_yr = 2018
    # end if elif
elif period == "MC":
    sta_yr = 2040
    end_yr = 2060
    period_str = "_MC"
elif period == "LC":
    sta_yr = 2080
    end_yr = 2100
    period_str = "_LC"
# end if elif


#%% extract NORA3 lat lon info
nora3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
nc_nora3 = Dataset(nora3_path + "NORA3_NorthNorway_Sub_2024.nc")
lon = nc_nora3.variables["longitude"][:]
lat = nc_nora3.variables["latitude"][:]


#%% loop over the region codes to extract the indices
indx_d = {}
indy_d = {}
lon_d = {}
lat_d = {}

not_avail = []  # set up a list for the regions that do not have grid cells in the given range
print("\nExtracting indices...\n")
l = len(reg_codes)
count = 0
print_progress_bar(0, l, prefix='Progress:', length=50)
for reg_code in reg_codes:

    # set region name according to region code
    if reg_code == 3009:
        region = "NordTroms"
    elif reg_code == 3010:
        region = "Lyngen"
    elif reg_code == 3011:
        region = "Tromsoe"
    elif reg_code == 3012:
        region = "SoerTroms"
    elif reg_code == 3013:
        region = "IndreTroms"
    # end if elif


    #% set data path
    shp_path = f"{path_par}/IMPETUS/NORA3/Cells_Between_Thres_Height/NorthernNorway_Subset/{region}/"


    # load the shp file to get the coordinates
    # print(shp_path + f"*between{h_low}_and_{h_hi}m*.csv")
    shp = np.loadtxt(glob.glob(shp_path + f"*between{h_low}_and_{h_hi}m*.csv")[0], delimiter=",", skiprows=1)

    if len(shp) == 0:
        suffix = f"No gridcells for {reg_code}. Continuing..."
        print_progress_bar(count, l, prefix='Progress:', suffix=suffix, length=50)
        not_avail.append(reg_code)
        count += 1
        continue
    # end if

    # extract NORA3 lat and lon above threshold
    indx_l = []
    indy_l = []
    lon_l = []
    lat_l = []
    for lo, la in zip(shp[:, 1], shp[:, 2]):
        ind = np.where((lo == lon) & (la == lat))
        indx_l.append(ind[0][0])
        indy_l.append(ind[1][0])
        lon_l.append(lo)
        lat_l.append(la)
    # end for lo, la

    indx_d[reg_code] = indx_l
    indy_d[reg_code] = indy_l
    lon_d[reg_code] = lon_l
    lat_d[reg_code] = lat_l

    count += 1
    print_progress_bar(count, l, prefix='Progress:', suffix="                                              ", length=50)

# end for reg_code


#%% load the nc file for the chosen year
data_path = f"{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}_"
re_path = "Regridded_to_NORA3"
yrs = np.arange(sta_yr, end_yr+1, 1)

print("\nExtracting grid cells...\n")
l = len(yrs) * np.sum(np.array([len(indx_d[k]) for k in indx_d.keys()]))
print_progress_bar(0, l, prefix='Progress:', suffix=f'{region}', length=50)
count = 0
region = ""
for yr in yrs:

    print_progress_bar(count, l, prefix='Progress:', suffix=f'| year {yr:3} | {region:20}', length=50)

    nc_vals = {}
    nc_vals3h = {}

    # open one dataset in xarray for the time array
    # print(data_path + f"tas/{period}/{re_path}/*{yr}*.nc")
    nc_xr = xr.open_dataset(sorted(glob.glob(data_path + f"tas/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_xr3h = xr.open_dataset(sorted(glob.glob(data_path + f"rlns/{period}/{re_path}/*_{yr}*.nc"))[0])

    # extract a get the time
    norcp_time = nc_xr.time
    norcp_time3h = nc_xr3h.time

    # get the time length
    t_len = len(norcp_time)
    t_len3h = len(norcp_time3h)

    # tas
    nc_tas = Dataset(sorted(glob.glob(data_path + f"tas/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals["tas"] = np.squeeze(nc_tas.variables["tas"][:])

    # Down SW
    nc_dsw = Dataset(sorted(glob.glob(data_path + f"rsds/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals["rsds"] = np.squeeze(nc_dsw.variables["rsds"][:])

    # Down LW
    nc_dsw = Dataset(sorted(glob.glob(data_path + f"rlds/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals["rlds"] = np.squeeze(nc_dsw.variables["rlds"][:])

    # RH
    nc_rh = Dataset(sorted(glob.glob(data_path + f"hurs/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals3h["hurs"] = np.squeeze(nc_rh.variables["hurs"][:]) / 100

    # NLW
    nc_nlw = Dataset(sorted(glob.glob(data_path + f"rlns/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals3h["rlns"] = np.squeeze(nc_nlw.variables["rlns"][:])

    # NSW
    nc_nsw = Dataset(sorted(glob.glob(data_path + f"rsns/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals3h["rsns"] = np.squeeze(nc_nsw.variables["rsns"][:])

    # pr
    nc_pr = Dataset(sorted(glob.glob(data_path + f"pr/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals["pr"] = np.squeeze(nc_pr.variables["pr"][:]) * 3600

    # rain and snow
    nc_s_r = Dataset(sorted(glob.glob(data_path + f"Rain_Snow/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals["rain"] = np.squeeze(nc_s_r.variables["rain"][:]) * 3600
    nc_vals["snow"] = np.squeeze(nc_s_r.variables["snow"][:]) * 3600

    # wind speed and direction
    nc_wind = Dataset(sorted(glob.glob(data_path + f"Wind_Speed_Dir/{period}/{re_path}/*_{yr}*.nc"))[0])
    nc_vals["wspeed"] = np.squeeze(nc_wind.variables["wspeed"][:])
    nc_vals["wdir"] = np.squeeze(nc_wind.variables["wdir"][:])


    # perform a test to check if the variables have the same shapes
    if not (np.shape(nc_vals3h["rlns"]) == np.shape(nc_vals3h["rsns"]) == np.shape(nc_vals3h["hurs"])):
        print("\n\nERROR: rlns, rsns, hurs do not have the same shape. Aborting.\n\n")
        sys.exit()
    # end if

    #% get the variable names from the nc_vals dictionary
    var = list(nc_vals.keys())
    var3h = list(nc_vals3h.keys())

    #% extract time units and calendar
    # ti_units = nc_tas.variables["time"].units  # nc_xr['time'].attrs['units']
    # ti_calendar = nc_xr['time'].attrs.get('calendar', 'standard')


    #% set up a dicitionary to store all data
    # print("\nStoring data...\n")
    # for reg_code in reg_codes:
    for reg_code in indx_d.keys():

        if reg_code in not_avail:
            continue
        # end if

        # set region name according to region code
        if reg_code == 3009:
            region = "NordTroms"
        elif reg_code == 3010:
            region = "Lyngen"
        elif reg_code == 3011:
            region = "Tromsoe"
        elif reg_code == 3012:
            region = "SoerTroms"
        elif reg_code == 3013:
            region = "IndreTroms"
        # end if elif

        print_progress_bar(count, l, prefix='Progress:', suffix=f'| year {yr:3} | {region:20}', length=50)

        out_path = (f"{path_par}/IMPETUS/NorCP/Avalanche_Region_Data/" +
                    f"/Between{h_low}_and_{h_hi}m/{model.upper()}_{scen}{period_str}/Annual_Files/{region}/")
        os.makedirs(out_path, exist_ok=True)

        # count = 0
        # l = len(indx_d[reg_code])
        # print(f"\n{l} locations.\n")
        # print_progress_bar(0, l, prefix='Progress:', suffix=f'Complete of {region}', length=50)

        temp_dic = {varn:[] for varn in var[:]}
        temp_dic["date"] = norcp_time
        temp_dic["x_loc"] = []
        temp_dic["y_loc"] = []

        temp3h_dic = {varn:[] for varn in var3h[:]}
        temp3h_dic["date"] = norcp_time3h
        temp3h_dic["x_loc"] = []
        temp3h_dic["y_loc"] = []
        for loc1, loc2 in zip(indx_d[reg_code], indy_d[reg_code]):

            # print([loc1, loc2])

            # add the locations to the dictionary
            temp_dic["x_loc"].append(loc1)
            temp_dic["y_loc"].append(loc2)
            temp3h_dic["x_loc"].append(loc1)
            temp3h_dic["y_loc"].append(loc2)

            # start a timer
            start_time = time.time()

            for varn in var[:]:
                # print(varn)
                temp_dic[varn].append(nc_vals[varn][:, loc1, loc2])
            # end for varn

            for varn in var3h[:]:
                # print(varn)
                temp3h_dic[varn].append(nc_vals3h[varn][:, loc1, loc2])
            # end for varn

            # count += 1
            # print_progress_bar(count, l, prefix='Progress:', suffix=f'Complete of {region}', length=50)

            count += 1
            print_progress_bar(count, l, prefix='Progress:', suffix=f'| year {yr:3} | {region:20}', length=50)

        # end for loc1, loc2

        # convert lists to arrays
        for varn in var:
            temp_dic[varn] = np.stack(temp_dic[varn])
        # end for varn

        for varn in var3h:
            temp3h_dic[varn] = np.stack(temp3h_dic[varn])
        # end for varn

        # generate a dictionary for the variable data and coords dictionary
        var_dic = {varn:(("loc", "time"), temp_dic[varn]) for varn in var}
        var3h_dic = {varn:(("loc", "time3h"), temp3h_dic[varn]) for varn in var3h}
        var_dic = var_dic | var3h_dic  # merge the dictionaries

        # var_dic = {"t2m":(["loc", "time"], temp_dic["tas"]),
        #            "rlns":(["loc", "time3h"], temp3h_dic["rlns"])}

        var_dic["loc_x"] = (("loc"), temp_dic["x_loc"])
        var_dic["loc_y"] = (("loc"), temp_dic["y_loc"])
        var_dic["lon"] = (("loc"), lon_d[reg_code])
        var_dic["lat"] = (("loc"), lat_d[reg_code])

        coords = {"loc":np.arange(len(temp_dic["x_loc"])),
                  "time":np.array(temp_dic["date"]),
                  "time3h":np.array(temp3h_dic["date"])}


        ds = xr.Dataset(var_dic, coords=coords,)
        ds.to_netcdf(out_path + f"NorCP_{region}_Between{h_low}_and_{h_hi}_{yr}.nc")

        # generate a nc dataset from the dictionary
        """
        print(f"\nGenerating {region} nc-file.\n")
        ncfile = Dataset(out_path + f"NorCP_{region}_{yr}.nc", 'w', format='NETCDF4')

        # Define dimensions
        ncfile.createDimension('time', t_len)  # unlimited
        ncfile.createDimension('loc', l)

        # Create variables
        nc_times = ncfile.createVariable('time', np.float64, ('time'))
        nc_locx = ncfile.createVariable("loc_x", np.float64, ("loc"))
        nc_locy = ncfile.createVariable("loc_y", np.float64, ("loc"))
        nc_vars = [ncfile.createVariable(varn, np.float64, ('loc', 'time')) for varn in var]

        # Assign data to variables
        for varn, nc_var in zip(var, nc_vars):
            nc_var[:] = temp_dic[varn]
        # end for varn, nc_var

        nc_times[:] = norcp_time
        nc_locx[:] = temp_dic["x_loc"]
        nc_locy[:] = temp_dic["y_loc"]

        # set the origin of the time coordinate
        nc_times.units = ti_units
        nc_times.calendar = ti_calendar

        # Add global attributes
        ncfile.description = f'{region} NorCP {model} {scen} data for gridcells between {h_low} and {h_hi} m.'
        ncfile.history = 'Created ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Close the file
        ncfile.close()
        """
    # end for reg_code
    # count += 1
    # print_progress_bar(count, l, prefix='Progress:', suffix=f'{region:20}', length=50)
# end for yr


#%%
# test = xr.open_dataset(out_path + f"NorCP_{region}_Between{h_low}_and_{h_hi}_{yr}.nc")

""" generate an xarray dataset with two different time dimensions
import pandas as pd
time1 = pd.date_range('2023-01-01', periods=5, freq='D')
time2 = pd.date_range('2023-01-01', periods=5, freq='2D')

data1 = np.random.rand(5, 3, 3)  # Random data for the first time coordinate
data2 = np.random.rand(5, 3, 3)  #

te_var_dic = {
              "var1": (["time1", "lat", "lon"], data1),
              "var2": (["time2", "lat", "lon"], data2)
             }
te_coords = {
             "time1": time1,
             "time2": time2,
             "lat": [10, 20, 30],
             "lon": [40, 50, 60]
             }
ds = xr.Dataset(te_var_dic, coords=te_coords)

"""