#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the NORA3 data as extracted using Extract_NORA3_GridCells_Between_NC.py

This produces the .smet and .sno files needed as input for SNOWPACK. Note that you also need a .ini file!
"""

#%% imports
import os
import sys
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from ava_functions.Snowpack_Helpers import gen_sno
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from nwp_provider import smet_generator as sg
import cartopy.crs as ccrs
from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path
from dask.diagnostics import ProgressBar


#%% set hights
h_low = 300
h_hi = 600
slope_angle = 0
slope_azi = 0
reg_code = 3012
update_smet = False


#%% generate the strings based on slope and aspect
slope_str = "Flat"
if slope_angle > 0:
    slope_str = f"{slope_angle}"
# end if
aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

if slope_angle == 0:
    aspect = ""
# end if


#%% set paths
data_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m/"
out_path = f"{path_par}/IMPETUS/NORA3/Snowpack/Input/Elev_Mean_SMET/Between{h_low}_{h_hi}m/"
warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% set the crs
# crs = ccrs.Orthographic(central_latitude=69, central_longitude=20)
crs = ccrs.PlateCarree()


#%% load the avalanche warning region info and calculate the centroid lat and lon
warn_info = gpd.read_file(warn_path + "Warning_Region_Info.gpkg")
warn_info = warn_info.to_crs(crs)
warn_info = warn_info[warn_info["reg_code"] == reg_code]
warn_info["centroid"] = warn_info.geometry.centroid


#%% set up the vstation file
vstation = {}
vstation["vstation"] = regions[reg_code] + f"_{h_low}_{h_hi}m"
vstation["filename"]    = vstation["vstation"]
vstation["name"]        = vstation["vstation"]
vstation['slope_angle'] = slope_angle
vstation['easting']  = str(-999)
vstation['northing'] = str(-999)
vstation['lat']  = float(warn_info["centroid"].y)
vstation['lon'] = float(warn_info["centroid"].x)
vstation["elev"] = str((h_low+h_hi)/2)


#%% use the helper function to generate the .sno file
sno_params = {}
sno_params["vstation"]    = vstation["vstation"] + f"_{slope_str}{aspect}"
sno_params["filename"]    = vstation["filename"] + f"_{slope_str}{aspect}"
sno_params["name"]        = vstation["name"] + f"_{slope_str}{aspect}"
sno_params['lat']         = float(warn_info["centroid"].y)
sno_params['lon']         = float(warn_info["centroid"].x)
sno_params["elev"]        = str((h_low+h_hi)/2)
sno_params["SlopeAngle"]  = str(slope_angle)
sno_params["SlopeAzi"]    = str(slope_azi)
prof_date = "1970-10-01T00:00:00"
gen_sno(params=sno_params, prof_date=prof_date, sno_dir=out_path, source="NORA3")


#%% check if the smet file already exists and if yes do not load the data
fn = regions[reg_code] + f"_{h_low}_{h_hi}m.smet"

f_exists = os.path.isfile(out_path + fn)

# set the variables and the names
varn = {"air_temperature_2m":'TA', "relative_humidity_2m":'RH', "wind_speed":'VW', "wind_direction":'DW',
        "surface_net_shortwave_radiation":'NET_SW', "ts":'TSS', "precipitation_amount_hourly":'PSUM'}

if ((not f_exists) | update_smet):

    with ProgressBar():
        #% load data
        # nc = Dataset(data_path + f"NORA3_{regions[reg_code]}_Between{h_low}_and_{h_hi}m.nc")
        ds = xr.open_dataset(data_path + f"NORA3_{regions[reg_code]}_Between{h_low}_and_{h_hi}m.nc",
                             chunks={'time':100000})
        ds = ds.drop(["loc_x", "loc_y", "surface_net_longwave_radiation"])

        # average
        # ds_means = ds.mean(dim="loc")
        ds_means = ds.median(dim="loc")

        #% convert to dataframe
        df = ds_means.to_dataframe()
    # end with

    #% rename the columns to accomodate SNOWPACK requirements
    df = df.rename(columns={o_n:n_n for o_n, n_n in zip(varn.keys(), varn.values())})


    #% add the ground temperature
    df["TSG"] = np.repeat(df["TA"].mean(), len(df["TA"]))


    #% use the AWESOME functions to generate the .smet files
    sg.df2smet(df=df, vstation=vstation, smet_dir=out_path)
else:

    print("\n.smet file exists and update not requested, only generating .sno file.\n")

# end else if


