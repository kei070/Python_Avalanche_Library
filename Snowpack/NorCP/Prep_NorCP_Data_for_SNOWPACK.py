#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the NorCP data as extracted using Extract_NorCP_GridCells_Between_NC.py

This produces the .smet and .sno files needed as input for SNOWPACK. Note that you also need a .ini file!
"""


#%% imports
import os
import sys
import argparse
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


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Prep SNOWPACK Input",
                    description="""Prepares the NorCP data for SNOWPACK.""")

# ...and add the arguments
parser.add_argument("--reg_code", default=3011, type=int, help="""The region code.""")
parser.add_argument("--low", default=0, type=int, help="""The lower elevation band threshold.""")
parser.add_argument("--high", default=300, type=int, help="""The upper elevation band threshold.""")
parser.add_argument("--model", default="EC-Earth", type=str, help="The NorCP model.")
parser.add_argument("--scen", default="rcp45", type=str, help="The NorCP scenario.")
parser.add_argument("--period", default="LC", type=str, help="The NorCP period.")

slope_angle = 0
slope_azi = 0

update_smet = False

args = parser.parse_args()


#%% get the values from the parser
reg_code = args.reg_code
h_low = args.low
h_hi = args.high

model = args.model
scen = args.scen
period = args.period

# set up the period string
per_str = f"_{period}" if len(period) > 0 else ""


#%% set up the years string
if scen == "historical":
    yrs_str = "1985-2005"
if scen == "evaluation":
    yrs_str = "1998-2018"
if period == "MC":
    yrs_str = "2040-2060"
if period == "LC":
    yrs_str = "2080-2100"
# end if

prof_date = f"{yrs_str[:4]}-10-01T00:00:00"


#%% generate the strings based on slope and aspect
slope_str = "Flat"
if slope_angle > 0:
    slope_str = "Slope"
# end if
aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

if slope_angle == 0:
    aspect = ""
# end if


#%% set paths
data_path = f"{path_par}/IMPETUS/NorCP/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m/" + \
                                                                                     f"{model.upper()}_{scen}{per_str}/"
out_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Input/Elev_Mean_SMET/Between{h_low}_{h_hi}m/" + \
                                                                                     f"{model.upper()}_{scen}{per_str}/"
warn_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% generate the output directory
os.makedirs(out_path, exist_ok=True)


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
vstation['slope_angle'] = 0
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
gen_sno(params=sno_params, prof_date=prof_date, sno_dir=out_path, source="NorCP")


#%% check if the smet file already exists and if yes do not load the data
fn = regions[reg_code] + f"_{h_low}_{h_hi}m.smet"

f_exists = os.path.isfile(out_path + fn)

if ((not f_exists) | update_smet):
    #% load data
    # nc = Dataset(data_path + f"NORA3_{regions[reg_code]}_Between{h_low}_and_{h_hi}m.nc")
    ds = xr.open_dataset(data_path + f"NorCP_{regions[reg_code]}_Between{h_low}_and_{h_hi}_{yrs_str}.nc")

    ds = ds.drop_vars(["rlns", "rsns", "snow", "rain"])

    #% loop over the data variables and drop the loc_x and loc_y variables
    # ds_means = ds.mean(dim="loc")
    ds_means = ds.median(dim="loc")  # .drop(["loc_x", "loc_y", "surface_net_longwave_radiation"])


    # extract the hurs which is 3h data instead of 1h
    df_rh = ds_means["hurs"].to_dataframe().resample('30T').interpolate(limit_direction="both", method='linear')
    df_rh = df_rh.resample("1h").interpolate(limit_direction="both", method='linear')

    ds_means = ds_means[["tas", "pr", "wspeed", "rlds", "rsds"]]

    #% convert to dataframe
    df = ds_means.to_dataframe()

    # add hurs to the df
    df = df.join(df_rh["hurs"], how="left")

    #% rename the columns to accomodate SNOWPACK requirements
    varn = {"tas":'TA', "wspeed":'VW', "wdir":'DW', "hurs":"RH",
            "rlds":'ILWR', "rsds":'ISWR', "pr":'PSUM'}
    df = df.rename(columns={o_n:n_n for o_n, n_n in zip(varn.keys(), varn.values())})


    #% add the ground temperature
    df["TSG"] = np.repeat(df["TA"].mean(), len(df["TA"]))


    #% use the AWESOME functions to generate the .smet files
    sg.df2smet(df=df, vstation=vstation, smet_dir=out_path)
else:

    print("\n.smet file exists and update not requested, only generating .sno file.\n")

# end else if
