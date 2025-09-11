#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the grid cells from the EURO-CORDEX data corresponding to the NORA3 grid cells.
"""


#%% imports
import os
import glob
import time
import argparse
import warnings
import numpy as np
import pylab as pl
import xarray as xr
from netCDF4 import Dataset
from ava_functions.Lists_and_Dictionaries.Paths import path_par, path_par3
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Progressbar import print_progress_bar


#%% suppress UserWarnings to suppress the warning about the empty shp-files (e.g., Soer-Troms does not have data
#   between 900-1200 m)
warnings.filterwarnings('ignore', category=UserWarning)


#%% parser
parser = argparse.ArgumentParser(
                    description="""Extracts the grid cells from the regridded NorCP subset for each of the avalanche
                    regions for a given range of years""")
parser.add_argument('--model', default="CNRM_RCA", choices=["MPI_RCA", "IPSL_RCA", "CNRM_RCA"], type=str,
                    help='Model used in the downscaling.')
parser.add_argument('--scen', default="rcp85", choices=["hist", "rcp45", "rcp85"], type=str,
                    help='Scenario used in the downscaling.')
parser.add_argument('--low', default=900, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=1200, type=int, help='The upper threshold for the altitude band to extract.')
parser.add_argument("--reg_codes", nargs="*", default=[3009, 3010, 3011, 3012, 3013],
                    help="""The numerical codes of the regions.""")

args = parser.parse_args()


#%% get the arguments from the parser
model = args.model  # "MPI_RCA"
scen = args.scen  # "hist"
h_low = args.low  # 400  # args.low
h_hi = args.high  # 900  # args.high
reg_codes = args.reg_codes


#%% set paths
eu_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Regridded/{scen}/{model}/"
nora3_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
out_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Avalanche_Region_Data/{scen}/{model}/Between{h_low}_and_{h_hi}m/"
os.makedirs(out_path, exist_ok=True)



#%% load NORA3 file and grid
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
    region = regions[reg_code]


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


#%% load the EURO-CORDEX file names
fn_list = glob.glob(f"{eu_path}{model}_*_daily_NORA3Grid_*_v4.nc")


#%% loop over the files and load the data

# set the variable list
eu_var_dic = {"RR":f"precipitation__map_{scen}_daily",
              # "SWE":f"snow_water_equivalent__map_{scen}_daily",
              "TM":f"air_temperature__map_{scen}_daily",
              "TN":f"min_air_temperature__map_{scen}_daily",
              "TX":f"max_air_temperature__map_{scen}_daily"}

nc_vals = {}
for fn in fn_list:

    print(fn)

    # get the components of the file name
    comps = fn.split("/")[-1].split("_")

    # find the variable name as the intersect between the variable list and the components of the file name
    var_n = list(set(eu_var_dic.keys()) & set(comps))[0]

    # print_progress_bar(count, l, prefix='Progress:', suffix=f'| year {yr:3} | {region:20}', length=50)

    # open one dataset in xarray for the time array
    # print(data_path + f"tas/{period}/{re_path}/*{yr}*.nc")
    nc_xr = xr.open_dataset(fn)

    # extract a get the time
    eu_time = nc_xr.time

    # get the time length
    t_len = len(eu_time)

    # load the nc file and the values
    nc = Dataset(fn)
    nc_vals[var_n] = np.squeeze(nc.variables[eu_var_dic[var_n]][:])

# end for fn


#%% perform the extraction
print("\n\nExtracting grid cells...\n")
l = len(regions)
print_progress_bar(0, l, prefix='Progress:', suffix="", length=50)
count = 0
for reg_code in regions.keys():

    if reg_code in not_avail:
        continue
    # end if

    # set region name according to region code
    region = regions[reg_code]

    temp_dic = {varn:[] for varn in eu_var_dic.keys()}
    temp_dic["date"] = eu_time
    temp_dic["x_loc"] = []
    temp_dic["y_loc"] = []
    temp_dic["lon"] = []
    temp_dic["lat"] = []

    for loc1, loc2, lon_i, lat_i in zip(indx_d[reg_code], indy_d[reg_code], lon_d[reg_code], lat_d[reg_code]):

        # print([loc1, loc2])

        # start a timer
        start_time = time.time()

        # jump those grid cells where the TM always is 273.15 --> these are the cells either over the ocean or outside
        # Norway
        if np.sum(nc_vals["TM"][:, loc1, loc2] - 273.15) == 0:
            continue
        # end if

        # add the locations to the dictionary
        temp_dic["x_loc"].append(loc1)
        temp_dic["y_loc"].append(loc2)
        temp_dic["lon"].append(lon_i)
        temp_dic["lat"].append(lat_i)

        for varn in eu_var_dic.keys():
            # print(varn)
            temp_dic[varn].append(nc_vals[varn][:, loc1, loc2])
        # end for varn

    # end for loc1, loc2

    # convert lists to arrays
    for varn in eu_var_dic.keys():
        temp_dic[varn] = np.stack(temp_dic[varn])
    # end for varn

    # generate a dictionary for the variable data and coords dictionary
    var_dic = {varn:(("loc", "time"), temp_dic[varn]) for varn in eu_var_dic.keys()}

    var_dic["loc_x"] = (("loc"), temp_dic["x_loc"])
    var_dic["loc_y"] = (("loc"), temp_dic["y_loc"])
    var_dic["lon"] = (("loc"), temp_dic["lon"])
    var_dic["lat"] = (("loc"), temp_dic["lat"])

    coords = {"loc":np.arange(len(temp_dic["x_loc"])),
              "time":np.array(temp_dic["date"])}


    ds = xr.Dataset(var_dic, coords=coords,)
    ds.to_netcdf(out_path + f"{model}_{scen}_{region}_Between{h_low}_and_{h_hi}.nc")

    count += 1
    print_progress_bar(count, l, prefix='Progress:', suffix=f"{reg_code} done.", length=50)
# end for reg_code


#%% brief final check to make sure that the cells are correct: are the x and y values (the georeferenced ones) the same?
x_n3 = nc_nora3.variables["x"][:]
x_eu = nc.variables["x"][:]

pl.plot(x_n3, linewidth=3, c="black", label="NORA3")
pl.plot(x_eu, linewidth=1.5, c="orange", label="EU")
pl.legend()
pl.show()
pl.close()



