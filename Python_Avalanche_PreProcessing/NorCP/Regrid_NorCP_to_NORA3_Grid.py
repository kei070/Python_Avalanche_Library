#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to regrid the NorCP data to the NORA3 grid.
Both are ona Lambert Conformal Conic projection but with different parameters.

USE CDO INSTEAD:

cdo remapbil,../NORA3_Grid_Info.txt tas_NEU-3_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_1hr_198501010000-198512312300.nc output_regridded_dataset.nc

--> call it with the subprocess module

To make sure the NORA3_Grid_Info.txt should (apparently) look like this:
gridtype          = projection
xsize             = 101
ysize             = 121
xfirst            = 1858360.9
xinc              = 3000
yfirst            = 1219523
yinc              = 3000
grid_mapping_name = lambert_conformal_conic
proj_params       = "+proj=lcc +lat_1=66.3 +lat_2=66.3 +lat_0=66.3 +lon_0=-42 +a=6371000. +b=6371000."

"""

#%% imports
import os
import sys
import glob
import subprocess
import time as time_module
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% wait with the execution
waiting_time = 60*60*0
print(f"\nWaiting {waiting_time} seconds...\n")
time_module.sleep(waiting_time)
t = time_module.localtime()
current_time = time_module.strftime("%H:%M:%S", t)
print("\nStarting execution at:")
print(current_time)
print("")


#%% set parameters
model = sys.argv[1]  # "GFDL-CM3"
scen = sys.argv[2]  # "rcp85"
period = sys.argv[3]  # "MC"  # either "", "MC", or "LC"
var = sys.argv[4]  # "uas"
regrid_meth = "remapbil"


#%% paths
ncp_path = f"{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}_{var}/{period}/"
out_path = f"{path_par}/IMPETUS/NorCP/{model.upper()}_{scen}_{var}/{period}/Regridded_to_NORA3/"
grid_fn = "NORA3_Grid_Info.txt"


#%% generate the output directory
os.makedirs(out_path, exist_ok=True)


#%% list files
fn_list = glob.glob(ncp_path + "*.nc")


#%% change working directory
os.chdir(f"{path_par}/IMPETUS/NorCP/")


#%% loop over the files to regrid them
operator_str = f"{regrid_meth},{grid_fn}"

count = 1
for fnf in fn_list:
    print(f"\n{count} of {len(fn_list)}...\n")

    fn = fnf.split("/")[-1]

    out_fn = fn.split("_")[:-1]
    out_fn.append("NORA3Grid")
    out_fn.append(fn.split("_")[-1])
    out_fn = "_".join(out_fn)

    out_fn = f"./{model.upper()}_{scen}_{var}/{period}/Regridded_to_NORA3/" + out_fn
    fn = f"./{model.upper()}_{scen}_{var}/{period}/" + fn

    cdo_command = ["cdo", operator_str, fn, out_fn]

    # print(cdo_command)

    subprocess.call(cdo_command)

    break

    count += 1
# end for fn

