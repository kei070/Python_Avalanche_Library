#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt to regrid the EURO-CORDEX data to the NORA3 grid.
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
from ava_functions.Lists_and_Dictionaries.Paths import path_par3


#%% set parameters
model = sys.argv[1]
scen = sys.argv[2]
var = sys.argv[3]
regrid_meth = "remapbil"


#%% paths
eur_path = f"{path_par3}/IMPETUS/EURO-CORDEX/{scen}/{model}/"
out_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Regridded/{scen}/{model}/"

grid_fn = "NORA3_Grid_Info.txt"


#%% generate the output directory
os.makedirs(out_path, exist_ok=True)


#%% list files
fn_list = glob.glob(eur_path + f"*_{var}_*.nc")


#%% change working directory
os.chdir(f"{path_par3}/IMPETUS/EURO-CORDEX/")


#%% loop over the files to regrid them
operator_str = f"{regrid_meth},{grid_fn}"

count = 1
for fnf in fn_list:
    print(f"\n{count} of {len(fn_list)}...\n")

    fn = fnf.split("/")[-1]

    out_fn = fn.split("_")[:-2]
    out_fn.append("NORA3Grid")
    out_fn.append("_".join(fn.split("_")[-2:]))
    out_fn = "_".join(out_fn)

    # out_fn = f"./{scen}/{model}/" + out_fn
    out_fn = out_path + out_fn
    fn = f"./{scen}/{model}/" + fn

    cdo_command = ["cdo", operator_str, fn, out_fn]

    print(cdo_command)

    subprocess.call(cdo_command)

    break

    count += 1
# end for fn

