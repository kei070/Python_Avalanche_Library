#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the EURO-CORDEX data from the thredds server as provided by Oskar Landgren.
"""

#%% imports
import os
import glob
import subprocess
import numpy as np

from ava_functions.Lists_and_Dictionaries.Paths import path_par, path_par2, path_par3


#%% set parameters for the download
vars_l = ["TX", "TN", "TM", "SWE", "RR"]
mods_l = ["CNRM_RCA"]  # ["IPSL_RCA", "MPI_RCA"]
scens_l = ["hist", "rcp45", "rcp85"]


#%% set directory
down_dir = f"/{path_par3}/IMPETUS/EURO-CORDEX/"


#%% thredds directory
thredds_dir = "https://thredds.met.no/thredds/fileServer/KSS/Klima_i_Norge_2100/utgave2015/"


#%% if the directory does not exist, create it
os.makedirs(down_dir, exist_ok=True)


#%% load the existing files
f_list = sorted(glob.glob(down_dir + "*.nc"))


#%% set up the file name based on the above parameters
for var in vars_l:
    for scen in scens_l:
        sta_yr = 1971 if scen == "hist" else 2006
        end_yr = 2000 if scen == "hist" else 2100

        for mod in mods_l:

            down_dir_final = f"{down_dir}/{scen}/{mod}/{var}/"
            os.makedirs(down_dir_final, exist_ok=True)

            for yr in np.arange(sta_yr, end_yr+1):
                f_name = f"{var}/{mod}/{scen}/{scen}_{mod}_{var}_daily_{yr}_v4.nc"
                # print(f_name)

                # if the file exists, proceed to next file
                if down_dir + f_name in f_list:
                    print(f"File for year {yr} exists, continuing with next file...")
                    continue
                # end if

                # if the file does not exist, download it
                subprocess.call(["wget", "-P", down_dir_final, thredds_dir + f_name])

            # end for yr
        # end for mod
    # end for scen
# end for var