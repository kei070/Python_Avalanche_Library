#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the info from the NORA3 DEM to add the grid cell elevation threshold to the avalanche region data.
"""


#%% imports
import os
import numpy as np
from netCDF4 import Dataset

from ava_functions.Extract_Subset import extract_subset
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Progressbar import print_progress_bar


#%% set parameters
h_low = 600
h_hi = 900
reg_code = 3011


#%% set paths
n3_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/"
sub_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"
dem_path = f"{path_par}/IMPETUS/NORA3/"


#%% load the
dem_nc = Dataset(dem_path + "nora3_topo.nc")
sub_nc = Dataset(sub_path + "NORA3_NorthNorway_Sub_2024.nc")


#%% get the DEM subset
dem_sub, y_inds, x_inds = extract_subset(arr_large=dem_nc, arr_small=sub_nc)


#%% loop over the files and update them
hs_low = [0, 300, 600, 900]
hs_hi = [300, 600, 900, 1200]

l = int(len(regions) * len(hs_hi))
count = 0
print_progress_bar(count, l)
for reg_code in regions.keys():
    for h_low, h_hi in zip(hs_low, hs_hi):

        path = n3_path + f"Between{h_low}_and_{h_hi}m/NORA3_{regions[reg_code]}_Between{h_low}_and_{h_hi}m.nc"

        if os.path.isfile(path):
            #% load the avalanche region data
            ava_nc = Dataset(path, mode="r+")

            #% load the locations of the avalanche data
            loc_x = ava_nc["loc_x"][:].squeeze()
            loc_y = ava_nc["loc_y"][:].squeeze()

            #% loop over the locations and extract the elevation information
            elevs = np.array([dem_sub[int(yi), int(xi)] for yi, xi in zip(loc_x, loc_y)])

            #% update the nc file
            try:
                nc_elev = ava_nc.createVariable("elev", np.float64, ('loc'))
                nc_elev[:] = elevs
                nc_elev.units = "m"
                nc_elev.description = "Elevation information taken from NORA3."
            except:
                # print("\nVariable elev already exists. Refilling it...")
                print_progress_bar(count, l, suffix="elev already exists")
                nc_elev = ava_nc["elev"]
                nc_elev[:] = elevs
            # end try except

            #% close the file
            ava_nc.close()
            count += 1
        else:
            print_progress_bar(count, l, suffix="file does not exist")
            count += 1
            continue
        # end if else

        print_progress_bar(count, l, suffix="")
    # end for h_low, h_hi
# end for reg_code



