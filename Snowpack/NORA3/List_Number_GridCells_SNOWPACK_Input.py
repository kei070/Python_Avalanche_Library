#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List the number of grid cells per region per elevation band to have some more info about the data on which the
SNOWPACK runs are based.
"""

#%% imports
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from netCDF4 import Dataset
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% parameters
h_low = 0
h_hi = 300
reg_code = 3009


#%% set paths
data_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/"


#%% load file
print(" "*10 + "    0-300  |  300-600  |  600-900  |  900-120  |")
for reg_code in regions.keys():
    n_cells_str = ""
    for h_low, h_hi in zip([0, 300, 600, 900], [300, 600, 900, 1200]):
        try:
            nc = Dataset(data_path +
                         f"Between{h_low}_and_{h_hi}m/NORA3_{regions[reg_code]}_Between{h_low}_and_{h_hi}m.nc")
            n_cells = nc.dimensions["loc"].size
        except:
            n_cells = 0
        # end try except

        n_cells_str += f"    {n_cells:3}    |"

    # end for h_low, h_hi

    print(f"{regions[reg_code]:10}{n_cells_str}")
# end for reg_code