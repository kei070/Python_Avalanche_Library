#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count the number of gridcells in a specific elevation band per warning region.
"""

#%% imports
import numpy as np
import pandas as pd

from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set paths
path = f"{path_par}/IMPETUS/NORA3/Cells_Between_Thres_Height/NorthernNorway_Subset/"


#%% set the elevation bands
low = np.arange(0, 901, 300)
high = np.arange(300, 1201, 300)


#%% loop over the regions and the elevation bands
for reg_code in regions.keys():
    for l, h in zip(low, high):

        # construct the file name
        fn = f"/NORA3_between{l}_and_{h}m_Coords_in_{regions[reg_code]}_Shape.csv"
        df = pd.read_csv(path + regions[reg_code] + fn)

        # count the number of cells
        n_cells = len(df)

        print(f"\n{regions[reg_code]} {l}-{h}m: {n_cells}")

    # end for l, h
# end for reg_code




