#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix the 1985 NSW data.
"""


#%% imports
import numpy as np
from netCDF4 import Dataset
from ava_functions.Lists_and_Dictionaries.Paths import nora3_sub_path, path_par


#%% set paths
path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/"  # "Annual_Files/"
fn = "NORA3_NorthNorway_Sub_198501.nc"


#%% load files
nc = Dataset(path + fn, mode="r+")


#%% load the NSW
nsw = nc["surface_net_shortwave_radiation"][:]

nsw[nsw.mask] = 0


#%% change the masked values to 0
nc["surface_net_shortwave_radiation"][:] = np.array(nsw)


#%% close the file
nc.close()