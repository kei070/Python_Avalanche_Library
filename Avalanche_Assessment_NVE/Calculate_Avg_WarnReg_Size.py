#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate average avalanche warning-region size in Norway.
"""


#%% imports
import sys
import numpy as np
import pandas as pd
import pylab as pl
from netCDF4 import Dataset
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% set some parameters
# reg_code = 3010
h_low = 15
h_hi = 2500
ndlev = 4
# n_points = 10  # number of grid points from which to take predictive feature data --> NEW: using all points!
split = [2021, 2023]


#%% set up the region codes
reg_codes = [3009, 3010, 3011, 3012, 3013]


#%% set paths
nora3_path = f"{path_par}/IMPETUS/NORA3/"


#%% load the topo and region data
topo_nc = Dataset(nora3_path + "nora3_topo.nc")
warn_nc = Dataset(nora3_path + "Warn_Region_Raster.nc")


#%% extract the projection
crs = topo_nc.variables["projection_lambert"].proj4


#%% load the values
topo = topo_nc.variables["ZS"][:].squeeze()
warn = warn_nc.variables["warn_regions"][:].astype(float)


#%% replace the zeros in warn with NaN
warn[warn == 0] = np.nan


#%% plot
fig = pl.figure()
ax00 = fig.add_subplot(111)

ax00.pcolormesh(warn)

pl.show()
pl.close()


#%% get the number of individual regions
regs_uniq = np.unique(warn)[:-1]  # [:-1] for dropping NaN
regs_uniq = [3009, 3010, 3011, 3012, 3013]


#%% loop over the regions, sum up the number of grid cells per region and multiply it by 3x3km^2
reg_size = {}

for reg_code in regs_uniq:
    reg_size[reg_code]  = np.sum(warn == reg_code) * 3 * 3
# end for reg_code

print(f"Mean size: {np.mean(np.array(list(reg_size.values())))} km^2")