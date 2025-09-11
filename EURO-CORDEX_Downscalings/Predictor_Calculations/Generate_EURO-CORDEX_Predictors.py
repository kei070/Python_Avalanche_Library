#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the predictors based on the EURO-CORDEX data.
"""


#######################################################################################
##                                                                                   ##
##   IMPORTANT: Some of the 0-300m grid cells appear to be located in the ocean      ##
##              meaning that the were originally NaN in the EURO-CORDEX data and     ##
##              are now either 273.15 (in the case of temperature) or 0 otherwise.   ##
##              These values should not be used. WHAT IS A GOOD SOLUTION FOR THIS?   ##
##                                                                                   ##
#######################################################################################


#%% imports
import os
import sys
import numpy as np
import pylab as pl
import pandas as pd
import xarray as xr

import ava_functions.Calc_Preds as cprf
from ava_functions.Lists_and_Dictionaries.Paths import path_par3
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Avalanche_Climate import ava_clim_code


#%% set some parameters
reg_code = int(sys.argv[1])  # 3010
h_lo = int(sys.argv[2])  # 300
h_hi = int(sys.argv[3])  # 600
scen = sys.argv[4]  # "rcp85"
model = sys.argv[5]  # "MPI_RCA"


#%% set paths
eu_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Avalanche_Region_Data/{scen}/{model}/Between{h_lo}_and_{h_hi}m/"
fn = f"{model}_{scen}_{regions[reg_code]}_Between{h_lo}_and_{h_hi}.nc"

out_path = f"{path_par3}/IMPETUS/EURO-CORDEX/Avalanche_Region_Predictors/{scen}/{model}/Between{h_lo}_and_{h_hi}m/"
os.makedirs(out_path, exist_ok=True)


#%% load data
nc = xr.open_dataset(eu_path + fn)


#%% some plotting
pl.plot(nc.RR[0, :])


#%% temperature predictors
print("\nCalculating predictors...\n")
rr_pred = cprf.calc_rr_pred_eu(data=nc)
# swe_pred = cprf.calc_swe_pred_eu(data=nc)
temp_pred = cprf.calc_temp_pred_eu(data=nc)
print("\nPredictor calculations done.\n")


#%% concatenate the predictors
print("\nConcatenating and storing predictors...\n")
# df = pd.concat([temp_pred.drop("date", axis=1), rr_pred.drop("date", axis=1), swe_pred.drop("date", axis=1)], axis=1)
df = pd.concat([temp_pred.drop("date", axis=1), rr_pred.drop("date", axis=1)], axis=1)


#%% generate a column with the "avalanche climate index" taken from the Avalanche_Climate script
df["ava_clim"] = np.repeat(a=ava_clim_code[reg_code], repeats=len(df))


#%% store the data as csv
region = regions[reg_code]

df.to_csv(out_path + f"{region}_{model}_Predictors_MultiCellMean_Between{h_lo}_and_{h_hi}m.csv")


