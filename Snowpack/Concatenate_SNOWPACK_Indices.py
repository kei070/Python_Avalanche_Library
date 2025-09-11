#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate the SNOWPACK indices: For the Lyngen region the NorCP-SNOWPACK simulation output is too large to generate
the stability indices for all timesteps at once.
"""

#%% imports
import sys
import pandas as pd
from ava_functions.Lists_and_Dictionaries.Paths import path_par
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% get the values from the parser
"""
reg_code = int(sys.argv[1])
h_low = int(sys.argv[2])
h_hi = int(sys.argv[3])
slope_angle = int(sys.argv[4])
slope_azi = int(sys.argv[5])
source = "NORA3"
"""

reg_code = 3012 # int(sys.argv[1])
h_low = 300  # int(sys.argv[2])
h_hi = 600  # int(sys.argv[3])
slope_angle = 0  # int(sys.argv[4])
slope_azi = 0  # int(sys.argv[5])
source = "NORA3"


#%% fill the model, scen, and period variables with empty strings if source is NORA3
if source == "NORA3":
    model = ""
    scen = ""
    period = ""
    per_str = ""
    yrs1 = [1970, 1985, 2000, 2015]
    yrs2 = [1985, 2000, 2015, 2024]
elif source == "NorCP":
    model = "GFDL-CM3"  # args.model
    scen = "_historical"  # "_" + args.scen
    period = ""  # args.period

    # set up the period string
    per_str = f"_{period}" if len(period) > 0 else ""

    yrs1 = [1985, 1992, 1999]
    yrs2 = [1992, 1999, 2005]
# end if elif


#%% generate the strings based on slope and aspect
"""
slope_str = "Flat"
if slope_angle > 0:
    slope_str = "Slope"
# end if
aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

if slope_angle == 0:
    aspect = ""
# end if
"""

#%% generate the strings based on slope and aspect
slope_path = "Flat"
if slope_angle > 0:
    aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

    slope_path = f"{slope_angle}" + aspect
# end if

if slope_angle == 0:
    aspect = ""
# end if


#%% set paths
data_path = f"{path_par}/IMPETUS/{source}/Snowpack/Timeseries/Daily/{model}{scen}{per_str}/{slope_path}/" + \
                                                                                              f"Between{h_low}_{h_hi}m/"


#%% loop over the periods and load the files
df_l = []
for sta_yr, end_yr in zip(yrs1, yrs2):

    # set paths
    temp_path = f"{data_path}/{sta_yr}_{end_yr}/"


    #% load the data
    fn = f"{regions[reg_code]}_SNOWPACK_Stability_TimeseriesDaily_Between{h_low}_{h_hi}m_{slope_path}.csv"
    df_l.append(pd.read_csv(temp_path + fn, index_col=0, parse_dates=True))

# end for sta_yr, end_yr


#%% merge the dataframes
df = pd.concat(df_l, axis=0)


#%% store the data
fn = f"{regions[reg_code]}_SNOWPACK_Stability_TimeseriesDaily_Between{h_low}_{h_hi}m_{slope_path}.csv"
df.to_csv(data_path + fn)
