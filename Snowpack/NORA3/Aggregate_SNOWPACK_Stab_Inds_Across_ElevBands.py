#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the SNOWPACK stability indices from several elevation bands (and aspects) by selecting the most extreme index
per region across the elevation bands and aspects.
"""


#%% import
import os
import glob
import numpy as np
import pandas as pd

from ava_functions.Lists_and_Dictionaries.Region_Codes import regions
from ava_functions.Lists_and_Dictionaries.Paths import path_par


#%% test mode?
# --> if this is true dataset that do not fit the required length will be considered anyway and it will later be checked
#     where the datasets do not overlap (that is, if in fact they do not overlap 100%)
test_mode = False


#%% parameters
reg_code = 3012

agg_slope = False

slope_angle = 0
slope_azi = 0


#%% set the number of expected dataset
if reg_code in [3011, 3012]:
    n_dfs = 3
else:
    n_dfs = 4
# end if


#%% generate the strings based on slope and aspect
slope_path = ""
fn_suff = "ElevSlopeAgg"
if not agg_slope:
    fn_suff = "ElevAgg"
    slope_path = "Flat"
    if slope_angle > 0:
        aspect = {0:"_N", 90:"_E", 180:"_S", 270:"_W"}[slope_azi]

        slope_path = f"{slope_angle}" + aspect
    # end if

    if slope_angle == 0:
        aspect = ""
    # end if
# end if


#%% data paths
data_path = f"{path_par}/IMPETUS/NORA3/Snowpack/Timeseries/Daily/{slope_path}/"  # "Between{h_low}_{h_hi}m/"


#%% list the subdirectories
# --> the list will include all available slopes as well as all available elevation bands
subfolders = []

if agg_slope:
    for f1 in os.scandir(data_path):
        if os.path.isdir(f1.path):
            subfolders += [f.path for f in os.scandir(f1) if os.path.isdir(f)]
        # end if
    # end for f1
else:
    subfolders = [f.path for f in os.scandir(data_path) if os.path.isdir(f)]
# end if else


#%% loop over the files
df_list = []
for subdir in subfolders:

    try:
        df = pd.read_csv(glob.glob(subdir + f"/{regions[reg_code]}_*.csv")[0], index_col=0, parse_dates=True)
        if test_mode:
            df_list.append(df)
        else:
            if len(df) != 19632:
                print(f"Length {'/'.join(subdir.split('/')[-2:])} for {regions[reg_code]} is not 19632. " +
                      "It is disgarded...\n")
                continue
            else:
                df_list.append(df)
            # end if else
        # end if else

    except:
        print(f"\nNo {'/'.join(subdir.split('/')[-2:])} data available for {regions[reg_code]}. Continuing...\n")

# end for subdir


#%% depending on the mode (test or not) execute the following actions
if test_mode:
    # check discrepancies between datasets that hinder concatenation
    array1 = np.array(df_list[0].index)
    array2 = np.array(df_list[1].index)

    intersection = np.intersect1d(array1, array2)
    # Find elements in array1 but not in the intersection
    not_in_intersection_1 = np.setdiff1d(array1, intersection)
    # Find elements in array2 but not in the intersection
    not_in_intersection_2 = np.setdiff1d(array2, intersection)

    if len(not_in_intersection_1) + len(not_in_intersection_2) != 0:
        print(f"\n{reg_code} has missing data...\n")
    else:
        print(f"\nUsing {len(df_list)} datasets with length {len(df_list[0])}. Should be {n_dfs}\n\n")
    # end if else

    print("Elements in 1 but not in the intersection:")
    print(not_in_intersection_1)
    print("\nElements in 2 but not in the intersection:")
    print(not_in_intersection_2)

else:

    #% get the indices
    indfull_dict = {}
    index_dict = {}
    for k in df.columns:
        temp = np.array([dfk[k] for dfk in df_list])

        indfull_dict[k] = temp

        if k in ["snow_depth", "snow_depth3", "snow_depth7",
                 "snow_depth_d1", "snow_depth3_d1", "snow_depth7_d1",
                 "snow_depth_d2", "snow_depth3_d2", "snow_depth7_d2",
                 "snow_depth_d3", "snow_depth3_d3", "snow_depth7_d3"]:  # for snow depth take the maximum and minimum
            index_dict[k + "_emax"] = np.max(temp, axis=0)
            index_dict[k + "_emin"] = np.min(temp, axis=0)
        elif k in ["lwc_i", "lwc_sum", "lwc_max", "lws_s_top", "t_top",
                   "lwc_i_d1", "lwc_sum_d1", "lwc_max_d1", "lws_s_top_d1", "t_top_d1",
                   "lwc_i_d2", "lwc_sum_d2", "lwc_max_d2", "lws_s_top_d2", "t_top_d2",
                   "lwc_i_d3", "lwc_sum_d3", "lwc_max_d3", "lws_s_top_d3", "t_top_d3"]:
            index_dict[k] = np.max(temp, axis=0)
        else:  # take the index minimum
            index_dict[k] = np.min(temp, axis=0)
        # end if else

    # end for k


    #% generate a dataframe from the indices
    out_df = pd.DataFrame(index_dict, index=df.index.date)


    #% store the data
    out_df.to_csv(data_path + f"{regions[reg_code]}_SNOWPACK_Stability_TimeseriesDaily_{fn_suff}.csv")

# end if else