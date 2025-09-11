#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the SNOWPACK stability indices from several elevation bands (and aspects) by selecting the most extreme index
per region across the elevation bands and aspects.

NOTE THE DIFFERENCE TO THE NORA3 SCRIPT: Here there is not (yet) the possibility to aggregate across elevation bands AND
                                         slopes as there is for NORA3. The reason behind this is that in the evaluation
                                         of the NORA3 results this was not shown to yield a benefit to the machine-
                                         learning model performance. Thus, it is not implemented for the NorCP data.
"""


#%% import
import os
import sys
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

reg_code = 3009
model = "EC-Earth"
scen = "rcp85"
period = "_MC"
"""
reg_code = int(sys.argv[1])  # 3010 #
model = sys.argv[2]  # "EC-Earth" #
scen = sys.argv[3]  # "rcp45" #
period = sys.argv[4]  # "_MC" #
"""

#%% set the number of expected dataset
if reg_code in [3011, 3012]:
    n_dfs = 3
else:
    n_dfs = 4
# end if


#%% set up the period-years depending on the scenario and period to be able to complete the path
if scen == "historical":
    per_yrs = "1985_2005"
else:
    if period == "_MC":
        per_yrs = "2040_2060"
    elif period == "_LC":
        per_yrs = "2080_2100"
    # end if elif
# end if else


#%% data paths
data_path = f"{path_par}/IMPETUS/NorCP/Snowpack/Timeseries/Daily/{model}_{scen}{period}/Flat/"  # "Between{h_low}_{h_hi}m/"


#%% list the subdirectories
subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]


#%% loop over the files
df_list = []
for subdir in subfolders:
    # print(subdir)
    try:
        df = pd.read_csv(glob.glob(subdir + f"/{per_yrs}/{regions[reg_code]}_*.csv")[0], index_col=0, parse_dates=True)
        if test_mode:
            df_list.append(df)
        else:
            if len(df) not in [7212, 7213]:
                print(f"Length {'/'.join(subdir.split('/')[-2:])} for {regions[reg_code]} is not 7212 or 7213. " +
                      "It is disgarded...\n")
                continue
            else:
                df_list.append(df)
            # end if else
        # end if else
    except:
        print(f"\nNo {subdir.split('/')[-1]} data available for {regions[reg_code]}. Continuing...\n")
    # end try except
# end for subdir


#%% get the indices
"""
indfull_dict = {}
index_dict = {}
for k in df.columns:
    temp = np.array([dfk[k] for dfk in df_list])

    indfull_dict[k] = temp

    if k in ["snow_depth", "snow_depth3", "snow_depth7"]:  # for snow depth take the maximum and minimum
        index_dict[k + "_emax"] = np.max(temp, axis=0)
        index_dict[k + "_emin"] = np.min(temp, axis=0)
    else:  # take the index minimum
        index_dict[k] = np.min(temp, axis=0)
    # end if else

# end for k
"""

#%% depending on the mode (test or not) execute the following actions
if test_mode:
    # check discrepancies between datasets that hinder concatenation
    array1 = np.array(df_list[0].index)
    array2 = np.array(df_list[-1].index)

    intersection = np.intersect1d(array1, array2)
    # Find elements in array1 but not in the intersection
    not_in_intersection_1 = np.setdiff1d(array1, intersection)
    # Find elements in array2 but not in the intersection
    not_in_intersection_2 = np.setdiff1d(array2, intersection)

    if len(not_in_intersection_1) + len(not_in_intersection_2) != 0:
        print(f"\n{model} {scen} {period} has missing data in {reg_code}...\n")
    else:
        print(f"\nUsing {len(df_list)} datasets with length {len(df_list[0])}. Should be {n_dfs}\n\n")
    # end if else

    """
    print("Elements in 1 but not in the intersection:")
    print(not_in_intersection_1)
    print("\nElements in 2 but not in the intersection:")
    print(not_in_intersection_2)
    """
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
            index_dict[k + "_emax"] = np.nanmax(temp, axis=0)
            index_dict[k + "_emin"] = np.nanmin(temp, axis=0)
        elif k in ["lwc_i", "lwc_sum", "lwc_max", "lws_s_top", "t_top",
                   "lwc_i_d1", "lwc_sum_d1", "lwc_max_d1", "lws_s_top_d1", "t_top_d1",
                   "lwc_i_d2", "lwc_sum_d2", "lwc_max_d2", "lws_s_top_d2", "t_top_d2",
                   "lwc_i_d3", "lwc_sum_d3", "lwc_max_d3", "lws_s_top_d3", "t_top_d3"]:
            index_dict[k] = np.nanmax(temp, axis=0)
        else:  # take the index minimum
            index_dict[k] = np.nanmin(temp, axis=0)
        # end if else

    # end for k
    #% generate a dataframe from the indices
    out_df = pd.DataFrame(index_dict, index=df_list[0].index.date)

    #% fill the NaNs with 0
    out_df.fillna(0, inplace=True)

    #% store the data
    out_df.to_csv(data_path + f"{regions[reg_code]}_{model}_{scen}{period}_SNOWPACK_Stability_TimeseriesDaily_" +
                  "ElevAgg_Flat.csv")

# end if