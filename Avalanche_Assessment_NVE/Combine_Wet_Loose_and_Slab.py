#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine the wet loose and slab avalanche problems.
"""

#%% imports
import numpy as np
import pandas as pd
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"

loose = "wet_loose_Danger_List.csv"
slab = "wet_slab_Danger_List.csv"

loose_df = pd.read_csv(f_path + loose, header=0)  # , low_memory=False)
slab_df = pd.read_csv(f_path + slab, header=0)  # , low_memory=False)


#%% adjust the NaNs of the danger levels
loose_df["danger_level"][loose_df["danger_level"].isna()] = 0
slab_df["danger_level"][slab_df["danger_level"].isna()] = 0

wet_dl = np.zeros(len(loose_df))
l_g_s = loose_df["danger_level"] >= slab_df["danger_level"]
s_g_l = slab_df["danger_level"] >= loose_df["danger_level"]
wet_dl[l_g_s] = loose_df["danger_level"][l_g_s]
wet_dl[s_g_l] = slab_df["danger_level"][s_g_l]


#%% adjust the NaNs of the expositions
expos_dict = {"N":1, "NE":2, "E":3, "SE":4, "S":5, "SW":6, "W":7, "NW":8}
wet_expos_dict = {}
for ex in expos_dict.keys():
    loose_df[ex][loose_df[ex].isna()] = False
    slab_df[ex][slab_df[ex].isna()] = False

    wet_expos = loose_df[ex] | slab_df[ex]
    wet_expos_dict[ex] = wet_expos
# end for ex


#%% adjust the NaNs of the elevations
loose_df["elev_min"][loose_df["elev_min"].isna()] = 1e9
loose_df["elev_max"][loose_df["elev_max"].isna()] = -1
slab_df["elev_min"][slab_df["elev_min"].isna()] = 1e9
slab_df["elev_max"][slab_df["elev_max"].isna()] = -1

# always take the smaller min and the larger max
wet_min = np.zeros(len(slab_df))
wet_max = np.zeros(len(slab_df))

min_l_s_s = loose_df["elev_min"] <= slab_df["elev_min"]   # include the equal case here
min_s_s_l = loose_df["elev_min"] > slab_df["elev_min"]
max_l_g_s = loose_df["elev_max"] >= slab_df["elev_max"]   # include the equal case here
max_s_g_l = loose_df["elev_max"] < slab_df["elev_max"]

wet_min[min_l_s_s] = loose_df["elev_min"][min_l_s_s]
wet_min[min_s_s_l] = slab_df["elev_min"][min_s_s_l]

wet_max[max_l_g_s] = loose_df["elev_max"][max_l_g_s]
wet_max[max_s_g_l] = slab_df["elev_max"][max_s_g_l]

wet_min[wet_min == 1e9] = np.nan
wet_max[wet_max == -1] = np.nan


#%% add all the data together into a dictionary
wet_df = pd.DataFrame({"region":loose_df["region"], "date":loose_df["date"], "danger_level":wet_dl, "elev_min":wet_min,
                       "elev_max":wet_max} | wet_expos_dict)  # --> note that the | operator merges both dictionaries


#%% store the data
wet_df.to_csv(f_path + "wet_Danger_List.csv")


