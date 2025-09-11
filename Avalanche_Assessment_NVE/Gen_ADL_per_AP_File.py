#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the avalanche danger levels PER avalanche problem based on the Bavarian matrix.
--> the current version creates both a single file containing the danger level of all avalanche problems as well as
    (redundant) files for each individual avalanche problem which have the same format as the general danger level file
"""

#%% imports
import os
import sys
import numpy as np
import pandas as pd
import pylab as pl
from ava_functions.Lists_and_Dictionaries.Paths import obs_path
from ava_functions.AvaProb_to_ADL import ap_to_adl


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
f_name = "Avalanche_Problems_Final.csv"

df = pd.read_csv(f_path + f_name, header=0, index_col=0)  # , low_memory=False)


#%% extract only 3009 and only 2016
# df1 = df[df["region"] == 3009]
# df1.index = pd.to_datetime(df1.index)
# df1 = df1[df1.index.year < 2017]


#%% get the unique names of the avalanche problems from the original file
f_orig = "Avalanche_Problems_List.csv"
ava_ps = np.unique(np.array(pd.read_csv(f_path + f_orig, nrows=1, index_col=0, header=None).dropna(axis=1)))


#%% loop over the avalanche problems and generate the danger levels
ava_p_dl = {"region":df["region"]}
for ava_p in ava_ps:
    sens_s = np.array(df[ava_p + "_sensitivity"])
    dist_s = np.array(df[ava_p + "_distribution"])
    size_s = np.array(df[ava_p + "_size"])

    adl_s = ap_to_adl(sens_s, dist_s, size_s)

    ava_p_dl[ava_p] = adl_s
# end for ava_p


#%% convert to dataframe add a date column
adl_df = pd.DataFrame(data=ava_p_dl, index=df.index)

# add the general danger level as a column (= max of all columns)
adl_df["ADL"] = adl_df[list(adl_df.columns)[1:]].max(axis=1)


#%% store the dataframe
adl_df.to_csv(f_path + "Avalanche_Problems_ADL.csv")


#%% to make the statistical model application easier generate one file per avalanche problem using the exact same
#   structure as the general ADL file
for ava_p in ava_ps:
    ap_df = pd.DataFrame({"region":df["region"],
                          "date":df.index,
                          "danger_level":ava_p_dl[ava_p],
                          "elev_min":df[ava_p + "_elevation_min"],
                          "elev_max":df[ava_p + "_elevation_max"],
                          "N":df[ava_p + "_exposition_N"],
                          "NE":df[ava_p + "_exposition_NE"],
                          "E":df[ava_p + "_exposition_E"],
                          "SE":df[ava_p + "_exposition_SE"],
                          "S":df[ava_p + "_exposition_S"],
                          "SW":df[ava_p + "_exposition_SW"],
                          "W":df[ava_p + "_exposition_W"],
                          "NW":df[ava_p + "_exposition_NW"]}).reset_index(drop=True)
    ap_df.to_csv(f_path + ava_p + "_Danger_List.csv", index=False)
# end for ava_p


#%% generate a dataframe with ADLs based on the APs --> simply take the maximum ADL from the APs
"""
gen_adl_from_ap = pd.DataFrame(data={"region":adl_df["region"], "ADL":adl_df[ava_ps].max(axis=1)}, index=adl_df.index)


#%% load the general ADL file
gen_adl = pd.read_csv(f_path + "Avalanche_Danger_List.csv")
gen_adl.set_index("date", inplace=True)


#%% compare the ADLs
reg_c = 3009

reg_adl = gen_adl[gen_adl["region"] == reg_c]
reg_adl_f_ap = gen_adl_from_ap[gen_adl_from_ap["region"] == reg_c]


#%% plot
fig = pl.figure()
ax00 = fig.add_subplot(111)

ax00.plot(reg_adl["danger_level"] - reg_adl_f_ap["ADL"])

pl.show()
pl.close()
"""