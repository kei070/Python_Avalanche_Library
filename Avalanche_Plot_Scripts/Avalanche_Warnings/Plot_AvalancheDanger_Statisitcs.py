#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot some avalanche danger statistics to find
    - the most frequently identified AP
    - the AP with the on average highest danger level
    - the distribution for the 2-level case
    - the distribution for the 4-level case
"""

#%% imports
import os
import sys
import numpy as np
import pandas as pd
import pylab as pl

from ava_functions.Lists_and_Dictionaries.Paths import path_par, obs_path
from ava_functions.Lists_and_Dictionaries.Region_Codes import regions


#%% set plot path
pl_path = f"{obs_path}/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"


#%% get all the danger-level "observations"
dl_path = f"/{obs_path}/IMPETUS/Avalanches_Danger_Files/"
dl_name = "Avalanche_Danger_List.csv"

dl_df = pd.read_csv(dl_path + dl_name)
dl_df.date = pd.to_datetime(dl_df.date)


#%% load the danger levels per avalanche problem
# ap_name = "Avalanche_Problems_ADL.csv"
ap_name = "Avalanche_Problems_ADL_Extended.csv"  # including the "wet" problem (loose & slab combined)

ap_df = pd.read_csv(dl_path + ap_name)
ap_df.date = pd.to_datetime(ap_df.date)


#%% extract the region according to the region code
dl_df_reg = []
ap_df_reg = []
for reg_code in regions.keys():
    dl_df_reg.append(dl_df[dl_df.region == reg_code])
    ap_df_reg.append(ap_df[ap_df.region == reg_code])
# end for reg_code

dl_df_reg = pd.concat(dl_df_reg)
ap_df_reg = pd.concat(ap_df_reg)


#%% drop the ADL from the ADL file and add the danger level from the dedicated dataset
ap_df_reg.drop("ADL", axis=1, inplace=True)

ap_df_reg["danger_level"] = dl_df_reg["danger_level"]


#%% convert the original levels to the binary classification
ap_df_reg_bin = {k:np.zeros(len(ap_df_reg)) for k in ap_df_reg.columns}

ap_df_reg_bin["date"] = ap_df_reg["date"]
ap_df_reg_bin["region"] = ap_df_reg["region"]


for ap in list(ap_df_reg.columns):

    if ap in ["date", "region"]:
        continue
    # end if

    ap_df_reg_bin[ap][ap_df_reg[ap] > 2] = 1

# end for ap

ap_df_reg_bin = pd.DataFrame(ap_df_reg_bin)
ap_df_reg_bin.set_index("date", inplace=True)


#%% also set the index of the original dataframe and convert the NanNs to zeros
ap_df_reg.set_index("date", inplace=True)
ap_df_reg.fillna(0, inplace=True)


#%% count the number of days a specific AP is identified per region
#   also calculate the average danger level per AP --> ONLY ON THE DAYS THAT PROBLEM IS IDENTIFIED!!
ap_freq = {k:{} for k in regions.keys()}
ap_freq["total"] = {}
avg_dl = {k:{} for k in regions.keys()}
avg_dl["total"] = {}
for ap in list(ap_df_reg.columns):
    if ap in ["date", "region"]:
        continue
    # end if
    ap_freq["total"][ap] = np.sum(ap_df_reg[ap] > 0)
    avg_dl["total"][ap] = np.mean(ap_df_reg[ap][ap_df_reg[ap] > 0])

    for reg_code in regions.keys():
        ap_freq[reg_code][ap] = np.sum(ap_df_reg[ap][ap_df_reg["region"] == reg_code] > 0)
        avg_dl[reg_code][ap] = np.mean(ap_df_reg[ap][ap_df_reg["region"] == reg_code][ap_df_reg[ap][ap_df_reg["region"]\
                                                                                                    == reg_code] > 0])
    # end for reg_code
# end for ap


#%% plot
aps_pl = ["Glide\nslab", "New\nloose", "New\nslab", "PWL\nslab", "Wet\nloose", "Wet\nslab", "Wind\nslab", "Wet",
          "ADL"]

fig = pl.figure(figsize=(4, 3))
ax0 = fig.add_subplot(111)
ax1 = ax0.twinx()  # fig.add_subplot(212)

p00, = ax0.plot(np.arange(len(ap_freq["total"].keys())), ap_freq["total"].values(), marker="o", c="black",
                linewidth=0.25, label="# of days")
p01, = ax1.plot(np.arange(len(ap_freq["total"].keys())), avg_dl["total"].values(), marker="s", c="red", linewidth=0.25,
                label="Average DL")

ax0.legend(handles=[p00, p01])

ax0.set_ylabel("Number of days")
ax1.set_ylabel("Average danger level", color="red")

# ax0.set_xticklabels([])
ax0.set_xticks(np.arange(len(ap_freq["total"].keys())))
ax0.set_xticklabels(aps_pl)

ax0.spines['right'].set_color('red')
ax1.spines['right'].set_color('red')
ax1.tick_params(axis='y', colors='red')

ax0.set_title("Avalanche danger level statistics")

fig.subplots_adjust(hspace=0.1)

pl.savefig(pl_path + "ADL_AP_Statisitics.pdf", dpi=200, bbox_inches="tight")

pl.show()
pl.close()







