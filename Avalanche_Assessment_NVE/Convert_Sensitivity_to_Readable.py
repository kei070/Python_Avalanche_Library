#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the sensitivity values from the downloaded to the readible values.
"""


#%% imports
import pandas as pd
import pylab as pl
import numpy as np
from ava_functions.Lists_and_Dictionaries.Paths import obs_path


#%% regions dictionary
regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}
regions_pl = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Tromsø", 3012:"Sør-Troms", 3013:"Indre Troms"}


#%% load the .csv with Pandas
f_path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"
f_name = "Avalanche_Problems_Reduced.csv"

df = pd.read_csv(f_path + f_name, header=0)  # , low_memory=False)


#%% convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"])

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.set_index("date", inplace=True)


#%% set up a replacement matrix
replacement_dict = {10:1, 20:2, 30:3, 40:4, 45:4, 50:4}


#%% employ the replacement matrix on the avalanche problems list
df = df.replace(replacement_dict)


#%% store the converted file
df.to_csv(f_path + "Avalanche_Problems_Final.csv")


#%% get the unique names of the avalanche problems from the original file
f_orig = "Avalanche_Problems_List.csv"
ava_ps = np.unique(np.array(pd.read_csv(f_path + f_orig, nrows=1, index_col=0, header=None).dropna(axis=1)))


#%% for each avalanche problem count for how many days it exists
avap_count = {}
reg_codes = [3009, 3010, 3011, 3012, 3013]
for reg_c in reg_codes:
    avap_count[reg_c] = {}
    for ava_p in ava_ps:
        # set the column name
        col_name = ava_p + "_sensitivity"  # arbitrarily take size as parameter; this should be irrelevant (but check!)

        # extract the region
        df_reg = df[df.region == reg_c]

        # count the number of days with the given avalanche problem
        avap_count[reg_c][ava_p] = len(df_reg[col_name].dropna(axis=0))

        # print(col_name + " " + str(reg_c))
        # print(len(df_reg[col_name].dropna(axis=0)))
        # print("")
    # end for ava_p
#  end for reg_c


#%% plot
ylim = (0, 950)

fig = pl.figure(figsize=(10, 10))

ax00 = fig.add_subplot(321)
ax01 = fig.add_subplot(322)
ax10 = fig.add_subplot(323)
ax11 = fig.add_subplot(324)
ax20 = fig.add_subplot(325)

regc = 3009
ax00.bar(avap_count[regc].keys(), avap_count[regc].values())
ax00.set_title(regions_pl[regc])
ax00.set_ylabel("Number of Days")
ax00.set_ylim(ylim)
ax00.set_xticklabels([])

regc = 3010
ax01.bar(avap_count[regc].keys(), avap_count[regc].values())
ax01.set_title(regions_pl[regc])
# ax01.set_ylabel("Number of Days")
ax01.set_ylim(ylim)
ax01.set_xticklabels([])
ax01.set_yticklabels([])

regc = 3011
ax10.bar(avap_count[regc].keys(), avap_count[regc].values())
ax10.set_title(regions_pl[regc])
ax10.set_ylabel("Number of Days")
ax10.set_ylim(ylim)
ax10.set_xticklabels([])

regc = 3012
ax11.bar(avap_count[regc].keys(), avap_count[regc].values())
ax11.set_title(regions_pl[regc])
# ax11.set_ylabel("Number of Days")
ax11.set_ylim(ylim)
ax11.set_xticklabels(avap_count[regc].keys(), rotation=30)
ax11.set_yticklabels([])

regc = 3013
ax20.bar(avap_count[regc].keys(), avap_count[regc].values())
ax20.set_title(regions_pl[regc])
ax20.set_ylabel("Number of Days")
ax20.set_ylim(ylim)
ax20.set_xticklabels(avap_count[regc].keys(), rotation=30)

fig.subplots_adjust(wspace=0.05)

pl.show()
pl.close()