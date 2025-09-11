#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the avalanche exposures as a stacked bar plot for all regions.
"""


#%% imports
import numpy as np
import pandas as pd
import pylab as pl
from datetime import datetime

from ava_functions.Lists_and_Dictionaries.Paths import obs_path
from ava_functions.Discrete_Hist import disc_hist_data


#%% AP dictionary
a_ps = {"wind_slab":"wind slab", "new_loose":"new loose", "new_slab":"new slab",
        "pwl_slab":"PWL slab", "wet_loose":"wet loose", "wet_slab":"wet slab", "glide_slab":"glide slab"}


#%% region dictionary
regs_pl = {3009:"Nord-Troms", 3010:"Lyngen", 3011:"Tromsø", 3012:"Sør-Troms", 3013:"Indre Troms"}


#%% set parameters
sta_date = "01-12-2016"
end_date = "01-06-2024"


#%% get start and end date in datatime format
sta_date_dt = datetime.strptime(sta_date, "%d-%m-%Y")
end_date_dt = datetime.strptime(end_date, "%d-%m-%Y")


#%% paths
path = f"{obs_path}/IMPETUS/Avalanches_Danger_Files/"


#%% load a file
dfs = {}
expos_dfs = {}
exp_freqs_dfs = {}
for a_p in a_ps.keys():

    # load the data
    df = pd.read_csv(path + f"{a_p}_Danger_List.csv")

    # extract the northern region
    reg_dfs = {}
    for reg_code in regs_pl.keys():
        df_reg = df[df["region"] == reg_code]
        df_reg["date"] = pd.to_datetime(df_reg["date"])
        df_reg.set_index("date", inplace=True)

        reg_dfs[reg_code] = df_reg
    # end for reg_code

    dfs[a_p] = reg_dfs

    # loop over all exposition columns and turn NaN to False
    # --> one exposition column with values from 1 to 8
    expos_dict = {"N":1, "NE":2, "E":3, "SE":4, "S":5, "SW":6, "W":7, "NW":8}
    expos = {}
    for reg_code in regs_pl.keys():
        expos[reg_code] = np.zeros(len(reg_dfs[reg_code]))
        # --> IMPORTANT: This means that the first element in the final frequency array corresponds to the number of
        #                time where there was no exposure given!

        for k in expos_dict:
            reg_dfs[reg_code][k][reg_dfs[reg_code][k].isna()] = False

            # convert the expositions so as to be able to plot them
            expos[reg_code][reg_dfs[reg_code][k].astype(bool)] = expos_dict[k]
        # end for k
    # end for reg_code
    expos_dfs[a_p] = expos

    # get the indices
    ind_rs = {}
    for reg in regs_pl:
        ind_rs[reg] = ((reg_dfs[reg].index > sta_date_dt) & (reg_dfs[reg].index < end_date_dt))
    # end for reg
    ind_rl = list(ind_rs.values())

    # calculate the exposure frequencies
    exp_freqs = disc_hist_data([expos[reg][ind_r] for reg, ind_r in zip(regs_pl, ind_rl)], classes=np.arange(9))[1]
    # --> IMPORTANT: 9 classes, because class 0 corresponds to no exposure, i.e., the avalanche problem was not forecast

    exp_freqs_dfs[a_p] = exp_freqs

# end for a_p


#%% calculate the total frequency
tot_freq = dict()
for i, reg_code in enumerate(regs_pl.keys()):
    tot_freq[i] = np.zeros(9)
    for a_p in a_ps.keys():

        tot_freq[i] += np.array(exp_freqs_dfs[a_p][i])

    # end for a_p
# end for i, reg_code


#%% stacked bar plot
pl_path = "/home/kei070/Documents/IMPETUS/Publishing/The Cryosphere/Avalanche_Paper_2/00_Figures/"

a_p1 = "wind_slab"
a_p2 = "pwl_slab"
a_p3 = "wet_slab"
a_p4 = "wet_loose"
a_p5 = "new_slab"
a_p6 = "new_loose"
a_p7 = "glide_slab"

colors = ["black", "gray", "red", "blue", "brown", "orange", "violet", "lightblue"]

fig = pl.figure(figsize=(8, 13))
ax00 = fig.add_subplot(421)
ax01 = fig.add_subplot(422)
ax10 = fig.add_subplot(423)
ax11 = fig.add_subplot(424)
ax20 = fig.add_subplot(425)
ax21 = fig.add_subplot(426)
ax30 = fig.add_subplot(427)
ax31 = fig.add_subplot(428)

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax00.bar(x=x, height=exp_freqs_dfs[a_p1][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p1][x][i]
    # end for i
# end for x

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax01.bar(x=x, height=exp_freqs_dfs[a_p2][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p2][x][i]
    # end for i
# end for x

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax10.bar(x=x, height=exp_freqs_dfs[a_p3][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p3][x][i]
    # end for i
# end for x

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax11.bar(x=x, height=exp_freqs_dfs[a_p4][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p4][x][i]
    # end for i
# end for x

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax20.bar(x=x, height=exp_freqs_dfs[a_p5][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p5][x][i]
    # end for i
# end for x

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax21.bar(x=x, height=exp_freqs_dfs[a_p6][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p6][x][i]
    # end for i
# end for x

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax30.bar(x=x, height=exp_freqs_dfs[a_p7][x][i], bottom=bottom, color=colors[i-1])
        bottom += exp_freqs_dfs[a_p7][x][i]
    # end for i
# end for x

# LEGEND
for i in np.arange(1, 9, 1):
    ax30.plot(np.zeros(8), np.zeros(8), color=colors[i-1], label=list(expos_dict.keys())[i-1], linewidth=7)
# end for i

ax30.legend(ncol=2, loc="upper left")
# ax31.axis("off")

for x in np.arange(len(regs_pl)):
    bottom = 0
    for i in np.arange(1, 9, 1):
        ax31.bar(x=x, height=tot_freq[x][i], bottom=bottom, color=colors[i-1])
        bottom += tot_freq[x][i]
    # end for i
# end for x

ax21.set_ylim((0, 14))

ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax10.set_xticklabels([])
ax11.set_xticklabels([])
ax20.set_xticklabels([])
ax21.set_xticklabels([])

ax30.set_xticks(np.arange(len(regs_pl)))
ax30.set_xticklabels(regs_pl.values(), rotation=10)
ax31.set_xticks(np.arange(len(regs_pl)))
ax31.set_xticklabels(regs_pl.values(), rotation=10)

ax00.set_ylabel("Frequency of occurrence")
ax10.set_ylabel("Frequency of occurrence")
ax20.set_ylabel("Frequency of occurrence")
ax30.set_ylabel("Frequency of occurrence")

ax00.set_title("Exposures " + a_ps[a_p1])
ax01.set_title("Exposures " + a_ps[a_p2])
ax10.set_title("Exposures " + a_ps[a_p3])
ax11.set_title("Exposures " + a_ps[a_p4])
ax20.set_title("Exposures " + a_ps[a_p5])
ax21.set_title("Exposures " + a_ps[a_p6])
ax30.set_title("Exposures " + a_ps[a_p7])
ax31.set_title("All exposures")

pl.savefig(pl_path + "02_Fig03_Exposure_Frequency.pdf", bbox_inches="tight", dpi=200)

pl.show()
pl.close()


